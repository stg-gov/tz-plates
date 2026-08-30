"""LightningModule wrapping the CRNN + CTC loss (spec §14)."""

from __future__ import annotations

import lightning as L
import torch
from torch import nn

from tz_alpr.ocr.charset import Charset, load_charset
from tz_alpr.ocr.ctc_decode import greedy_decode
from tz_alpr.ocr.metrics import score_batch
from tz_alpr.ocr.model import build_crnn


class OcrLitModule(L.LightningModule):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.charset: Charset = load_charset()
        self.model = build_crnn(cfg["model"], num_classes=self.charset.size)
        self.criterion = nn.CTCLoss(blank=self.charset.blank_index, zero_infinity=True)
        self._val_preds: list[str] = []
        self._val_gts: list[str] = []

    # ------------------------------------------------------------------ forward
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)  # (T, B, C) log-probs

    def _ctc_loss(self, log_probs: torch.Tensor, batch: dict) -> torch.Tensor:
        t, b, _ = log_probs.shape
        input_lengths = torch.full((b,), t, dtype=torch.long, device=log_probs.device)
        return self.criterion(
            log_probs, batch["targets"], input_lengths, batch["target_lengths"]
        )

    # ----------------------------------------------------------------- training
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        log_probs = self(batch["images"])
        loss = self._ctc_loss(log_probs, batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # --------------------------------------------------------------- validation
    def validation_step(self, batch: dict, batch_idx: int) -> None:
        log_probs = self(batch["images"])
        loss = self._ctc_loss(log_probs, batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

        lp = log_probs.permute(1, 0, 2).detach().float().cpu().numpy()  # (B, T, C)
        for i, gt in enumerate(batch["texts"]):
            decoded = greedy_decode(lp[i], self.charset, topk=2)
            self._val_preds.append(decoded.text)
            self._val_gts.append(gt)

    def on_validation_epoch_end(self) -> None:
        scores = score_batch(self._val_preds, self._val_gts)
        for k, v in scores.as_dict("val").items():
            self.log(k, v, prog_bar=k.endswith("seq_acc"))
        self._val_preds.clear()
        self._val_gts.clear()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        scores = score_batch(self._val_preds, self._val_gts)
        for k, v in scores.as_dict("test").items():
            self.log(k, v)
        self._val_preds.clear()
        self._val_gts.clear()

    # --------------------------------------------------------------- optimizers
    def configure_optimizers(self):
        t = self.cfg["train"]
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(t.get("lr", 1e-3)),
            weight_decay=float(t.get("weight_decay", 1e-4)),
        )
        sched_name = t.get("scheduler", "onecycle")
        if sched_name == "onecycle":
            steps = self.trainer.estimated_stepping_batches
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=float(t.get("lr", 1e-3)), total_steps=max(1, steps)
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=int(t.get("max_epochs", 60))
            )
            return {"optimizer": opt, "lr_scheduler": scheduler}
        return opt
