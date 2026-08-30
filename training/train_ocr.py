#!/usr/bin/env python3
"""Train the CRNN+CTC OCR model with PyTorch Lightning (spec §6, §14).

Recommended two-stage recipe (spec §12):
    # 1. pretrain on synthetic plates only
    python training/train_ocr.py --stage pretrain

    # 2. fine-tune on real crops (+ synthetic mixed in), warm-started from stage 1
    python training/train_ocr.py --stage finetune --init models/ocr/v1/ocr_crnn_pretrained.pt

Single-stage from scratch on real data:
    python training/train_ocr.py --stage scratch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from tz_alpr.ocr.datamodule import OcrDataModule
from tz_alpr.ocr.lightning_module import OcrLitModule


def _load_cfg(path: Path, stage: str, max_epochs: int | None) -> dict:
    cfg = yaml.safe_load(path.read_text())
    pre = cfg.get("pretrain", {})
    if stage == "pretrain":
        cfg["train"]["max_epochs"] = pre.get("max_epochs", cfg["train"]["max_epochs"])
        cfg["train"]["lr"] = pre.get("lr", cfg["train"]["lr"])
        cfg["train"]["min_epochs"] = pre.get("min_epochs", 1)
    if max_epochs:
        cfg["train"]["max_epochs"] = max_epochs
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs/ocr.yaml")
    ap.add_argument("--training-config", type=Path, default=REPO_ROOT / "configs/training.yaml")
    ap.add_argument("--stage", choices=["pretrain", "finetune", "scratch"], default="scratch")
    ap.add_argument("--init", type=Path, default=None, help="warm-start weights (.pt/.ckpt)")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--precision", default=None, help="override train.precision (e.g. 32, 16-mixed)")
    ap.add_argument("--num-workers", type=int, default=None, help="override data.num_workers")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit-train-batches", type=float, default=None)
    ap.add_argument("--fast-dev-run", action="store_true")
    args = ap.parse_args()

    tcfg = yaml.safe_load(args.training_config.read_text())
    L.seed_everything(tcfg.get("seed", 1337), workers=True)

    cfg = _load_cfg(args.config, args.stage, args.max_epochs)
    if args.precision:
        cfg["train"]["precision"] = args.precision
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    use_synth = args.stage in ("pretrain", "finetune")
    synth_only = args.stage == "pretrain"

    dm = OcrDataModule(cfg, use_synthetic=use_synth, synthetic_only=synth_only)
    model = OcrLitModule(cfg)

    if args.init and args.init.exists():
        state = torch.load(str(args.init), map_location="cpu")
        state = state.get("state_dict", state)
        cleaned = {k.removeprefix("model."): v for k, v in state.items()}
        missing, unexpected = model.model.load_state_dict(cleaned, strict=False)
        print(f"warm-start: missing={len(missing)} unexpected={len(unexpected)}")

    t = cfg["train"]
    monitor = t.get("monitor", "val/char_acc")
    mode = t.get("monitor_mode", "max")
    min_epochs = int(t.get("min_epochs", 1))
    ckpt_cb = ModelCheckpoint(
        dirpath=tcfg["checkpoint"]["dirpath"],
        filename=f"ocr-{args.stage}-{{epoch:02d}}-{{val/char_acc:.4f}}",
        monitor=monitor,
        mode=mode,
        save_top_k=tcfg["checkpoint"]["save_top_k"],
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [
        ckpt_cb,
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=t.get("early_stop_patience", 15),
            # `min_epochs` on the Trainer already blocks stopping early; this is a
            # second guard so a noisy metric during warm-up can't trip it.
            check_finite=True,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=t["max_epochs"],
        min_epochs=min_epochs,
        precision=t.get("precision", "16-mixed"),
        accelerator=tcfg.get("accelerator", "auto"),
        devices=tcfg.get("devices", "auto"),
        gradient_clip_val=t.get("grad_clip", 5.0),
        accumulate_grad_batches=t.get("accumulate_grad_batches", 1),
        logger=[
            TensorBoardLogger(save_dir="lightning_logs", name=f"ocr_{args.stage}"),
            CSVLogger(save_dir="lightning_logs", name=f"ocr_{args.stage}"),
        ],
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches or 1.0,
        log_every_n_steps=25,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=str(args.resume) if args.resume else None)

    if args.fast_dev_run:
        return

    dm.setup("test")
    if len(dm.test_dataloader().dataset):
        trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path="best")
        trainer.test(model, dataloaders=dm.hard_test_dataloader(), ckpt_path="best")

    best = OcrLitModule.load_from_checkpoint(ckpt_cb.best_model_path, cfg=cfg)
    suffix = "pretrained" if args.stage == "pretrain" else "crnn"
    out = REPO_ROOT / f"models/ocr/v1/ocr_{suffix}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best.model.state_dict(), out)
    print(f"\nBest checkpoint: {ckpt_cb.best_model_path}")
    print(f"Runtime weights: {out}")
    print(f"Export ONNX   : python tools/export_onnx.py ocr --ckpt {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()
