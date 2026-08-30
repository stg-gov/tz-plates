"""LightningDataModule for OCR training (spec §14)."""

from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from tz_alpr.ocr.charset import Charset, load_charset
from tz_alpr.ocr.dataset import (
    OcrDataset,
    OcrSample,
    ctc_collate,
    dedupe_by_image,
    load_samples_from_csv,
)
from tz_alpr.ocr.transforms import build_eval_transform, build_train_transform


class OcrDataModule(L.LightningDataModule):
    def __init__(self, cfg: dict, use_synthetic: bool = True, synthetic_only: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg["data"]
        self.charset: Charset = load_charset()
        self._use_synth = use_synthetic
        self._synth_only = synthetic_only
        self._train_tf = build_train_transform(cfg)
        self._eval_tf = build_eval_transform(cfg)
        self._sets: dict[str, OcrDataset] = {}

    # -------------------------------------------------------------------- setup
    def setup(self, stage: str | None = None) -> None:
        d = self.data_cfg
        real_csv = Path(d["ocr_annotations_csv"])
        synth_csv = Path(d.get("synthetic_csv", ""))
        min_conf = float(d.get("min_det_conf", 0.0))

        train_samples: list[OcrSample] = []
        if not self._synth_only and real_csv.exists():
            train_samples += load_samples_from_csv(
                real_csv, split="train", source_tag="real", min_det_conf=min_conf
            )
        if self._use_synth and synth_csv.exists():
            train_samples += load_samples_from_csv(synth_csv, split=None, source_tag="synthetic")
        if not train_samples:
            raise FileNotFoundError(
                f"No OCR training samples. Expected {real_csv} (run tools/prepare_dataset.py) "
                f"or synthetic data at {synth_csv} (run tools/generate_tanzania_plates.py)."
            )

        val_samples = (
            load_samples_from_csv(real_csv, split="val", source_tag="real", min_det_conf=min_conf)
            if real_csv.exists()
            else []
        )
        test_samples = (
            load_samples_from_csv(real_csv, split="test", source_tag="real")
            if real_csv.exists()
            else []
        )
        hard_samples = (
            load_samples_from_csv(real_csv, split="hard_test", source_tag="real")
            if real_csv.exists()
            else []
        )

        common = dict(
            charset=self.charset,
            image_root=d.get("image_root", "."),
            min_len=int(d.get("min_text_len", 4)),
            max_len=int(d.get("max_text_len", 10)),
        )
        self._sets["train"] = OcrDataset(train_samples, transform=self._train_tf, **common)
        self._sets["val"] = OcrDataset(
            dedupe_by_image(val_samples) or train_samples[:256],
            transform=self._eval_tf,
            **common,
        )
        self._sets["test"] = OcrDataset(dedupe_by_image(test_samples), transform=self._eval_tf, **common)
        self._sets["hard_test"] = OcrDataset(
            dedupe_by_image(hard_samples), transform=self._eval_tf, **common
        )

    # ------------------------------------------------------------------ loaders
    def _loader(self, name: str, shuffle: bool) -> DataLoader:
        d = self.data_cfg
        return DataLoader(
            self._sets[name],
            batch_size=int(d.get("batch_size", 128)),
            shuffle=shuffle,
            num_workers=int(d.get("num_workers", 4)),
            pin_memory=bool(d.get("pin_memory", True)),
            drop_last=shuffle,
            collate_fn=ctc_collate,
            persistent_workers=int(d.get("num_workers", 4)) > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)

    def hard_test_dataloader(self) -> DataLoader:
        return self._loader("hard_test", shuffle=False)
