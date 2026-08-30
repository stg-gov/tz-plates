"""OCR dataset: reads ``ocr_annotations.csv`` and yields (image, target) pairs.

CSV schema (written by tools/prepare_dataset.py, spec §11):
    image_path,plate_text,country,plate_type,split,group_key,source

`image_path` is relative to `image_root` (repo root by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from tz_alpr.ocr.charset import Charset


@dataclass
class OcrSample:
    image_path: str
    text: str
    plate_type: str = "UNKNOWN"
    group_key: str = ""
    source: str = "real"


class OcrDataset(Dataset):
    def __init__(
        self,
        samples: list[OcrSample],
        charset: Charset,
        transform,
        image_root: str | Path = ".",
        min_len: int = 4,
        max_len: int = 10,
    ) -> None:
        self._charset = charset
        self._transform = transform
        self._root = Path(image_root)
        self._samples = [s for s in samples if min_len <= len(s.text) <= max_len]

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> list[OcrSample]:
        return self._samples

    def __getitem__(self, idx: int):
        sample = self._samples[idx]
        path = self._root / sample.image_path
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            # Skip unreadable file by borrowing the next one; keeps the loader alive.
            return self.__getitem__((idx + 1) % len(self._samples))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self._transform(image=img)["image"]

        target = torch.tensor(self._charset.encode(sample.text), dtype=torch.long)
        return {
            "image": tensor,
            "target": target,
            "target_length": torch.tensor(len(target), dtype=torch.long),
            "text": sample.text,
        }


def ctc_collate(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets = torch.cat([b["target"] for b in batch], dim=0)
    target_lengths = torch.stack([b["target_length"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]
    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths,
        "texts": texts,
    }


def load_samples_from_csv(
    csv_path: str | Path,
    split: str | None = None,
    source_tag: str | None = None,
    min_det_conf: float = 0.0,
) -> list[OcrSample]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split]
    if min_det_conf > 0.0 and "det_conf" in df.columns:
        conf = pd.to_numeric(df["det_conf"], errors="coerce").fillna(0.0)
        df = df[conf >= min_det_conf]
    out: list[OcrSample] = []
    for row in df.itertuples(index=False):
        d = row._asdict()
        out.append(
            OcrSample(
                image_path=d["image_path"],
                text=d["plate_text"].strip().upper(),
                plate_type=d.get("plate_type", "UNKNOWN") or "UNKNOWN",
                group_key=d.get("group_key", "") or "",
                source=source_tag or d.get("source", "real") or "real",
            )
        )
    return out


def dedupe_by_image(samples: list[OcrSample]) -> list[OcrSample]:
    seen: set[str] = set()
    out: list[OcrSample] = []
    for s in samples:
        if s.image_path in seen:
            continue
        seen.add(s.image_path)
        out.append(s)
    return out
