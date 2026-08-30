"""Minimal plate-detect + CRNN OCR for the Hugging Face Space."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import nn

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
H, W = 32, 256
MODEL_REPO = os.environ.get("HF_MODEL_REPO", "japhari/tz-alpr-ocr")


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CRNN(nn.Module):
    def __init__(self, num_classes: int = 37) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            _ConvBlock(1, 64),
            nn.MaxPool2d(2, 2),
            _ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),
            _ConvBlock(128, 256),
            _ConvBlock(256, 256),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            _ConvBlock(256, 512),
            _ConvBlock(512, 512),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, 2, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.collapse = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(512, num_classes)

    def infer_logits(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.collapse(self.backbone(x)).squeeze(2).permute(0, 2, 1)
        seq, _ = self.rnn(feat)
        return self.classifier(seq)


def _greedy_ctc(logits_tc: np.ndarray) -> tuple[str, float]:
    idx = logits_tc.argmax(axis=1)
    # softmax conf of argmax
    x = logits_tc - logits_tc.max(axis=1, keepdims=True)
    p = np.exp(x)
    p = p / p.sum(axis=1, keepdims=True)
    chars: list[str] = []
    probs: list[float] = []
    prev = 0
    for t, i in enumerate(idx.tolist()):
        if i != 0 and i != prev:
            chars.append(CHARSET[i - 1])
            probs.append(float(p[t, i]))
        prev = i
    text = "".join(chars)
    conf = float(np.mean(probs)) if probs else 0.0
    return text, conf


def _classify(text: str) -> tuple[str, str]:
    t = "".join(c for c in text.upper() if c.isalnum())
    if len(t) == 7 and t[0] == "T" and t[1:4].isdigit() and t[4:].isalpha():
        return t, "PRIVATE"
    if t.startswith("MC") and len(t) == 8 and t[2:5].isdigit() and t[5:].isalpha():
        return t, "MOTORCYCLE"
    return t, "UNKNOWN"


def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
    arr = gray.astype(np.float32) / 255.0
    return arr[None, None, ...]


@lru_cache(maxsize=1)
def load_models():
    from ultralytics import YOLO

    ocr_path = hf_hub_download(MODEL_REPO, "ocr_crnn.pt")
    cfg_path = hf_hub_download(MODEL_REPO, "config.json")
    cfg = json.loads(Path(cfg_path).read_text())
    model = CRNN(num_classes=int(cfg.get("num_classes", 37)))
    state = torch.load(ocr_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    try:
        yolo_path = hf_hub_download(MODEL_REPO, "plate_yolo.pt")
    except Exception:
        yolo_path = hf_hub_download(
            "morsetechlab/yolov11-license-plate-detection",
            "license-plate-finetune-v1n.pt",
        )
    detector = YOLO(yolo_path)
    return model, detector


def recognize(image_bgr: np.ndarray) -> dict:
    ocr, detector = load_models()
    results = detector.predict(image_bgr, verbose=False, conf=0.25)
    vis = image_bgr.copy()
    plates = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            with torch.no_grad():
                logits = ocr.infer_logits(torch.from_numpy(_preprocess(crop)))[0].numpy()
            raw, ocr_p = _greedy_ctc(logits)
            plate, ptype = _classify(raw)
            det_p = float(box.conf[0])
            fused = float((max(det_p, 1e-3) * max(ocr_p, 1e-3) * (0.92 if ptype != "UNKNOWN" else 0.55)) ** (1 / 3))
            review = "auto_accept" if fused >= 0.90 else "review" if fused >= 0.70 else "manual"
            plates.append(
                {
                    "plate": plate,
                    "raw_ocr": raw,
                    "plate_type": ptype,
                    "ocr_confidence": round(ocr_p, 4),
                    "detection_confidence": round(det_p, 4),
                    "confidence": round(fused, 4),
                    "review_status": review,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                }
            )
            color = (30, 180, 70) if review == "auto_accept" else (40, 160, 220) if review == "review" else (40, 40, 200)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            cv2.putText(vis, f"{plate}  {fused:.2f}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if not plates:
        # whole-image fallback (already a crop)
        with torch.no_grad():
            logits = ocr.infer_logits(torch.from_numpy(_preprocess(image_bgr)))[0].numpy()
        raw, ocr_p = _greedy_ctc(logits)
        plate, ptype = _classify(raw)
        plates.append(
            {
                "plate": plate,
                "raw_ocr": raw,
                "plate_type": ptype,
                "ocr_confidence": round(ocr_p, 4),
                "detection_confidence": 0.0,
                "confidence": round(ocr_p, 4),
                "review_status": "review" if ocr_p >= 0.7 else "manual",
                "bbox": None,
                "note": "no detector box — OCR on full frame",
            }
        )
    return {"results": plates, "annotated_bgr": vis}
