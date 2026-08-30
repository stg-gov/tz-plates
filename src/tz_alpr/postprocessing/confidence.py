"""Calibrated confidence strategy (spec §16).

We deliberately do NOT just multiply the stage scores. Naive multiplication is
biased low (independent-error assumption is false — a good detection and a
confident OCR are correlated) and it destroys calibration.

Strategy:
  1. Temperature-scale the raw OCR sequence confidence (optionally Platt-scaled
     from validation data via ``calibrate``). Detector and validation scores are
     already reasonably calibrated and pass through.
  2. Fuse the three stage scores with a *weighted geometric mean*, each score
     floored at ``min_stage_floor`` so one weak stage lowers — but cannot zero —
     the result.
  3. Apply explicit, auditable penalties: one per rule-driven character swap, and
     one if the string length disagrees with the matched plate category.
  4. Clamp to [0, 1] and route to auto-accept / review / manual using the
     configurable thresholds in ``configs/tanzania.yaml``.

``calibrate`` fits a 1-D logistic (Platt) map from raw fused score -> P(correct)
using operator-verified outcomes, so thresholds become meaningful over time
(spec §23).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

_EPS = 1e-6


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(1.0 - _EPS, max(_EPS, p))
    return math.log(p / (1.0 - p))


@dataclass
class StageScores:
    vehicle: float = 0.0
    plate_detection: float = 0.0
    ocr_seq: float = 0.0
    plate_validation: float = 0.0


@dataclass
class FusedConfidence:
    vehicle_confidence: float
    plate_detection_confidence: float
    ocr_confidence: float
    plate_validation_confidence: float
    final_confidence: float
    review_status: str


class ConfidenceModel:
    def __init__(self, confidence_cfg: dict, review_cfg: dict) -> None:
        w = confidence_cfg.get("weights", {})
        self._w = {
            "plate_detection": float(w.get("plate_detection", 0.20)),
            "ocr": float(w.get("ocr", 0.55)),
            "plate_validation": float(w.get("plate_validation", 0.25)),
        }
        self._floor = float(confidence_cfg.get("min_stage_floor", 0.05))
        self._temp = float(confidence_cfg.get("ocr_temperature", 1.6))
        self._bias = float(confidence_cfg.get("ocr_bias", 0.0))
        self._swap_penalty = float(confidence_cfg.get("swap_penalty_per_char", 0.03))
        self._len_penalty = float(confidence_cfg.get("length_mismatch_penalty", 0.25))
        # Platt parameters for raw fused score -> calibrated probability.
        self._platt_a = 1.0
        self._platt_b = 0.0
        self._auto_accept = float(review_cfg.get("auto_accept", 0.90))
        self._review_low = float(review_cfg.get("review_band_low", 0.70))

    # ------------------------------------------------------------------- fuse
    def calibrate_ocr(self, ocr_seq_conf: float) -> float:
        return _sigmoid(_logit(ocr_seq_conf) / max(self._temp, _EPS) + self._bias)

    def fuse(
        self, stages: StageScores, n_swaps: int = 0, length_mismatch: bool = False
    ) -> FusedConfidence:
        ocr_cal = self.calibrate_ocr(stages.ocr_seq)

        parts = {
            "plate_detection": max(stages.plate_detection, self._floor),
            "ocr": max(ocr_cal, self._floor),
            "plate_validation": max(stages.plate_validation, self._floor),
        }
        wsum = sum(self._w.values()) or 1.0
        log_mean = sum(self._w[k] * math.log(parts[k]) for k in parts) / wsum
        fused = math.exp(log_mean)

        fused -= self._swap_penalty * max(0, n_swaps)
        if length_mismatch:
            fused -= self._len_penalty
        fused = min(1.0, max(0.0, fused))

        calibrated = _sigmoid(self._platt_a * _logit(fused) + self._platt_b)
        final = min(1.0, max(0.0, calibrated))

        return FusedConfidence(
            vehicle_confidence=round(stages.vehicle, 4),
            plate_detection_confidence=round(stages.plate_detection, 4),
            ocr_confidence=round(ocr_cal, 4),
            plate_validation_confidence=round(stages.plate_validation, 4),
            final_confidence=round(final, 4),
            review_status=self.route(final),
        )

    def route(self, final_confidence: float) -> str:
        if final_confidence >= self._auto_accept:
            return "auto_accept"
        if final_confidence >= self._review_low:
            return "review"
        return "manual"

    # -------------------------------------------------------------- calibration
    def calibrate(
        self, samples: list[tuple[float, bool]], iters: int = 500, lr: float = 0.1
    ) -> tuple[float, float]:
        """Fit Platt scaling: P(correct) = sigmoid(a * logit(raw) + b)."""
        if len(samples) < 20:
            raise ValueError("Need at least 20 verified samples to calibrate confidence.")
        a, b = 1.0, 0.0
        xs = [_logit(min(1.0 - _EPS, max(_EPS, s))) for s, _ in samples]
        ys = [1.0 if ok else 0.0 for _, ok in samples]
        n = len(samples)
        for _ in range(iters):
            ga = gb = 0.0
            for x, y in zip(xs, ys):
                p = _sigmoid(a * x + b)
                err = p - y
                ga += err * x
                gb += err
            a -= lr * ga / n
            b -= lr * gb / n
        self._platt_a, self._platt_b = a, b
        return a, b

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"platt_a": self._platt_a, "platt_b": self._platt_b}, indent=2)
        )

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self._platt_a = float(data.get("platt_a", 1.0))
        self._platt_b = float(data.get("platt_b", 0.0))


def build_confidence_model(tanzania_config: str = "configs/tanzania.yaml") -> ConfidenceModel:
    from tz_alpr.config import load_yaml

    cfg = load_yaml(tanzania_config)
    return ConfidenceModel(cfg.get("confidence", {}), cfg.get("review", {}))
