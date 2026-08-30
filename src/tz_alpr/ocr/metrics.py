"""OCR evaluation metrics (spec §15): character accuracy, CER, sequence accuracy.

Kept torch-free so both the Lightning module and the standalone evaluator use
the same implementation.
"""

from __future__ import annotations

from dataclasses import dataclass


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class OcrScores:
    n: int
    sequence_accuracy: float
    character_accuracy: float
    character_error_rate: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        p = f"{prefix}/" if prefix else ""
        return {
            f"{p}seq_acc": self.sequence_accuracy,
            f"{p}char_acc": self.character_accuracy,
            f"{p}cer": self.character_error_rate,
        }


def score_batch(preds: list[str], gts: list[str]) -> OcrScores:
    if not gts:
        return OcrScores(0, 0.0, 0.0, 1.0)
    total_chars = 0
    total_edits = 0
    exact = 0
    for p, g in zip(preds, gts):
        total_chars += max(1, len(g))
        total_edits += levenshtein(p, g)
        exact += int(p == g)
    cer = total_edits / total_chars
    return OcrScores(
        n=len(gts),
        sequence_accuracy=exact / len(gts),
        character_accuracy=max(0.0, 1.0 - cer),
        character_error_rate=cer,
    )
