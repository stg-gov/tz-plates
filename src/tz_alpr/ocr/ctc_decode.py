"""CTC decoding utilities (numpy only, so the ONNX runtime path needs no torch).

``greedy_decode`` returns not just a string but a per-output-character posterior
distribution, which the Tanzania-aware post-processor needs to decide whether a
rule-driven character swap is justified (spec §7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tz_alpr.ocr.charset import Charset


@dataclass
class CharPosterior:
    char: str
    prob: float
    alternatives: list[tuple[str, float]] = field(default_factory=list)  # (char, prob) desc

    def prob_of(self, char: str) -> float:
        if char == self.char:
            return self.prob
        for c, p in self.alternatives:
            if c == char:
                return p
        return 0.0


@dataclass
class DecodedSequence:
    text: str
    seq_confidence: float
    positions: list[CharPosterior]

    @property
    def mean_char_prob(self) -> float:
        if not self.positions:
            return 0.0
        return float(np.mean([p.prob for p in self.positions]))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def greedy_decode(
    logits_or_probs: np.ndarray,
    charset: Charset,
    topk: int = 4,
    is_probs: bool = False,
) -> DecodedSequence:
    """Decode a single (T, C) sequence.

    `logits_or_probs`: (T, C) array. Set `is_probs=True` if already softmaxed.
    """
    probs = logits_or_probs if is_probs else softmax(np.asarray(logits_or_probs), axis=-1)
    if probs.ndim != 2:
        raise ValueError(f"Expected (T, C) array, got shape {probs.shape}")

    argmax = probs.argmax(axis=1)
    blank = charset.blank_index

    positions: list[CharPosterior] = []
    group_start = None
    prev = blank

    def flush(end: int) -> None:
        if group_start is None:
            return
        window = probs[group_start:end].mean(axis=0)
        order = np.argsort(window)[::-1]
        alts: list[tuple[str, float]] = []
        for idx in order:
            if idx == blank:
                continue
            alts.append((charset.char_at(int(idx)), float(window[idx])))
            if len(alts) >= topk:
                break
        top_char, top_prob = alts[0]
        positions.append(CharPosterior(char=top_char, prob=top_prob, alternatives=alts))

    for t, a in enumerate(argmax):
        if a == prev:
            continue
        if prev != blank:
            flush(t)
        group_start = t if a != blank else None
        prev = a
    if prev != blank:
        flush(len(argmax))

    text = "".join(p.char for p in positions)
    if positions:
        logs = np.log(np.clip([p.prob for p in positions], 1e-9, 1.0))
        seq_conf = float(np.exp(logs.mean()))
    else:
        seq_conf = 0.0
    return DecodedSequence(text=text, seq_confidence=seq_conf, positions=positions)
