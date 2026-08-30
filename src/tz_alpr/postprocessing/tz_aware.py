"""Tanzania-aware recognition (spec §7).

Combines OCR per-character posteriors with the plate-rule engine's per-position
character-class expectations and OCR-confusion table to produce the final
prediction. Every character that is changed relative to the raw OCR argmax is
recorded in ``corrections`` and returned — corrections are never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tz_alpr.country_rules.base import CategoryMatch
from tz_alpr.country_rules.tanzania import TanzaniaRules
from tz_alpr.ocr.ctc_decode import CharPosterior

_CLASS_CHECKS = {
    "A": str.isalpha,
    "N": str.isdigit,
    "X": lambda _c: True,
}


@dataclass
class TzDecodeResult:
    raw_ocr: str                     # un-spaced OCR argmax
    normalized_text: str             # after rule-guided swaps
    display_text: str                # canonical spaced form
    category: CategoryMatch
    corrections: list[str] = field(default_factory=list)
    ocr_candidates: list[tuple[str, float]] = field(default_factory=list)
    final_char_probs: list[float] = field(default_factory=list)
    seq_confidence: float = 0.0

    @property
    def n_swaps(self) -> int:
        return len(self.corrections)

    @property
    def raw_spaced(self) -> str:
        return self.raw_ocr


class TanzaniaAwareDecoder:
    def __init__(self, rules: TanzaniaRules) -> None:
        self._rules = rules

    def decode(
        self, raw_ocr_text: str, positions: list[CharPosterior], seq_confidence: float
    ) -> TzDecodeResult:
        normalized_raw = self._rules.clean(raw_ocr_text)
        if not normalized_raw:
            empty_cat = self._rules.classify("")
            return TzDecodeResult(
                raw_ocr="",
                normalized_text="",
                display_text="",
                category=empty_cat,
                seq_confidence=seq_confidence,
            )

        category0 = self._rules.classify(normalized_raw)
        template = category0 if category0.matched else self._rules.best_template(normalized_raw)
        chars = list(normalized_raw)
        probs = [_prob_for(positions, i, chars[i]) for i in range(len(chars))]
        corrections: list[str] = []

        # Only attempt slot-guided correction when we have a per-position schema
        # (matched category, or an inferred template) of the same length.
        if template and template.slots and len(template.slots) == len(chars):
            category0 = template
            for i, ch in enumerate(chars):
                expected = self._rules.expected_class_at(len(chars), i, category0)
                if expected == "X" or _CLASS_CHECKS[expected](ch):
                    continue

                argmax_prob = _prob_for(positions, i, ch)
                if argmax_prob >= self._rules.hard_lock_prob:
                    continue

                best_alt, best_alt_prob = self._best_alternative(positions, i, ch, expected)
                if best_alt is None:
                    continue
                if best_alt_prob < argmax_prob - self._rules.swap_margin:
                    continue

                corrections.append(
                    f"pos {i}: {ch}->{best_alt} "
                    f"(p {argmax_prob:.2f}->{best_alt_prob:.2f}, slot={expected})"
                )
                chars[i] = best_alt
                probs[i] = best_alt_prob

        normalized = "".join(chars)
        category = self._rules.classify(normalized)
        display = self._rules._format_display(normalized, category)
        candidates = self._alternative_strings(normalized, positions, seq_confidence)

        return TzDecodeResult(
            raw_ocr=normalized_raw,
            normalized_text=normalized,
            display_text=display,
            category=category,
            corrections=corrections,
            ocr_candidates=candidates,
            final_char_probs=probs,
            seq_confidence=seq_confidence,
        )

    # ------------------------------------------------------------------ helpers
    def _best_alternative(
        self, positions: list[CharPosterior], i: int, current: str, expected: str
    ) -> tuple[str | None, float]:
        options = self._rules.confusion_alternatives(current)
        check = _CLASS_CHECKS[expected]
        best: str | None = None
        best_p = -1.0
        for alt in options:
            if len(alt) != 1 or not check(alt):
                continue
            p = _prob_for(positions, i, alt)
            if p > best_p:
                best, best_p = alt, p
        return best, max(best_p, 0.0)

    def _alternative_strings(
        self, normalized: str, positions: list[CharPosterior], seq_conf: float
    ) -> list[tuple[str, float]]:
        """Surface a few whole-string alternatives for human verification (spec §23)."""
        cands: dict[str, float] = {normalized: seq_conf}
        for i in range(min(len(normalized), len(positions))):
            pos = positions[i]
            for alt_char, alt_p in pos.alternatives[1:3]:
                variant = normalized[:i] + alt_char + normalized[i + 1 :]
                score = seq_conf * (alt_p / max(pos.prob, 1e-6))
                cands[variant] = max(cands.get(variant, 0.0), min(score, 0.999))
        ordered = sorted(cands.items(), key=lambda kv: kv[1], reverse=True)
        return ordered[:4]


def _prob_for(positions: list[CharPosterior], i: int, char: str) -> float:
    if i >= len(positions):
        return 0.0
    return positions[i].prob_of(char)
