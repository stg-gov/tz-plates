"""Tanzania plate-rule engine (spec §2, §7).

Everything that varies by policy — regex, per-position character classes, plate
categories, OCR-confusion table, confidence adjustments — is read from
``configs/tanzania.yaml``. This module is the logic that applies that data; it
does not hard-code plate shapes.

Key principle: corrections are never silent. Any character the rule engine
changes relative to the raw OCR argmax is recorded and returned to the caller.
"""

from __future__ import annotations

import re
from dataclasses import replace
from functools import lru_cache

from tz_alpr.config import load_yaml
from tz_alpr.country_rules.base import (
    CategoryMatch,
    CountryRules,
    NormalizationResult,
    register,
)

_CLASS_ALPHA = "A"
_CLASS_NUM = "N"
_CLASS_ANY = "X"

_DEFAULT_CONFIG = "configs/tanzania.yaml"


class _CompiledCategory:
    __slots__ = ("meta", "normalized_regex", "priority", "regex")

    def __init__(self, raw: dict) -> None:
        self.regex = re.compile(raw["regex"])
        self.normalized_regex = re.compile(raw["normalized_regex"])
        self.priority = int(raw.get("priority", 0))
        self.meta = CategoryMatch(
            name=raw["name"],
            description=raw.get("description", ""),
            slots=raw.get("slots"),
            display_format=raw.get("display_format", "{raw}"),
            plate_colour=raw.get("plate_colour", "unknown"),
            lines=int(raw.get("lines", 1)),
            confidence_bonus=float(raw.get("confidence_bonus", 0.0)),
        )


class TanzaniaRules(CountryRules):
    country_code = "TZ"
    country_name = "Tanzania"

    def __init__(self, config: dict) -> None:
        self._cfg = config
        self.charset = config["charset"]
        self._categories = sorted(
            (_CompiledCategory(c) for c in config["categories"]),
            key=lambda c: c.priority,
            reverse=True,
        )
        self._unknown = next(
            (c for c in self._categories if c.meta.name == "UNKNOWN"), self._categories[-1]
        )
        self._confusions: dict[str, list[str]] = {
            k.upper(): [v.upper() for v in vals]
            for k, vals in config.get("ocr_confusions", {}).items()
        }
        self._swap_margin = float(config.get("swap_margin", 0.35))
        self._hard_lock = float(config.get("min_char_prob_for_hard_lock", 0.985))
        self._conf_cfg = config.get("confidence", {})
        self._review_cfg = config.get("review", {})

    # ------------------------------------------------------------------ classify
    def classify(self, normalized: str) -> CategoryMatch:
        for cat in self._categories:
            if cat.meta.name == "UNKNOWN":
                continue
            if cat.normalized_regex.match(normalized):
                return replace(cat.meta, matched=True)
        return replace(self._unknown.meta, matched=False)

    def best_template(self, normalized: str) -> CategoryMatch | None:
        """Provisional per-position schema when the raw string matches no category.

        Real OCR errors (I<->1, S<->5, ...) stop the raw string from matching any
        regex, so we cannot rely on ``classify`` alone to obtain a slot schema.
        Pick the slotted category whose length matches and whose positions are
        already mostly satisfied — directly or via a known OCR confusion — and
        return it with ``matched=False`` so callers know it is a hypothesis.
        """
        best: CategoryMatch | None = None
        best_score = 0.0
        for cat in self._categories:
            slots = cat.meta.slots
            if not slots or len(slots) != len(normalized):
                continue
            hits = 0.0
            for ch, want in zip(normalized, slots):
                if want == "X" or (want == "A" and ch.isalpha()) or (want == "N" and ch.isdigit()):
                    hits += 1.0
                elif any(
                    (want == "A" and alt.isalpha()) or (want == "N" and alt.isdigit())
                    for alt in self._confusions.get(ch, [])
                ):
                    hits += 0.7
            score = hits / len(slots) + cat.priority * 1e-4
            if score > best_score and score >= 0.6:
                best, best_score = replace(cat.meta, matched=False), score
        return best

    # ----------------------------------------------------------------- normalize
    def normalize(self, raw_text: str) -> NormalizationResult:
        normalized = self.clean(raw_text)
        category = self.classify(normalized)
        display = self._format_display(normalized, category)
        return NormalizationResult(
            raw_text=raw_text,
            normalized_text=normalized,
            display_text=display,
            category=category,
            country_code=self.country_code,
            corrections=[],
            is_valid=category.matched,
        )

    def _format_display(self, normalized: str, category: CategoryMatch) -> str:
        fmt = category.display_format
        if "{raw}" in fmt:
            return normalized
        try:
            return fmt.format(*list(normalized))
        except (IndexError, KeyError):
            return normalized

    # -------------------------------------------------------- decoding helpers
    def expected_class_at(
        self, normalized_len: int, position: int, category: CategoryMatch
    ) -> str:
        slots = category.slots
        if not slots or len(slots) != normalized_len or not (0 <= position < len(slots)):
            return _CLASS_ANY
        token = slots[position]
        return token if token in (_CLASS_ALPHA, _CLASS_NUM, _CLASS_ANY) else _CLASS_ANY

    def confusion_alternatives(self, char: str) -> list[str]:
        return list(self._confusions.get(char.upper(), []))

    @property
    def swap_margin(self) -> float:
        return self._swap_margin

    @property
    def hard_lock_prob(self) -> float:
        return self._hard_lock

    @property
    def review_thresholds(self) -> dict:
        return {
            "auto_accept": float(self._review_cfg.get("auto_accept", 0.90)),
            "review_band_low": float(self._review_cfg.get("review_band_low", 0.70)),
        }

    @property
    def confidence_config(self) -> dict:
        return dict(self._conf_cfg)

    # ------------------------------------------------------- validation score
    def validation_confidence(
        self, normalized: str, category: CategoryMatch, n_swaps: int
    ) -> float:
        base = 0.86 if category.matched else 0.45
        base += category.confidence_bonus

        if category.slots and len(category.slots) != len(normalized):
            base -= float(self._conf_cfg.get("length_mismatch_penalty", 0.25))

        base -= n_swaps * float(self._conf_cfg.get("swap_penalty_per_char", 0.03))
        return max(0.02, min(0.99, base))

    # convenience for tests / tooling
    def is_valid_plate(self, text: str) -> bool:
        return self.classify(self.clean(text)).matched


@lru_cache(maxsize=4)
def build(config_path: str = _DEFAULT_CONFIG) -> TanzaniaRules:
    return TanzaniaRules(load_yaml(config_path))


# Auto-register the default Tanzania engine on import.
try:  # pragma: no cover - defensive: never let a bad config break the import graph
    register(build())
except FileNotFoundError:
    pass
