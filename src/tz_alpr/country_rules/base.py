"""Country-rule engine contract + registry.

Adding a country (Kenya, Uganda, Rwanda, ...) means dropping a new module that
subclasses :class:`CountryRules` and calling :func:`register`. Nothing about the
detector or the OCR head changes (spec §2, §3).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Unicode / look-alike glyphs occasionally emitted by upstream OCR or present in
# copy-pasted ground truth. Applied before any country-specific logic.
_UNIVERSAL_GLYPH_MAP = {
    "–": "-", "—": "-", "‐": "-", " ": " ",
    "Ａ": "A", "А": "A", "Ο": "O", "V": "V",
    "|": "I", "!": "I", "\\": "", "/": "", ".": "", ",": "", "·": "", "•": "",
}


@dataclass
class CategoryMatch:
    name: str
    description: str = ""
    slots: str | None = None            # per-position class string: A / N / X
    display_format: str = "{raw}"
    plate_colour: str = "unknown"
    lines: int = 1
    confidence_bonus: float = 0.0
    matched: bool = False               # True if a real category regex matched


@dataclass
class NormalizationResult:
    raw_text: str                       # OCR output, spacing preserved
    normalized_text: str                # upper-case, no separators
    display_text: str                   # canonical spaced form, e.g. "T 331 EBG"
    category: CategoryMatch
    country_code: str
    corrections: list[str] = field(default_factory=list)
    is_valid: bool = False


class CountryRules(ABC):
    country_code: str = "XX"
    country_name: str = "Unknown"
    charset: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def clean(self, raw_text: str) -> str:
        """Strip separators and normalize glyphs, WITHOUT applying any semantic swaps."""
        text = raw_text.strip()
        for src, dst in _UNIVERSAL_GLYPH_MAP.items():
            text = text.replace(src, dst)
        text = text.upper()
        text = re.sub(r"[\s\-_]+", "", text)
        text = re.sub(r"[^0-9A-Z]", "", text)
        return text

    @abstractmethod
    def classify(self, normalized: str) -> CategoryMatch:
        ...

    @abstractmethod
    def normalize(self, raw_text: str) -> NormalizationResult:
        ...

    @abstractmethod
    def expected_class_at(self, normalized_len: int, position: int, category: CategoryMatch) -> str:
        """Return 'A', 'N' or 'X' for the given position given the resolved category."""

    @abstractmethod
    def confusion_alternatives(self, char: str) -> list[str]:
        ...

    @abstractmethod
    def validation_confidence(
        self, normalized: str, category: CategoryMatch, n_swaps: int
    ) -> float:
        ...


_REGISTRY: dict[str, CountryRules] = {}


def register(rules: CountryRules) -> None:
    _REGISTRY[rules.country_code.upper()] = rules


def get_country_rules(country_code: str) -> CountryRules:
    code = country_code.upper()
    if code not in _REGISTRY:
        raise KeyError(
            f"No rule engine registered for '{code}'. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[code]


def available_countries() -> list[str]:
    return sorted(_REGISTRY)
