"""Country rule engines. Import side effects register the built-in engines."""

from tz_alpr.country_rules import tanzania
from tz_alpr.country_rules.base import (
    CategoryMatch,
    CountryRules,
    NormalizationResult,
    available_countries,
    get_country_rules,
    register,
)

__all__ = [
    "CategoryMatch",
    "CountryRules",
    "NormalizationResult",
    "available_countries",
    "get_country_rules",
    "register",
    "tanzania",
]
