"""Plate string normalization: separators, glyphs, casing (spec §2, §28)."""

import pytest


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("T 331 EBG", "T331EBG"),
        ("  t331ebg  ", "T331EBG"),
        ("T·331·EBG", "T331EBG"),
        ("T—331—EBG", "T331EBG"),
        ("T/331/EBG", "T331EBG"),
        ("T.331.EBG", "T331EBG"),
        ("T|33|EBG", "TI33IEBG"),
    ],
)
def test_clean_strips_separators(tz_rules, raw, expected):
    assert tz_rules.clean(raw) == expected


def test_clean_is_pure_no_semantic_swaps(tz_rules):
    # clean() must not turn letters into digits — that is the decoder's job.
    assert tz_rules.clean("TI33IEBG") == "TI33IEBG"
    assert tz_rules.clean("O0O0") == "O0O0"


def test_normalize_preserves_raw_text(tz_rules):
    result = tz_rules.normalize("T 331 EBG")
    assert result.raw_text == "T 331 EBG"
    assert result.normalized_text == "T331EBG"
    assert result.country_code == "TZ"


def test_registry_exposes_tanzania():
    from tz_alpr.country_rules import available_countries, get_country_rules

    assert "TZ" in available_countries()
    assert get_country_rules("tz").country_name == "Tanzania"
