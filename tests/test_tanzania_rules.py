"""Tanzania regex validation + plate categorization (spec §2, §28)."""

import pytest


@pytest.mark.parametrize(
    "raw,expected_norm,expected_type",
    [
        ("T 331 EBG", "T331EBG", "PRIVATE"),
        ("t331ebg", "T331EBG", "PRIVATE"),
        ("T-123-ABC", "T123ABC", "PRIVATE"),
        ("T456DEF", "T456DEF", "PRIVATE"),
        ("MC 101 AAA", "MC101AAA", "MOTORCYCLE"),
        ("MC529ATT", "MC529ATT", "MOTORCYCLE"),
        ("SU 1234", "SU1234", "GOVERNMENT"),
        ("SM 11736", "SM11736", "GOVERNMENT"),
        ("SM11736", "SM11736", "GOVERNMENT"),
        ("DFPA 2925", "DFPA2925", "SPECIAL"),
        ("DFPA2925", "DFPA2925", "SPECIAL"),
    ],
)
def test_normalize_and_classify(tz_rules, raw, expected_norm, expected_type):
    result = tz_rules.normalize(raw)
    assert result.normalized_text == expected_norm
    assert result.category.name == expected_type
    assert result.is_valid is True


@pytest.mark.parametrize("bad", ["", "!!", "T33", "ZZZZZZZZZZZZ", "12"])
def test_rejects_invalid(tz_rules, bad):
    result = tz_rules.normalize(bad)
    assert result.category.name == "UNKNOWN"
    assert result.is_valid is False


def test_display_formatting(tz_rules):
    assert tz_rules.normalize("T331EBG").display_text == "T 331 EBG"
    assert tz_rules.normalize("MC101AAA").display_text == "MC 101 AAA"


def test_slot_schema(tz_rules):
    cat = tz_rules.classify("T331EBG")
    assert cat.slots == "ANNNAAA"
    assert tz_rules.expected_class_at(7, 0, cat) == "A"
    assert tz_rules.expected_class_at(7, 1, cat) == "N"
    assert tz_rules.expected_class_at(7, 6, cat) == "A"


def test_confusion_table_present(tz_rules):
    assert "1" in tz_rules.confusion_alternatives("I")
    assert "8" in tz_rules.confusion_alternatives("B")
    assert "5" in tz_rules.confusion_alternatives("S")


def test_charset_matches_config(tz_rules):
    assert set("T331EBG") <= set(tz_rules.charset)
    assert len(tz_rules.charset) == 36


def test_validation_confidence_penalises_swaps(tz_rules):
    cat = tz_rules.classify("T331EBG")
    high = tz_rules.validation_confidence("T331EBG", cat, n_swaps=0)
    low = tz_rules.validation_confidence("T331EBG", cat, n_swaps=3)
    assert high > low
    assert 0.0 <= low <= high <= 1.0
