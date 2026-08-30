"""Tanzania-aware OCR post-processing (spec §7, §28).

Corrections must be applied only when the plate schema demands it AND the OCR
posteriors support it, and must always be reported.
"""

from tz_alpr.ocr.ctc_decode import CharPosterior


def _pos(char, prob, alts):
    return CharPosterior(char=char, prob=prob, alternatives=[(char, prob), *alts])


def _clean_positions(text):
    return [_pos(c, 0.97, []) for c in text]


def test_swaps_letter_to_digit_when_slot_requires_number(decoder):
    # raw OCR "T33IEBG": index 3 is 'I' but slot 3 must be numeric.
    positions = _clean_positions("T33IEBG")
    positions[3] = _pos("I", 0.55, [("1", 0.42), ("L", 0.02)])

    result = decoder.decode("T33IEBG", positions, seq_confidence=0.80)

    assert result.raw_ocr == "T33IEBG"
    assert result.normalized_text == "T331EBG"
    assert result.n_swaps == 1
    assert "I->1" in result.corrections[0]
    assert result.category.name == "PRIVATE"


def test_no_swap_when_ocr_is_confident(decoder):
    positions = _clean_positions("T33IEBG")
    positions[3] = _pos("I", 0.995, [("1", 0.004)])  # above hard-lock

    result = decoder.decode("T33IEBG", positions, seq_confidence=0.9)

    assert result.normalized_text == "T33IEBG"
    assert result.n_swaps == 0


def test_no_swap_when_alternative_too_unlikely(decoder):
    positions = _clean_positions("T33IEBG")
    positions[3] = _pos("I", 0.90, [("1", 0.02)])  # gap 0.88 > swap_margin

    result = decoder.decode("T33IEBG", positions, seq_confidence=0.7)

    assert result.normalized_text == "T33IEBG"
    assert result.n_swaps == 0


def test_correct_reading_is_untouched(decoder):
    positions = _clean_positions("T331EBG")
    result = decoder.decode("T331EBG", positions, seq_confidence=0.96)
    assert result.normalized_text == "T331EBG"
    assert result.corrections == []
    assert result.display_text == "T 331 EBG"


def test_digit_to_letter_in_series_block(decoder):
    # "T3315BG": index 4 should be a letter; '5' -> 'S'
    positions = _clean_positions("T3315BG")
    positions[4] = _pos("5", 0.5, [("S", 0.45)])
    result = decoder.decode("T3315BG", positions, seq_confidence=0.75)
    assert result.normalized_text == "T331SBG"
    assert result.n_swaps == 1


def test_empty_ocr_is_safe(decoder):
    result = decoder.decode("", [], seq_confidence=0.0)
    assert result.normalized_text == ""
    assert result.category.name == "UNKNOWN"
    assert result.n_swaps == 0


def test_candidates_surfaced_for_review(decoder):
    positions = _clean_positions("T331EBG")
    positions[6] = _pos("G", 0.6, [("6", 0.3), ("C", 0.05)])
    result = decoder.decode("T331EBG", positions, seq_confidence=0.8)
    texts = [t for t, _ in result.ocr_candidates]
    assert "T331EBG" in texts
    assert any(t != "T331EBG" for t in texts)
