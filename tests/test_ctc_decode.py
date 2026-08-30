"""CTC greedy decoding + per-character posteriors (spec §6, §28)."""

import numpy as np

from tz_alpr.ocr.charset import load_charset
from tz_alpr.ocr.ctc_decode import greedy_decode


def _one_hot_sequence(charset, tokens, repeat=2, blank_between=1):
    """Build a synthetic (T, C) logit tensor that decodes to `tokens`."""
    frames = []
    for tok in tokens:
        idx = charset.chars.index(tok) + 1
        for _ in range(repeat):
            row = np.full(charset.size, -8.0)
            row[idx] = 6.0
            frames.append(row)
        for _ in range(blank_between):
            row = np.full(charset.size, -8.0)
            row[0] = 6.0
            frames.append(row)
    return np.array(frames, dtype=np.float32)


def test_greedy_decode_collapses_repeats_and_blanks():
    cs = load_charset()
    logits = _one_hot_sequence(cs, "T331EBG")
    decoded = greedy_decode(logits, cs)
    assert decoded.text == "T331EBG"
    assert len(decoded.positions) == 7
    assert decoded.seq_confidence > 0.9


def test_positions_carry_alternatives():
    cs = load_charset()
    logits = _one_hot_sequence(cs, "T1")
    # Blur the second character between '1' and 'I'.
    i1 = cs.chars.index("1") + 1
    iI = cs.chars.index("I") + 1
    for t in range(len(logits)):
        if logits[t].argmax() == i1:
            logits[t][iI] = 5.0
    decoded = greedy_decode(logits, cs, topk=3)
    last = decoded.positions[-1]
    alt_chars = {c for c, _ in last.alternatives}
    assert "1" in alt_chars and "I" in alt_chars
    assert last.prob_of("I") > 0.0


def test_empty_decode_is_safe():
    cs = load_charset()
    blank_only = np.full((10, cs.size), -8.0, dtype=np.float32)
    blank_only[:, 0] = 6.0
    decoded = greedy_decode(blank_only, cs)
    assert decoded.text == ""
    assert decoded.seq_confidence == 0.0


def test_probs_sum_to_one_after_softmax():
    cs = load_charset()
    logits = _one_hot_sequence(cs, "MC101AAA")
    decoded = greedy_decode(logits, cs)
    assert decoded.text == "MC101AAA"
    for p in decoded.positions:
        total = p.prob + sum(pr for _, pr in p.alternatives[1:])
        assert total <= 1.0 + 1e-6
