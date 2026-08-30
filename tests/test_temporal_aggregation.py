"""Temporal OCR aggregation: OCR-probability-weighted voting over a track (spec §17)."""

from tz_alpr.ocr.ctc_decode import CharPosterior
from tz_alpr.tracking.aggregator import PlateObservation, TemporalPlateAggregator


def _positions(text, conf=0.95):
    return [CharPosterior(char=c, prob=conf, alternatives=[(c, conf)]) for c in text]


def _obs(frame, text, final_conf, positions=None, corrections=None):
    return PlateObservation(
        frame_idx=frame,
        timestamp=frame * 0.2,
        raw_ocr=text,
        normalized_text=text,
        positions=positions or _positions(text, final_conf),
        seq_confidence=final_conf,
        final_confidence=final_conf,
        plate_det_conf=0.9,
        plate_type="PRIVATE",
        corrections=corrections or [],
        vehicle_type="car",
    )


def test_spec_example_weighted_vote(tz_rules, confidence_model):
    agg = TemporalPlateAggregator(tz_rules, confidence_model)
    # frame reads: T331E8G .72 / T331EBG .91 / T331EBG .96 / T331E8G .80
    agg.add(1, _obs(1, "T331E8G", 0.72))
    agg.add(1, _obs(2, "T331EBG", 0.91))
    agg.add(1, _obs(3, "T331EBG", 0.96))
    agg.add(1, _obs(4, "T331E8G", 0.80))

    result = agg.aggregate(1)
    assert result.plate == "T331EBG"
    assert result.plate_type == "PRIVATE"
    assert result.n_frames == 4
    # noisy-OR over the two agreeing 0.91 / 0.96 frames -> higher than any single frame
    assert result.confidence > 0.96
    assert result.first_frame == 1 and result.last_frame == 4


def test_posterior_breaks_the_tie(tz_rules, confidence_model):
    agg = TemporalPlateAggregator(tz_rules, confidence_model)
    # one frame reads B, one reads 8 at index 5 (equal frame weight); posteriors decide
    p1 = _positions("T331EBG", 0.8)
    p1[5] = CharPosterior(char="B", prob=0.6, alternatives=[("B", 0.6), ("8", 0.3)])
    p2 = _positions("T331E8G", 0.8)
    p2[5] = CharPosterior(char="8", prob=0.4, alternatives=[("8", 0.4), ("B", 0.45)])
    agg.add(2, _obs(1, "T331EBG", 0.8, positions=p1))
    agg.add(2, _obs(2, "T331E8G", 0.8, positions=p2))

    assert agg.aggregate(2).plate == "T331EBG"


def test_dominant_length_wins(tz_rules, confidence_model):
    agg = TemporalPlateAggregator(tz_rules, confidence_model)
    agg.add(3, _obs(1, "T331EBG", 0.9))
    agg.add(3, _obs(2, "T331EBG", 0.9))
    agg.add(3, _obs(3, "T331EB", 0.4))  # a short mis-read
    result = agg.aggregate(3)
    assert result.plate == "T331EBG"
    assert len(result.plate) == 7


def test_corrections_are_propagated(tz_rules, confidence_model):
    agg = TemporalPlateAggregator(tz_rules, confidence_model)
    agg.add(4, _obs(1, "T331EBG", 0.9, corrections=["pos 3: I->1 (p 0.55->0.42, slot=N)"]))
    agg.add(4, _obs(2, "T331EBG", 0.9))
    assert "pos 3: I->1 (p 0.55->0.42, slot=N)" in agg.aggregate(4).corrections


def test_empty_track_returns_none(tz_rules, confidence_model):
    agg = TemporalPlateAggregator(tz_rules, confidence_model)
    assert agg.aggregate(99) is None


def test_per_frame_audit_trail(tz_rules, confidence_model):
    agg = TemporalPlateAggregator(tz_rules, confidence_model)
    agg.add(5, _obs(1, "T331E8G", 0.72))
    agg.add(5, _obs(2, "T331EBG", 0.95))
    trail = agg.aggregate(5).per_frame
    assert trail == [(1, "T331E8G", 0.72), (2, "T331EBG", 0.95)]
