"""Confidence fusion, routing thresholds and calibration (spec §16, §23, §28)."""

import pytest

from tz_alpr.postprocessing.confidence import ConfidenceModel, StageScores


@pytest.fixture
def model():
    return ConfidenceModel(
        {
            "weights": {"plate_detection": 0.2, "ocr": 0.55, "plate_validation": 0.25},
            "min_stage_floor": 0.05,
            "ocr_temperature": 1.6,
            "swap_penalty_per_char": 0.03,
            "length_mismatch_penalty": 0.25,
        },
        {"auto_accept": 0.90, "review_band_low": 0.70},
    )


def test_not_a_plain_product(model):
    stages = StageScores(0.0, 0.8, 0.8, 0.8)
    fused = model.fuse(stages)
    assert fused.final_confidence > 0.8 * 0.8 * 0.8  # geometric-mean fusion, not multiply


def test_monotonic_in_ocr(model):
    low = model.fuse(StageScores(0.0, 0.9, 0.5, 0.9)).final_confidence
    high = model.fuse(StageScores(0.0, 0.9, 0.95, 0.9)).final_confidence
    assert high > low


def test_swap_penalty_reduces_confidence(model):
    base = model.fuse(StageScores(0.0, 0.9, 0.9, 0.9), n_swaps=0).final_confidence
    penalised = model.fuse(StageScores(0.0, 0.9, 0.9, 0.9), n_swaps=3).final_confidence
    assert penalised < base


def test_length_mismatch_penalty(model):
    base = model.fuse(StageScores(0.0, 0.9, 0.9, 0.9)).final_confidence
    mism = model.fuse(StageScores(0.0, 0.9, 0.9, 0.9), length_mismatch=True).final_confidence
    assert base - mism == pytest.approx(0.25, abs=0.05)


def test_routing_bands(model):
    assert model.route(0.95) == "auto_accept"
    assert model.route(0.80) == "review"
    assert model.route(0.50) == "manual"


def test_bounds(model):
    for ocr in (0.0, 0.01, 0.5, 0.99, 1.0):
        f = model.fuse(StageScores(0.0, 0.5, ocr, 0.5)).final_confidence
        assert 0.0 <= f <= 1.0


def test_calibration_fit_moves_platt_params(model):
    samples = [(0.95, True)] * 30 + [(0.6, False)] * 30 + [(0.8, True)] * 20 + [(0.4, False)] * 20
    a, b = model.calibrate(samples, iters=200)
    assert a != 1.0 or b != 0.0
    # after fitting, a clearly-good score still routes high
    assert model.fuse(StageScores(0.0, 0.95, 0.97, 0.95)).final_confidence > 0.7


def test_calibration_requires_enough_samples(model):
    with pytest.raises(ValueError):
        model.calibrate([(0.9, True), (0.5, False)])
