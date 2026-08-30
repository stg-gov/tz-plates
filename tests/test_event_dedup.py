"""Event deduplication: one vehicle dwell => one event, not one per frame (spec §18)."""

from datetime import datetime, timezone

from tz_alpr.ocr.ctc_decode import CharPosterior
from tz_alpr.tracking.aggregator import PlateObservation, TemporalPlateAggregator
from tz_alpr.tracking.events import EventManager, EventManagerConfig

START = datetime(2024, 6, 1, 8, 0, 0, tzinfo=timezone.utc)


def _obs(frame, text, conf):
    return PlateObservation(
        frame_idx=frame,
        timestamp=frame * 0.2,
        raw_ocr=text,
        normalized_text=text,
        positions=[CharPosterior(char=c, prob=conf, alternatives=[(c, conf)]) for c in text],
        seq_confidence=conf,
        final_confidence=conf,
        plate_det_conf=0.9,
        plate_type="PRIVATE",
        vehicle_type="car",
    )


def _manager(cfg=None):
    from tz_alpr.country_rules import get_country_rules
    from tz_alpr.postprocessing.confidence import build_confidence_model

    agg = TemporalPlateAggregator(get_country_rules("TZ"), build_confidence_model())
    mgr = EventManager(
        aggregator=agg,
        camera_id="DODOMA_PARKING_01",
        model_version="tz-alpr-1.2.0",
        start_datetime=START,
        cfg=cfg or EventManagerConfig(),
    )
    return agg, mgr


def test_fifty_frames_one_track_one_event():
    agg, mgr = _manager()
    emitted = []
    for f in range(50):
        agg.add(7, _obs(f, "T331EBG", 0.93))
        ev = mgr.consider(7)
        if ev is not None:
            emitted.append(ev)
    mgr.flush([7])

    assert len(mgr.emitted_events) == 1
    assert len(emitted) == 1
    event = mgr.emitted_events[0]
    assert event.plate == "T331EBG"
    assert event.frame_count == 50
    assert event.camera_id == "DODOMA_PARKING_01"
    assert event.first_seen.startswith("2024-06-01T08:00:00")


def test_low_confidence_track_still_flushes_one_event():
    agg, mgr = _manager(EventManagerConfig(emit_min_confidence=0.9, flush_min_confidence=0.4))
    for f in range(6):
        agg.add(3, _obs(f, "T331EBG", 0.55))
        assert mgr.consider(3) is None  # below emit bar
    events = mgr.flush([3])
    assert len(events) == 1
    assert events[0].review_status in ("manual", "review")


def test_same_plate_from_fragmented_track_is_suppressed():
    agg, mgr = _manager(EventManagerConfig(dedup_window_seconds=120))
    for f in range(5):
        agg.add(1, _obs(f, "T331EBG", 0.92))
    assert mgr.consider(1) is not None
    # a second track, same plate, a few seconds later
    for f in range(10, 15):
        agg.add(2, _obs(f, "T331EBG", 0.92))
    assert mgr.consider(2) is None
    assert len(mgr.emitted_events) == 1


def test_plate_correction_reissues_same_event_id():
    agg, mgr = _manager(EventManagerConfig(replace_confidence_delta=0.05))
    for f in range(4):
        agg.add(1, _obs(f, "T331E8G", 0.72))
    first = mgr.consider(1)
    assert first is not None and first.plate == "T331E8G"

    for f in range(4, 12):
        agg.add(1, _obs(f, "T331EBG", 0.96))
    second = mgr.consider(1)
    assert second is not None
    assert second.plate == "T331EBG"
    assert second.event_id == first.event_id


def test_no_event_below_min_frames():
    agg, mgr = _manager(EventManagerConfig(min_frames_for_event=5))
    for f in range(3):
        agg.add(9, _obs(f, "T331EBG", 0.99))
    assert mgr.consider(9) is None
