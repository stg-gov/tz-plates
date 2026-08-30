"""ByteTracker: id stability, recovery from low-score dets, track death (spec §17)."""

from tz_alpr.tracking.bytetrack import ByteTracker


def _box(x, y, w=120, h=90):
    return (float(x), float(y), float(x + w), float(y + h))


def test_single_object_keeps_one_id_across_frames():
    tr = ByteTracker(high_thresh=0.5, match_iou=0.3)
    ids = []
    for i in range(8):
        tracks = tr.update([(_box(100 + i * 8, 200), 0.9, "car")], frame_idx=i)
        assert len(tracks) == 1
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1
    assert tracks[0].hits == 8
    assert tracks[0].is_confirmed


def test_two_non_crossing_objects_get_stable_distinct_ids():
    tr = ByteTracker()
    seen = set()
    for i in range(6):
        tracks = tr.update(
            [(_box(50 + i * 5, 100), 0.9, "car"), (_box(600 - i * 5, 400), 0.85, "truck")],
            frame_idx=i,
        )
        assert len(tracks) == 2
        seen |= {t.track_id for t in tracks}
    assert seen == {1, 2}


def test_low_score_detection_recovers_a_track():
    tr = ByteTracker(high_thresh=0.5, low_thresh=0.1, match_iou=0.3)
    tr.update([(_box(100, 200), 0.9, "car")], frame_idx=0)
    tr.update([(_box(108, 200), 0.9, "car")], frame_idx=1)
    # now only a weak detection — must still update the existing track, not spawn a new one
    tracks = tr.update([(_box(116, 200), 0.2, "car")], frame_idx=2)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].hits == 3


def test_track_is_removed_after_max_age():
    tr = ByteTracker(high_thresh=0.5, max_age=3)
    for i in range(3):
        tr.update([(_box(100, 200), 0.9, "car")], frame_idx=i)
    for i in range(3, 10):
        tr.update([], frame_idx=i)
    assert tr.update([], frame_idx=11) == []
    assert all(t.state == "removed" for t in tr.finalize())


def test_only_high_score_detections_start_tracks():
    tr = ByteTracker(high_thresh=0.5, low_thresh=0.1)
    assert tr.update([(_box(10, 10), 0.3, "car")], frame_idx=0) == []
    assert tr.finalize() == []


def test_tiny_boxes_are_ignored():
    tr = ByteTracker(min_box_area=64.0)
    assert tr.update([((0.0, 0.0, 3.0, 3.0), 0.99, "car")], frame_idx=0) == []
