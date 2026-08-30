"""ByteTrack-style multi-object tracker (spec §17).

A dependency-free implementation of the ByteTrack idea: associate high-score
detections first, then use the *low-score* detections to recover tracks that
would otherwise be lost to occlusion / motion blur. Motion is predicted with an
EMA-smoothed constant-velocity model instead of a Kalman filter — adequate at the
few-FPS sampling rates used for parking cameras and keeps the module portable
(no scipy/lap).

Works with any ``VehicleDetector`` output (or plate detections) — it only needs
boxes + scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tz_alpr.utils.geometry import iou

_TRACKED = "tracked"
_LOST = "lost"
_REMOVED = "removed"


@dataclass
class Track:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    label: str = "vehicle"
    state: str = _TRACKED
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    start_frame: int = 0
    last_frame: int = 0
    _velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 2 and self.state == _TRACKED

    def predict(self) -> tuple[float, float, float, float]:
        vx1, vy1, vx2, vy2 = self._velocity
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + vx1, y1 + vy1, x2 + vx2, y2 + vy2)

    def update(self, bbox: tuple[float, float, float, float], score: float, frame: int) -> None:
        ox1, oy1, ox2, oy2 = self.bbox_xyxy
        nx1, ny1, nx2, ny2 = bbox
        alpha = 0.5
        self._velocity = (
            alpha * (nx1 - ox1) + (1 - alpha) * self._velocity[0],
            alpha * (ny1 - oy1) + (1 - alpha) * self._velocity[1],
            alpha * (nx2 - ox2) + (1 - alpha) * self._velocity[2],
            alpha * (ny2 - oy2) + (1 - alpha) * self._velocity[3],
        )
        self.bbox_xyxy = bbox
        self.score = score
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.state = _TRACKED
        self.last_frame = frame

    def mark_missed(self) -> None:
        self.age += 1
        self.time_since_update += 1
        if self.state == _TRACKED:
            self.state = _LOST


@dataclass
class _Det:
    bbox: tuple[float, float, float, float]
    score: float
    label: str


@dataclass
class ByteTracker:
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    match_iou: float = 0.3
    max_age: int = 30
    min_box_area: float = 64.0
    _next_id: int = field(default=1, init=False)
    _tracks: list[Track] = field(default_factory=list, init=False)

    def reset(self) -> None:
        self._next_id = 1
        self._tracks = []

    @property
    def tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.state != _REMOVED]

    def update(self, detections: list[tuple[tuple[float, float, float, float], float, str]],
               frame_idx: int) -> list[Track]:
        dets = [
            _Det(tuple(map(float, b)), float(s), lbl)
            for (b, s, lbl) in detections
            if _area(b) >= self.min_box_area
        ]
        high = [d for d in dets if d.score >= self.high_thresh]
        low = [d for d in dets if self.low_thresh <= d.score < self.high_thresh]

        active = [t for t in self._tracks if t.state != _REMOVED]
        for t in active:
            t.bbox_xyxy = t.predict()

        # --- association pass 1: confirmed/tracked tracks vs high-score dets
        matches1, un_tracks1, un_high = _associate(active, high, self.match_iou)
        for ti, di in matches1:
            active[ti].update(high[di].bbox, high[di].score, frame_idx)
            active[ti].label = high[di].label

        # --- association pass 2: remaining tracks vs low-score dets
        remaining = [active[i] for i in un_tracks1]
        matches2, _un_tracks2, _un_low = _associate(remaining, low, self.match_iou * 0.7)
        for ti, di in matches2:
            remaining[ti].update(low[di].bbox, low[di].score, frame_idx)

        matched_remaining = {ti for ti, _ in matches2}
        for idx, t in enumerate(remaining):
            if idx not in matched_remaining:
                t.mark_missed()

        # --- new tracks from unmatched high-score dets
        for di in un_high:
            d = high[di]
            self._tracks.append(
                Track(
                    track_id=self._next_id,
                    bbox_xyxy=d.bbox,
                    score=d.score,
                    label=d.label,
                    start_frame=frame_idx,
                    last_frame=frame_idx,
                )
            )
            self._next_id += 1

        for t in self._tracks:
            if t.time_since_update > self.max_age:
                t.state = _REMOVED

        return [t for t in self._tracks if t.state == _TRACKED]

    def finalize(self) -> list[Track]:
        """Return every track ever seen (for end-of-video event flushing)."""
        return [t for t in self._tracks]


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _associate(tracks: list[Track], dets: list[_Det], iou_thresh: float):
    """Greedy IoU matching. Returns (matches, unmatched_track_idx, unmatched_det_idx)."""
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))

    pairs = []
    for ti, t in enumerate(tracks):
        for di, d in enumerate(dets):
            score = iou(t.bbox_xyxy, d.bbox)
            if score >= iou_thresh:
                pairs.append((score, ti, di))
    pairs.sort(reverse=True)

    matched_t: set[int] = set()
    matched_d: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, ti, di in pairs:
        if ti in matched_t or di in matched_d:
            continue
        matched_t.add(ti)
        matched_d.add(di)
        matches.append((ti, di))

    un_t = [i for i in range(len(tracks)) if i not in matched_t]
    un_d = [i for i in range(len(dets)) if i not in matched_d]
    return matches, un_t, un_d
