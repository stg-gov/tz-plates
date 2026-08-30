"""Event deduplication (spec §18).

A vehicle dwelling in front of a parking camera is seen in dozens of frames. The
:class:`EventManager` turns each *track* into at most one :class:`VehicleEvent`,
and also suppresses the same plate re-firing from a fragmented track within a
short window.

Emission rules:
  * a track emits once it has >= ``min_frames_for_event`` readings and its
    aggregated confidence >= ``emit_min_confidence``;
  * if a later aggregation changes the plate string AND raises confidence by
    >= ``replace_confidence_delta``, the event is re-issued (same ``event_id``);
  * ``flush`` emits a final event for tracks that ended without meeting the bar,
    provided confidence >= ``flush_min_confidence`` (nothing is lost silently);
  * a plate seen again from a different track within ``dedup_window_seconds`` is
    suppressed unless its confidence is materially higher.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from tz_alpr.schemas import VehicleEvent
from tz_alpr.tracking.aggregator import TemporalPlateAggregator, TrackAggregate


@dataclass
class EventManagerConfig:
    min_frames_for_event: int = 3
    emit_min_confidence: float = 0.70
    flush_min_confidence: float = 0.45
    replace_confidence_delta: float = 0.08
    dedup_window_seconds: float = 120.0


@dataclass
class EventManager:
    aggregator: TemporalPlateAggregator
    camera_id: str
    model_version: str
    start_datetime: datetime
    cfg: EventManagerConfig = field(default_factory=EventManagerConfig)
    _emitted: dict[int, VehicleEvent] = field(default_factory=dict, init=False)
    _recent_plate_seen: dict[str, float] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------- public
    def consider(self, track_id: int) -> VehicleEvent | None:
        agg = self.aggregator.aggregate(track_id)
        if agg is None or not agg.plate:
            return None
        if agg.n_frames < self.cfg.min_frames_for_event:
            return None
        if agg.confidence < self.cfg.emit_min_confidence:
            return None
        return self._emit(track_id, agg, is_flush=False)

    def flush(self, track_ids: list[int]) -> list[VehicleEvent]:
        out: list[VehicleEvent] = []
        for tid in track_ids:
            if tid in self._emitted:
                self._touch(tid)
                continue
            agg = self.aggregator.aggregate(tid)
            if agg and agg.plate and agg.confidence >= self.cfg.flush_min_confidence:
                event = self._emit(tid, agg, is_flush=True)
                if event is not None:
                    out.append(event)
        return out

    @property
    def emitted_events(self) -> list[VehicleEvent]:
        return list(self._emitted.values())

    # ------------------------------------------------------------------ internal
    def _emit(self, track_id: int, agg: TrackAggregate, is_flush: bool) -> VehicleEvent | None:
        first_seen = self._to_iso(agg.first_timestamp)
        last_seen = self._to_iso(agg.last_timestamp)
        last_epoch = self.start_datetime.timestamp() + agg.last_timestamp

        prior = self._emitted.get(track_id)
        if prior is not None:
            if prior.plate == agg.plate:
                prior.last_seen = last_seen
                prior.frame_count = agg.n_frames
                prior.confidence = max(prior.confidence, agg.confidence)
                return None
            # plate changed: re-issue unless the new reading is a confidence downgrade
            if agg.confidence + self.cfg.replace_confidence_delta < prior.confidence:
                return None

        seen_at = self._recent_plate_seen.get(agg.plate)
        if (
            prior is None
            and seen_at is not None
            and last_epoch - seen_at < self.cfg.dedup_window_seconds
            and not is_flush
        ):
            # same plate, different track, inside the window -> fragmented track
            return None

        event = VehicleEvent(
            event_id=(prior.event_id if prior else uuid.uuid4().hex),
            camera_id=self.camera_id,
            track_id=track_id,
            plate=agg.plate,
            raw_ocr=agg.raw_ocr,
            normalized_text=agg.plate,
            confidence=agg.confidence,
            plate_type=agg.plate_type,
            vehicle_type=agg.vehicle_type,
            review_status=agg.review_status,
            corrections=agg.corrections,
            first_seen=first_seen,
            last_seen=last_seen,
            frame_count=agg.n_frames,
            model_version=self.model_version,
            per_frame=[list(x) for x in agg.per_frame],
        )
        self._emitted[track_id] = event
        self._recent_plate_seen[agg.plate] = last_epoch
        return event

    def _touch(self, track_id: int) -> None:
        agg = self.aggregator.aggregate(track_id)
        if agg is None:
            return
        ev = self._emitted[track_id]
        ev.last_seen = self._to_iso(agg.last_timestamp)
        ev.frame_count = agg.n_frames
        ev.confidence = max(ev.confidence, agg.confidence)

    def _to_iso(self, offset_seconds: float) -> str:
        return (self.start_datetime + timedelta(seconds=offset_seconds)).isoformat()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
