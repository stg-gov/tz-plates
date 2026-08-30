"""Vehicle tracking + temporal OCR aggregation + event dedup (spec §17, §18).

Active from Phase 3. ``ByteTracker`` gives stable per-vehicle ids across frames;
``TemporalPlateAggregator`` fuses plate readings over a track with
OCR-probability-weighted voting; ``EventManager`` collapses a track into one
deduplicated ``VehicleEvent``.
"""

from tz_alpr.tracking.aggregator import (
    PlateObservation,
    TemporalPlateAggregator,
    TrackAggregate,
)
from tz_alpr.tracking.bytetrack import ByteTracker, Track
from tz_alpr.tracking.events import EventManager, EventManagerConfig, utc_now

__all__ = [
    "ByteTracker",
    "EventManager",
    "EventManagerConfig",
    "PlateObservation",
    "TemporalPlateAggregator",
    "Track",
    "TrackAggregate",
    "utc_now",
]
