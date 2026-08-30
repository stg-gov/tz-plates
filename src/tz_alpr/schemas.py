"""Public data shapes: pipeline results and API responses.

These are the contract for ``POST /v1/plate-reader`` (spec §19) and are reused
internally so the pipeline and the API never drift apart.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> BBox:
        return cls(x=int(round(x1)), y=int(round(y1)),
                   width=int(round(x2 - x1)), height=int(round(y2 - y1)))

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height


class VehicleInfo(BaseModel):
    type: str = "unknown"
    confidence: float = 0.0
    bbox: BBox | None = None


class OcrCandidate(BaseModel):
    """One alternative reading, surfaced for human verification (spec §23)."""

    text: str
    confidence: float


class ConfidenceBreakdown(BaseModel):
    """All the intermediate scores, never collapsed to a single opaque number (spec §16)."""

    vehicle_confidence: float = 0.0
    plate_detection_confidence: float = 0.0
    ocr_confidence: float = 0.0
    plate_validation_confidence: float = 0.0
    final_confidence: float = 0.0


class PlateResult(BaseModel):
    plate: str = Field(description="Normalized registration, e.g. T331EBG")
    raw_text: str = Field(description="OCR output before normalization, e.g. 'T 331 EBG'")
    raw_ocr: str = Field(description="Un-spaced raw OCR string, e.g. T33IEBG")
    normalized_text: str = Field(description="Alias of `plate`, kept for clarity")

    confidence: float
    confidence_breakdown: ConfidenceBreakdown

    plate_bbox: BBox
    plate_bbox_in_vehicle: BBox | None = None
    plate_quad: list[list[int]] | None = Field(
        default=None, description="4 corner points of the plate in image coords"
    )

    vehicle: VehicleInfo = Field(default_factory=VehicleInfo)
    vehicle_track_id: int | None = Field(
        default=None, description="Stable track id across frames (populated from Phase 3)"
    )

    country: str = "TZ"
    plate_type: str = "UNKNOWN"
    plate_colour: str = "unknown"
    plate_lines: int = 1

    corrections: list[str] = Field(
        default_factory=list,
        description="Human-readable list of every rule swap applied; corrections are never hidden (spec §7)",
    )
    ocr_candidates: list[OcrCandidate] = Field(default_factory=list)

    review_status: Literal["auto_accept", "review", "manual"] = "manual"


class PlateReaderResponse(BaseModel):
    processing_time_ms: int
    model_version: str
    results: list[PlateResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class VehicleEvent(BaseModel):
    """One deduplicated vehicle observation from a video / stream (spec §17, §18).

    One physical vehicle passing a camera produces exactly one of these, not one
    per frame. Field names match the webhook payload (spec §20).
    """

    event_id: str
    event: Literal["vehicle.detected"] = "vehicle.detected"
    camera_id: str
    track_id: int
    plate: str
    raw_ocr: str
    normalized_text: str
    confidence: float
    plate_type: str = "UNKNOWN"
    vehicle_type: str = "unknown"
    review_status: Literal["auto_accept", "review", "manual"] = "manual"
    corrections: list[str] = Field(default_factory=list)
    first_seen: str = Field(description="ISO-8601 timestamp of first frame in the track")
    last_seen: str = Field(description="ISO-8601 timestamp of last frame in the track")
    frame_count: int = 0
    model_version: str = ""
    per_frame: list[list] = Field(
        default_factory=list, description="[frame_idx, reading, confidence] audit trail"
    )


class VideoInfo(BaseModel):
    duration_s: float
    fps_source: float
    frames_total: int
    frames_sampled: int
    sample_fps: float


class VideoResponse(BaseModel):
    processing_time_ms: int
    model_version: str
    camera_id: str
    video: VideoInfo
    tracks_seen: int
    events: list[VehicleEvent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    detail: str = ""


class VersionResponse(BaseModel):
    package_version: str
    model_version: str
    components: dict[str, str]
