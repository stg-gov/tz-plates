"""Video ALPR pipeline (spec §17, §18, §31 Phase 3).

    video -> frame sampling -> vehicle detection -> ByteTrack
          -> per-track plate detection + OCR + Tanzania-aware decode
          -> temporal OCR aggregation -> event deduplication -> VehicleEvent[]

Reuses the components of an :class:`AlprPipeline` (detectors, rectifier, OCR
engine, rule engine, confidence model) so image and video recognition never
diverge. Designed to also back the RTSP ``stream-worker`` in Phase 4: the same
per-frame step is called with live frames instead of decoded ones.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from tz_alpr.config import get_config
from tz_alpr.logging_conf import get_logger
from tz_alpr.pipeline.alpr import AlprPipeline, get_pipeline
from tz_alpr.plate_detection.base import PlateDetection
from tz_alpr.schemas import VideoInfo, VideoResponse
from tz_alpr.tracking import (
    ByteTracker,
    EventManager,
    EventManagerConfig,
    PlateObservation,
    TemporalPlateAggregator,
    utc_now,
)

log = get_logger(__name__)


class VideoPipeline:
    def __init__(
        self,
        alpr: AlprPipeline | None = None,
        tracking_config: str = "configs/tracking.yaml",
    ) -> None:
        self._alpr = alpr or get_pipeline()
        self._cfg = get_config(tracking_config)
        self._sample_fps_default = float(self._cfg.get("sample_fps", 5))
        self._max_plates = int(self._cfg.get("max_plates_per_vehicle", 2))

    # ------------------------------------------------------------------- public
    def process(
        self,
        video_path: str | Path,
        camera_id: str = "upload",
        sample_fps: float | None = None,
        max_seconds: float = 0.0,
        start_datetime: datetime | None = None,
    ) -> VideoResponse:
        t0 = time.perf_counter()
        start_datetime = start_datetime or utc_now()
        sample_fps = float(sample_fps or self._sample_fps_default)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps_source = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps_source / max(sample_fps, 0.01))))

        # When there is no vehicle detector we track plate boxes directly; the
        # plate detector (especially the classical fallback) reports much lower
        # scores than a YOLO vehicle detector, so relax the gates.
        no_vehicle = self._alpr._vehicle_detector is None
        weak_plates = no_vehicle and self._alpr._plate_detector.name == "classical"
        tracker = ByteTracker(
            high_thresh=(
                self._cfg.get("plate_track_high_thresh", 0.15)
                if weak_plates
                else self._cfg.get("track_high_thresh", 0.5)
            ),
            low_thresh=(
                self._cfg.get("plate_track_low_thresh", 0.05)
                if weak_plates
                else self._cfg.get("track_low_thresh", 0.1)
            ),
            match_iou=self._cfg.get("track_match_iou", 0.3),
            max_age=self._cfg.get("track_max_age", 30),
        )
        aggregator = TemporalPlateAggregator(self._alpr._rules, self._alpr._confidence)
        events = EventManager(
            aggregator=aggregator,
            camera_id=camera_id,
            model_version=self._alpr.model_version,
            start_datetime=start_datetime,
            cfg=EventManagerConfig(
                min_frames_for_event=self._cfg.get("event_min_frames", 3),
                emit_min_confidence=self._cfg.get("event_min_confidence", 0.70),
                flush_min_confidence=self._cfg.get("event_flush_min_confidence", 0.45),
                replace_confidence_delta=self._cfg.get("event_replace_delta", 0.08),
                dedup_window_seconds=self._cfg.get("event_dedup_window_s", 120.0),
            ),
        )

        frame_idx = 0
        sampled = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            timestamp = frame_idx / fps_source
            if max_seconds and timestamp > max_seconds:
                break

            self._process_frame(frame, frame_idx, timestamp, tracker, aggregator, events)
            sampled += 1
            frame_idx += 1

        all_track_ids = [t.track_id for t in tracker.finalize()]
        events.flush(all_track_ids)
        cap.release()

        warnings = self._alpr._static_warnings()
        if self._alpr._vehicle_detector is None:
            warnings.append(
                "No vehicle detector: tracking runs on plate boxes directly, which is less "
                "robust to occlusion. Install detection extras for best temporal results."
            )

        return VideoResponse(
            processing_time_ms=int(round((time.perf_counter() - t0) * 1000)),
            model_version=self._alpr.model_version,
            camera_id=camera_id,
            video=VideoInfo(
                duration_s=round(frames_total / fps_source, 2) if frames_total else round(
                    frame_idx / fps_source, 2
                ),
                fps_source=round(fps_source, 3),
                frames_total=frames_total,
                frames_sampled=sampled,
                sample_fps=sample_fps,
            ),
            tracks_seen=len(all_track_ids),
            events=sorted(events.emitted_events, key=lambda e: e.first_seen),
            warnings=warnings,
        )

    # ------------------------------------------------------------------ per frame
    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        tracker: ByteTracker,
        aggregator: TemporalPlateAggregator,
        events: EventManager,
    ) -> None:
        alpr = self._alpr
        proc = alpr._enhancer.enhance(frame) if alpr._auto_enhance else frame

        if alpr._vehicle_detector is not None:
            vdets = alpr._vehicle_detector.detect(proc)
            det_tuples = [(d.bbox_xyxy, d.confidence, d.vehicle_class) for d in vdets]
            track_is_plate = False
        else:
            pdets = alpr._plate_detector.detect(proc, max_plates=alpr._max_plates)
            det_tuples = [(p.bbox_xyxy, p.confidence, "plate") for p in pdets]
            track_is_plate = True

        for track in tracker.update(det_tuples, frame_idx):
            if not track.is_confirmed:
                continue
            obs = self._read_track_plate(proc, track, track_is_plate, frame_idx, timestamp)
            if obs is not None:
                aggregator.add(track.track_id, obs)
                events.consider(track.track_id)

    def _read_track_plate(
        self,
        frame: np.ndarray,
        track,
        track_is_plate: bool,
        frame_idx: int,
        timestamp: float,
    ) -> PlateObservation | None:
        alpr = self._alpr
        h, w = frame.shape[:2]
        tx1, ty1, tx2, ty2 = (int(v) for v in track.bbox_xyxy)
        tx1, ty1 = max(0, tx1), max(0, ty1)
        tx2, ty2 = min(w, tx2), min(h, ty2)
        if tx2 - tx1 < 4 or ty2 - ty1 < 4:
            return None

        vehicle_type = "unknown" if track_is_plate else track.label

        if track_is_plate:
            quad = np.array(
                [[tx1, ty1], [tx2, ty1], [tx2, ty2], [tx1, ty2]], dtype=np.float32
            )
            det = PlateDetection(
                bbox_xyxy=(tx1, ty1, tx2, ty2), confidence=track.score, source="track", quad=quad
            )
        else:
            vcrop = frame[ty1:ty2, tx1:tx2]
            local = alpr._plate_detector.detect(vcrop, max_plates=self._max_plates)
            if not local:
                return None
            best = max(local, key=lambda d: d.confidence)
            bx1, by1, bx2, by2 = best.bbox_xyxy
            quad = best.quad.copy() if best.quad is not None else None
            if quad is not None:
                quad[:, 0] += tx1
                quad[:, 1] += ty1
            det = PlateDetection(
                bbox_xyxy=(bx1 + tx1, by1 + ty1, bx2 + tx1, by2 + ty1),
                confidence=best.confidence,
                source=best.source,
                quad=quad,
            )

        rec = alpr.recognize(frame, det)
        if not rec.tz.raw_ocr and not rec.tz.normalized_text:
            return None

        return PlateObservation(
            frame_idx=frame_idx,
            timestamp=timestamp,
            raw_ocr=rec.tz.raw_ocr,
            normalized_text=rec.tz.normalized_text,
            positions=rec.ocr.positions,
            seq_confidence=rec.ocr.seq_confidence,
            final_confidence=rec.fused.final_confidence,
            plate_det_conf=det.confidence,
            plate_type=rec.tz.category.name,
            corrections=list(rec.tz.corrections),
            vehicle_type=vehicle_type,
        )


_VIDEO_PIPELINE: VideoPipeline | None = None


def get_video_pipeline() -> VideoPipeline:
    global _VIDEO_PIPELINE
    if _VIDEO_PIPELINE is None:
        _VIDEO_PIPELINE = VideoPipeline()
    return _VIDEO_PIPELINE
