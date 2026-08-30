"""End-to-end ALPR pipeline.

Phase 1 wiring (spec §31):
    image -> enhance -> plate detector -> rectification -> [super-res] -> OCR
          -> Tanzania-aware post-processing -> confidence fusion -> structured result

Phase 2 wiring (spec §3, §4, §31):
    image -> enhance -> vehicle detector -> per-vehicle plate detector
          -> (full-frame safety-net pass) -> rectification -> ... -> result

Each plate is reported with its box in BOTH the original image and its parent
vehicle crop (spec §4). If the vehicle detector is unavailable the pipeline
degrades to the Phase 1 full-frame path; the result contract is unchanged.
Tracking (Phase 3) only adds ``vehicle_track_id`` — nothing here changes.
"""

from __future__ import annotations

import threading
from copy import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from tz_alpr.config import get_config, get_settings
from tz_alpr.country_rules import get_country_rules
from tz_alpr.detection import build_vehicle_detector
from tz_alpr.detection.base import VehicleDetection
from tz_alpr.logging_conf import get_logger
from tz_alpr.ocr.engine import build_ocr_engine
from tz_alpr.plate_detection import build_plate_detector
from tz_alpr.plate_detection.base import PlateDetection
from tz_alpr.postprocessing.confidence import StageScores, build_confidence_model
from tz_alpr.postprocessing.tz_aware import TanzaniaAwareDecoder
from tz_alpr.preprocessing import ImageEnhancer
from tz_alpr.rectification import PlateRectifier
from tz_alpr.schemas import (
    BBox,
    ConfidenceBreakdown,
    OcrCandidate,
    PlateReaderResponse,
    PlateResult,
    VehicleInfo,
)
from tz_alpr.utils.geometry import iou
from tz_alpr.utils.timing import Stopwatch
from tz_alpr.version import COMPONENT_VERSIONS

log = get_logger(__name__)


@dataclass
class Recognition:
    """Result of :meth:`AlprPipeline.recognize` — everything needed to build a
    ``PlateResult`` or a temporal ``PlateObservation``."""

    tz: Any          # TzDecodeResult
    ocr: Any         # OcrPrediction
    fused: Any       # FusedConfidence
    rect: Any        # RectifiedPlate | _RectShim


class AlprPipeline:
    def __init__(self, inference_config: str = "configs/inference.yaml") -> None:
        self.cfg = get_config(inference_config)
        self.settings = get_settings()
        self.model_version = self.settings.model_version or self.cfg.get(
            "model_version", "tz-alpr-1.0.0"
        )

        pcfg = self.cfg.get("pipeline", {})
        self._use_vehicle_detector = bool(pcfg.get("use_vehicle_detector", False))
        self._use_rectification = bool(pcfg.get("use_rectification", True))
        self._use_super_res = bool(
            pcfg.get("use_super_resolution", False) or self.settings.enable_super_resolution
        )
        self._max_plates = int(pcfg.get("max_plates_per_image", 8))
        self._max_vehicles = int(pcfg.get("max_vehicles_per_image", 12))
        self._full_frame_safety_net = bool(pcfg.get("full_frame_safety_net", True))
        self._veh_pad = float(pcfg.get("vehicle_crop_pad_ratio", 0.06))

        prep = self.cfg.get("preprocessing", {})
        self._auto_enhance = bool(prep.get("auto_enhance", True))
        self._enhancer = ImageEnhancer(
            clahe_clip=prep.get("clahe_clip", 2.0),
            clahe_grid=prep.get("clahe_grid", 8),
            denoise=prep.get("denoise", False),
        )

        rcfg = self.cfg.get("rectification", {})
        self._rectifier = PlateRectifier(
            output_width=rcfg.get("output_width", 192),
            output_height=rcfg.get("output_height", 48),
            two_line_output_height=rcfg.get("two_line_output_height", 96),
        )

        self._plate_detector = build_plate_detector(
            self.cfg.get("plate_detector", {}).get("config", "configs/plate_detector.yaml")
        )
        self._ocr = build_ocr_engine(inference_config)
        self._rules = get_country_rules(self.cfg.get("country", "TZ"))
        self._tz_decoder = TanzaniaAwareDecoder(self._rules)
        self._confidence = build_confidence_model(
            self.cfg.get("country_rules_config", "configs/tanzania.yaml")
        )

        self._vehicle_detector = (
            build_vehicle_detector(
                self.cfg.get("vehicle_detector", {}).get("config", "configs/detector.yaml")
            )
            if self._use_vehicle_detector
            else None
        )
        self._sr = self._maybe_build_super_res()

        log.info(
            "pipeline.ready",
            plate_detector=self._plate_detector.name,
            ocr_backend=self._ocr.backend,
            vehicle_detector=(self._vehicle_detector.name if self._vehicle_detector else None),
            super_resolution=bool(self._sr),
        )

    # -------------------------------------------------------------------- public
    def read_image(self, image_bgr: np.ndarray) -> PlateReaderResponse:
        sw = Stopwatch()
        warnings = self._static_warnings()

        with sw("enhance"):
            frame = self._enhancer.enhance(image_bgr) if self._auto_enhance else image_bgr

        detections = self._detect_plates(frame, sw)

        results: list[PlateResult] = []
        for det in detections:
            result = self._process_plate(image_bgr, frame, det, sw)
            if result is not None:
                results.append(result)

        results.sort(key=lambda r: r.confidence, reverse=True)
        self.last_stage_timings = dict(sw.timings)
        return PlateReaderResponse(
            processing_time_ms=int(round(sw.total_ms)),
            model_version=self.model_version,
            results=results,
            warnings=warnings,
        )

    def read_bytes(self, data: bytes) -> PlateReaderResponse:
        from tz_alpr.utils.image_io import decode_image_bytes

        return self.read_image(decode_image_bytes(data))

    # ------------------------------------------------------------ detection stage
    def _detect_plates(self, frame: np.ndarray, sw: Stopwatch) -> list[PlateDetection]:
        if self._vehicle_detector is None:
            with sw("plate_detection"):
                return self._plate_detector.detect(frame, max_plates=self._max_plates)

        with sw("vehicle_detection"):
            vehicles = self._vehicle_detector.detect(frame)
        vehicles.sort(key=lambda v: v.confidence, reverse=True)
        vehicles = vehicles[: self._max_vehicles]

        hits: list[PlateDetection] = []
        with sw("plate_detection"):
            for veh in vehicles:
                vx1, vy1, vx2, vy2 = _pad_box(veh.bbox_xyxy, self._veh_pad, frame.shape)
                vcrop = frame[vy1:vy2, vx1:vx2]
                if vcrop.size == 0:
                    continue
                for pd in self._plate_detector.detect(vcrop, max_plates=self._max_plates):
                    hit = _translate_detection(pd, vx1, vy1)
                    hit.extra["vehicle"] = veh
                    hit.extra["bbox_in_vehicle"] = _rel_box(hit.bbox_xyxy, veh.bbox_xyxy)
                    hits.append(hit)

            if self._full_frame_safety_net or not hits:
                for pd in self._plate_detector.detect(frame, max_plates=self._max_plates):
                    if _best_iou(pd.bbox_xyxy, hits) < 0.3:
                        pd.extra.setdefault("vehicle", None)
                        hits.append(pd)

        hits = _dedupe_by_iou(hits, thresh=0.45)[: self._max_plates]

        for hit in hits:
            if hit.extra.get("vehicle") is None:
                veh = _containing_vehicle(hit.bbox_xyxy, vehicles)
                if veh is not None:
                    hit.extra["vehicle"] = veh
                    hit.extra["bbox_in_vehicle"] = _rel_box(hit.bbox_xyxy, veh.bbox_xyxy)
        return hits

    # --------------------------------------------------------- recognise one plate
    def recognize(
        self, frame: np.ndarray, det: PlateDetection, sw: Stopwatch | None = None
    ) -> Recognition:
        """Rectify -> [super-res] -> OCR -> Tanzania-aware decode -> confidence fusion
        for a single plate detection. Shared by the image and the video pipelines."""
        sw = sw or Stopwatch()
        veh: VehicleDetection | None = det.extra.get("vehicle")

        with sw("rectification"):
            rect = (
                self._rectifier.rectify(frame, det)
                if self._use_rectification
                else _RectShim(_crop(frame, det.bbox_xyxy))
            )
        plate_img = rect.image

        if self._sr is not None and plate_img.shape[1] < self._sr_threshold:
            with sw("super_resolution"):
                plate_img = self._sr(plate_img)

        with sw("ocr"):
            ocr_pred = self._ocr.predict(plate_img)

        with sw("postprocess"):
            tz = self._tz_decoder.decode(
                ocr_pred.raw_text, ocr_pred.positions, ocr_pred.seq_confidence
            )
            validation_conf = self._rules.validation_confidence(
                tz.normalized_text, tz.category, tz.n_swaps
            )
            length_mismatch = bool(
                tz.category.slots and len(tz.category.slots) != len(tz.normalized_text)
            )
            fused = self._confidence.fuse(
                StageScores(
                    vehicle=(veh.confidence if veh else 0.0),
                    plate_detection=det.confidence,
                    ocr_seq=ocr_pred.seq_confidence,
                    plate_validation=validation_conf,
                ),
                n_swaps=tz.n_swaps,
                length_mismatch=length_mismatch,
            )
        return Recognition(tz=tz, ocr=ocr_pred, fused=fused, rect=rect)

    # ------------------------------------------------------------- per-plate work
    def _process_plate(
        self, original: np.ndarray, frame: np.ndarray, det: PlateDetection, sw: Stopwatch
    ) -> PlateResult | None:
        veh: VehicleDetection | None = det.extra.get("vehicle")
        bbox_in_vehicle = det.extra.get("bbox_in_vehicle")

        rec = self.recognize(frame, det, sw)
        tz, fused, rect = rec.tz, rec.fused, rec.rect

        raw_spaced = self._space_like(tz.raw_ocr, tz.category)
        plate_str = tz.normalized_text or tz.raw_ocr

        return PlateResult(
            plate=plate_str,
            raw_text=raw_spaced,
            raw_ocr=tz.raw_ocr,
            normalized_text=tz.normalized_text,
            confidence=fused.final_confidence,
            confidence_breakdown=ConfidenceBreakdown(
                vehicle_confidence=fused.vehicle_confidence,
                plate_detection_confidence=fused.plate_detection_confidence,
                ocr_confidence=fused.ocr_confidence,
                plate_validation_confidence=fused.plate_validation_confidence,
                final_confidence=fused.final_confidence,
            ),
            plate_bbox=BBox.from_xyxy(*det.bbox_xyxy),
            plate_bbox_in_vehicle=(BBox.from_xyxy(*bbox_in_vehicle) if bbox_in_vehicle else None),
            plate_quad=(
                rect.quad_image_coords.astype(int).tolist()
                if getattr(rect, "quad_image_coords", None) is not None
                else (det.quad.astype(int).tolist() if det.quad is not None else None)
            ),
            vehicle=(
                VehicleInfo(
                    type=veh.vehicle_class,
                    confidence=round(veh.confidence, 4),
                    bbox=BBox.from_xyxy(*veh.bbox_xyxy),
                )
                if veh
                else VehicleInfo(type="unknown", confidence=0.0)
            ),
            vehicle_track_id=(veh.track_id if veh else None),
            country=self._rules.country_code,
            plate_type=tz.category.name,
            plate_colour=tz.category.plate_colour,
            plate_lines=tz.category.lines,
            corrections=tz.corrections,
            ocr_candidates=[
                OcrCandidate(text=t, confidence=round(c, 4)) for t, c in tz.ocr_candidates
            ],
            review_status=fused.review_status,
        )

    # ------------------------------------------------------------------- helpers
    def _static_warnings(self) -> list[str]:
        warnings: list[str] = []
        if not self._ocr.available:
            warnings.append(
                "OCR model weights are not loaded (NullOcrEngine). Plate strings are placeholders; "
                "train and export the OCR model, then set TZ_ALPR_OCR_WEIGHTS / TZ_ALPR_OCR_ONNX."
            )
        if self._plate_detector.name == "classical":
            warnings.append(
                "Using the classical plate-detector fallback. Train the YOLO plate detector and set "
                "TZ_ALPR_PLATE_DETECTOR_WEIGHTS for production accuracy."
            )
        if self._use_vehicle_detector and self._vehicle_detector is None:
            warnings.append(
                "Vehicle detector unavailable; running full-frame plate detection. "
                'Install detection extras: pip install -e ".[detect]".'
            )
        return warnings

    def _space_like(self, text: str, category) -> str:
        if not text:
            return ""
        spaced = self._rules._format_display(text, category)
        return spaced if spaced != text else text

    def _maybe_build_super_res(self):
        if not self._use_super_res:
            return None
        sr_cfg = self.cfg.get("super_resolution", {})
        self._sr_threshold = int(sr_cfg.get("apply_if_plate_width_below_px", 70))
        try:
            from tz_alpr.pipeline.super_resolution import build_super_resolver

            return build_super_resolver(sr_cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("pipeline.super_res_unavailable", error=str(exc))
            return None

    @property
    def component_versions(self) -> dict[str, str]:
        return {
            **COMPONENT_VERSIONS,
            "plate_detector_backend": self._plate_detector.name,
            "vehicle_detector_backend": (
                self._vehicle_detector.name if self._vehicle_detector else "none"
            ),
            "ocr_backend": self._ocr.backend,
        }


class _RectShim:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.quad_image_coords = None
        self.is_two_line = False
        self.method = "crop"


def _crop(img: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else img


def _pad_box(
    xyxy: tuple[int, int, int, int], ratio: float, shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = xyxy
    dx, dy = int((x2 - x1) * ratio), int((y2 - y1) * ratio)
    return (max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy))


def _translate_detection(det: PlateDetection, dx: int, dy: int) -> PlateDetection:
    x1, y1, x2, y2 = det.bbox_xyxy
    quad = None
    if det.quad is not None:
        quad = det.quad.copy()
        quad[:, 0] += dx
        quad[:, 1] += dy
    return PlateDetection(
        bbox_xyxy=(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
        confidence=det.confidence,
        source=det.source,
        quad=quad,
        extra=copy(det.extra),
    )


def _rel_box(
    box_xyxy: tuple[int, int, int, int], ref_xyxy: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    bx1, by1, bx2, by2 = box_xyxy
    rx1, ry1 = ref_xyxy[0], ref_xyxy[1]
    return (max(0, bx1 - rx1), max(0, by1 - ry1), max(0, bx2 - rx1), max(0, by2 - ry1))


def _best_iou(box: tuple[int, int, int, int], others: list[PlateDetection]) -> float:
    return max((iou(box, o.bbox_xyxy) for o in others), default=0.0)


def _dedupe_by_iou(dets: list[PlateDetection], thresh: float) -> list[PlateDetection]:
    ordered = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept: list[PlateDetection] = []
    for det in ordered:
        if any(iou(det.bbox_xyxy, k.bbox_xyxy) >= thresh for k in kept):
            # keep the richer record if the survivor has no vehicle but this one does
            for k in kept:
                if (
                    iou(det.bbox_xyxy, k.bbox_xyxy) >= thresh
                    and k.extra.get("vehicle") is None
                    and det.extra.get("vehicle") is not None
                ):
                    k.extra["vehicle"] = det.extra["vehicle"]
                    k.extra["bbox_in_vehicle"] = det.extra.get("bbox_in_vehicle")
            continue
        kept.append(det)
    return kept


def _containing_vehicle(
    box_xyxy: tuple[int, int, int, int], vehicles: list[VehicleDetection]
) -> VehicleDetection | None:
    cx = (box_xyxy[0] + box_xyxy[2]) / 2
    cy = (box_xyxy[1] + box_xyxy[3]) / 2
    best: VehicleDetection | None = None
    best_area = float("inf")
    for veh in vehicles:
        vx1, vy1, vx2, vy2 = veh.bbox_xyxy
        if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
            area = (vx2 - vx1) * (vy2 - vy1)
            if area < best_area:
                best, best_area = veh, area
    return best


_PIPELINE: AlprPipeline | None = None
_LOCK = threading.Lock()


def get_pipeline(inference_config: str = "configs/inference.yaml") -> AlprPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        with _LOCK:
            if _PIPELINE is None:
                _PIPELINE = AlprPipeline(inference_config)
    return _PIPELINE
