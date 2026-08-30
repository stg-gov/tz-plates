"""Phase 2: vehicle detection -> per-vehicle plate detection (spec §3, §4).

Uses stub detectors injected into a real ``AlprPipeline`` so the coordinate
translation, vehicle association and multi-vehicle handling are tested without
needing YOLO weights.
"""

import numpy as np
import pytest

from tz_alpr.detection.base import VehicleDetection, VehicleDetector
from tz_alpr.pipeline.alpr import AlprPipeline
from tz_alpr.plate_detection.base import PlateDetection, PlateDetector


class StubVehicleDetector(VehicleDetector):
    name = "stub"

    def __init__(self, boxes):
        self._boxes = boxes

    def detect(self, image):
        return [
            VehicleDetection(bbox_xyxy=b, vehicle_class=c, confidence=conf)
            for (b, c, conf) in self._boxes
        ]


class StubPlateDetector(PlateDetector):
    """Returns one plate box at a fixed offset inside whatever image it is given."""

    name = "stub"

    def detect(self, image, max_plates: int = 8):
        h, w = image.shape[:2]
        x1, y1 = 10, 10
        x2, y2 = min(w - 1, 70), min(h - 1, 30)
        quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        return [PlateDetection(bbox_xyxy=(x1, y1, x2, y2), confidence=0.8, source="stub", quad=quad)]


@pytest.fixture
def pipeline():
    p = AlprPipeline("configs/inference.yaml")
    p._plate_detector = StubPlateDetector()
    p._full_frame_safety_net = False
    return p


def _frame():
    return np.full((720, 1280, 3), 180, dtype=np.uint8)


def test_plate_box_is_translated_to_image_coords(pipeline):
    pipeline._vehicle_detector = StubVehicleDetector([((400, 300, 700, 560), "car", 0.92)])
    resp = pipeline.read_image(_frame())

    assert len(resp.results) == 1
    r = resp.results[0]
    # padded crop origin is <= vehicle origin, plus the stub's (10,10) offset
    assert r.plate_bbox.x < 400 + 20 and r.plate_bbox.x >= 400 - 40
    assert r.plate_bbox.y < 300 + 20
    assert r.plate_bbox_in_vehicle is not None
    assert r.plate_bbox_in_vehicle.x < r.plate_bbox.x  # vehicle-relative is smaller
    assert r.plate_quad is not None and len(r.plate_quad) == 4


def test_vehicle_attributes_attached(pipeline):
    pipeline._vehicle_detector = StubVehicleDetector([((100, 100, 500, 460), "truck", 0.87)])
    r = pipeline.read_image(_frame()).results[0]

    assert r.vehicle.type == "truck"
    assert r.vehicle.confidence == pytest.approx(0.87, abs=1e-3)
    assert r.vehicle.bbox is not None and r.vehicle.bbox.width == 400
    assert r.confidence_breakdown.vehicle_confidence == pytest.approx(0.87, abs=1e-3)


def test_multiple_vehicles_yield_multiple_plates(pipeline):
    pipeline._vehicle_detector = StubVehicleDetector(
        [
            ((50, 50, 350, 300), "car", 0.9),
            ((700, 400, 1100, 700), "motorcycle", 0.8),
            ((360, 60, 640, 320), "car", 0.75),
        ]
    )
    resp = pipeline.read_image(_frame())
    assert len(resp.results) == 3
    types = sorted(r.vehicle.type for r in resp.results)
    assert types == ["car", "car", "motorcycle"]


def test_falls_back_to_full_frame_when_no_vehicle_detector(pipeline):
    pipeline._vehicle_detector = None
    r = pipeline.read_image(_frame()).results[0]
    assert r.vehicle.type == "unknown"
    assert r.plate_bbox_in_vehicle is None
    assert r.plate_bbox.x == 10 and r.plate_bbox.y == 10  # untranslated full-frame coords


def test_no_vehicles_detected_still_scans_full_frame(pipeline):
    pipeline._vehicle_detector = StubVehicleDetector([])
    resp = pipeline.read_image(_frame())
    assert len(resp.results) == 1
    assert resp.results[0].vehicle.type == "unknown"


def test_safety_net_associates_plate_with_containing_vehicle():
    p = AlprPipeline("configs/inference.yaml")
    p._plate_detector = StubPlateDetector()
    p._full_frame_safety_net = True
    # vehicle box contains the full-frame plate detection at (10,10,70,30)
    p._vehicle_detector = StubVehicleDetector([((0, 0, 200, 200), "car", 0.6)])
    r = p.read_image(_frame()).results[0]
    assert r.vehicle.type == "car"
    assert r.plate_bbox_in_vehicle is not None
