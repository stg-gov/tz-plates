"""API contract tests for POST /v1/plate-reader, /health, /version (spec §19, §28).

These exercise the full Phase 1 pipeline with whatever models are present. On a
clean checkout that means the classical detector + NullOcrEngine: the response
must still be well-formed, and `warnings` must flag the missing weights.
"""

import io

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from fastapi.testclient import TestClient

from tz_alpr.api.main import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def _fake_plate_image() -> bytes:
    img = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.rectangle(img, (230, 210), (430, 270), (40, 200, 240), -1)   # yellow plate-ish
    cv2.putText(img, "T331EBG", (240, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "degraded")


def test_version(client):
    r = client.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"]
    assert "ocr" in body["components"]


def test_plate_reader_contract(client):
    r = client.post(
        "/v1/plate-reader",
        files={"upload": ("car.jpg", io.BytesIO(_fake_plate_image()), "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "processing_time_ms" in body
    assert "model_version" in body
    assert isinstance(body["results"], list)
    for res in body["results"]:
        assert {"plate", "raw_text", "raw_ocr", "confidence", "confidence_breakdown",
                "plate_bbox", "country", "plate_type", "review_status"} <= res.keys()
        cb = res["confidence_breakdown"]
        assert {"ocr_confidence", "plate_detection_confidence", "plate_validation_confidence",
                "final_confidence"} <= cb.keys()
        assert 0.0 <= res["confidence"] <= 1.0


def test_plate_reader_rejects_non_image(client):
    r = client.post(
        "/v1/plate-reader",
        files={"upload": ("x.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert r.status_code == 415


def test_plate_reader_rejects_empty(client):
    r = client.post(
        "/v1/plate-reader",
        files={"upload": ("x.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert r.status_code == 400


@pytest.mark.parametrize(
    "method,path",
    [("post", "/v1/streams"), ("get", "/v1/streams/abc"), ("delete", "/v1/streams/abc")],
)
def test_later_phase_endpoints_declared_but_501(client, method, path):
    r = getattr(client, method)(path)
    assert r.status_code == 501


def _tiny_video() -> tuple[bytes, str]:
    import tempfile
    from pathlib import Path

    for suffix, fourcc in ((".avi", "MJPG"), (".mp4", "mp4v"), (".avi", "XVID")):
        path = tempfile.mktemp(suffix=suffix)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), 10.0, (320, 240))
        if not writer.isOpened():
            writer.release()
            continue
        for i in range(20):
            frame = np.full((240, 320, 3), 200, dtype=np.uint8)
            x = 40 + i * 4
            cv2.rectangle(frame, (x, 120), (x + 90, 150), (40, 200, 240), -1)
            cv2.putText(frame, "T331EBG", (x, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            writer.write(frame)
        writer.release()
        data = Path(path).read_bytes()
        if data:
            return data, suffix
    pytest.skip("No usable OpenCV VideoWriter backend in this environment")


def test_video_endpoint_contract(client):
    data, suffix = _tiny_video()
    r = client.post(
        "/v1/video",
        files={"upload": (f"clip{suffix}", io.BytesIO(data), "video/x-msvideo")},
        data={"camera_id": "TEST_CAM", "sample_fps": "5"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["camera_id"] == "TEST_CAM"
    assert body["video"]["frames_sampled"] >= 1
    assert isinstance(body["events"], list)
    assert "model_version" in body


def test_video_endpoint_rejects_bad_sample_fps(client):
    data, suffix = _tiny_video()
    r = client.post(
        "/v1/video",
        files={"upload": (f"clip{suffix}", io.BytesIO(data), "video/x-msvideo")},
        data={"sample_fps": "999"},
    )
    assert r.status_code == 422
