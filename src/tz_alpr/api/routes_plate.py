"""Plate-reader + video endpoints (spec §19).

Phase 1 implements ``POST /v1/plate-reader``. Phase 3 implements ``POST /v1/video``
(synchronous; a job queue is Phase 4). The stream endpoints keep their final
contract but return ``501`` until Phase 4 — the OpenAPI schema stays stable so
clients can be built now.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from tz_alpr.api.deps import pipeline_dependency, settings_dependency
from tz_alpr.config import Settings
from tz_alpr.logging_conf import get_logger
from tz_alpr.pipeline import AlprPipeline
from tz_alpr.schemas import PlateReaderResponse, VideoResponse

router = APIRouter(prefix="/v1", tags=["alpr"])
log = get_logger(__name__)

_ALLOWED_IMAGE_PREFIXES = ("image/",)
_ALLOWED_VIDEO_PREFIXES = ("video/", "application/octet-stream")


@router.post("/plate-reader", response_model=PlateReaderResponse)
async def plate_reader(
    upload: UploadFile = File(..., description="Vehicle or plate image (JPEG/PNG/WebP)"),
    pipeline: AlprPipeline = Depends(pipeline_dependency),
    settings: Settings = Depends(settings_dependency),
) -> PlateReaderResponse:
    if upload.content_type and not upload.content_type.startswith(_ALLOWED_IMAGE_PREFIXES):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content-type: {upload.content_type}",
        )

    data = await upload.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty upload")
    if len(data) > max_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Image exceeds {settings.max_upload_mb} MB limit",
        )

    try:
        response = pipeline.read_bytes(data)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.error("plate_reader.failed", error=str(exc))
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Inference failed") from exc

    log.info(
        "plate_reader.ok",
        filename=upload.filename,
        n_results=len(response.results),
        ms=response.processing_time_ms,
    )
    return response


@router.post("/video", response_model=VideoResponse)
async def process_video(
    upload: UploadFile = File(..., description="Video file (mp4/mov/mkv/avi)"),
    camera_id: str = Form("upload"),
    sample_fps: float = Form(5.0),
    max_seconds: float = Form(0.0, description="0 = process the whole clip"),
    settings: Settings = Depends(settings_dependency),
) -> VideoResponse:
    if upload.content_type and not upload.content_type.startswith(_ALLOWED_VIDEO_PREFIXES):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported content-type: {upload.content_type}",
        )

    data = await upload.read()
    max_bytes = settings.max_upload_mb * 8 * 1024 * 1024  # videos: 8x the image limit
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty upload")
    if len(data) > max_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Video exceeds {settings.max_upload_mb * 8} MB limit",
        )
    if sample_fps <= 0 or sample_fps > 30:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "sample_fps must be in (0, 30]")

    from tz_alpr.pipeline.video import get_video_pipeline

    suffix = Path(upload.filename or "clip.mp4").suffix or ".mp4"
    tmp = Path(tempfile.mkstemp(suffix=suffix)[1])
    try:
        tmp.write_bytes(data)
        response = get_video_pipeline().process(
            tmp, camera_id=camera_id, sample_fps=sample_fps, max_seconds=max_seconds
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.error("process_video.failed", error=str(exc))
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Video processing failed") from exc
    finally:
        tmp.unlink(missing_ok=True)

    log.info(
        "process_video.ok",
        camera_id=camera_id,
        events=len(response.events),
        tracks=response.tracks_seen,
        frames=response.video.frames_sampled,
        ms=response.processing_time_ms,
    )
    return response


_NOT_YET = "Available from Phase 4 (RTSP stream-worker). The endpoint contract is frozen."


@router.post("/streams", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def register_stream() -> dict:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, _NOT_YET)


@router.get("/streams/{stream_id}", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def get_stream(stream_id: str) -> dict:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, _NOT_YET)


@router.delete("/streams/{stream_id}", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def delete_stream(stream_id: str) -> dict:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, _NOT_YET)
