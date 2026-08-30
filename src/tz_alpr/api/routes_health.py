"""Health & version endpoints (spec §19, §30)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from tz_alpr.api.deps import pipeline_dependency
from tz_alpr.pipeline import AlprPipeline
from tz_alpr.schemas import HealthResponse, VersionResponse
from tz_alpr.version import MODEL_VERSION, __version__

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(pipeline: AlprPipeline = Depends(pipeline_dependency)) -> HealthResponse:
    ocr_ready = pipeline._ocr.available
    detector_trained = pipeline._plate_detector.name != "classical"
    if ocr_ready and detector_trained:
        return HealthResponse(status="ok", detail="all models loaded")
    missing = []
    if not ocr_ready:
        missing.append("ocr-weights")
    if not detector_trained:
        missing.append("trained-plate-detector")
    return HealthResponse(status="degraded", detail=f"serving with fallbacks: {', '.join(missing)}")


@router.get("/version", response_model=VersionResponse)
def version(pipeline: AlprPipeline = Depends(pipeline_dependency)) -> VersionResponse:
    return VersionResponse(
        package_version=__version__,
        model_version=MODEL_VERSION,
        components=pipeline.component_versions,
    )
