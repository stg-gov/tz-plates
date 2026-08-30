"""FastAPI dependencies."""

from __future__ import annotations

from functools import lru_cache

from tz_alpr.config import Settings, get_settings
from tz_alpr.pipeline import AlprPipeline, get_pipeline


@lru_cache(maxsize=1)
def pipeline_dependency() -> AlprPipeline:
    return get_pipeline()


def settings_dependency() -> Settings:
    return get_settings()
