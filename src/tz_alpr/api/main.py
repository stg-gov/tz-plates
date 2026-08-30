"""FastAPI application factory + entrypoint (spec §19)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tz_alpr.api import routes_health, routes_plate
from tz_alpr.config import get_settings
from tz_alpr.logging_conf import configure_logging, get_logger
from tz_alpr.version import __version__

log = get_logger("tz_alpr.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=(settings.env == "production"))
    # Warm the pipeline so the first request is not slow / does not race.
    from tz_alpr.api.deps import pipeline_dependency

    pipeline_dependency()
    log.info("api.startup", version=__version__, env=settings.env)
    yield
    log.info("api.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="tz-alpr",
        version=__version__,
        description="Open-source Automatic License Plate Recognition — Tanzania.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(routes_health.router)
    app.include_router(routes_plate.router)
    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "tz_alpr.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.env == "development",
    )


if __name__ == "__main__":
    run()
