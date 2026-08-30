"""Configuration loading.

Two layers:
  1. YAML files under configs/ describe model architecture, rules and pipeline wiring.
  2. Environment variables (prefix ``TZ_ALPR_``) hold deployment-specific values and
     secrets. Env always wins over YAML.

Nothing here reads credentials from source-controlled YAML (spec §21).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, resolving relative paths against the repo root."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def resolve_path(value: str | Path | None) -> Path | None:
    """Resolve a possibly-relative path from a config against the repo root."""
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


class Settings(BaseSettings):
    """Deployment settings sourced from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="TZ_ALPR_", env_file=".env", extra="ignore", case_sensitive=False
    )

    env: str = "development"
    log_level: str = "INFO"

    model_version: str = "tz-alpr-1.2.0"
    device: str = "cpu"
    runtime: str = "torch"

    ocr_weights: str | None = None
    ocr_onnx: str | None = None
    plate_detector_weights: str | None = None
    vehicle_detector_weights: str | None = None

    enable_super_resolution: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 8080
    max_upload_mb: int = 25

    inference_config: str = "configs/inference.yaml"

    webhook_secret: str = Field(default="change-me", repr=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=8)
def get_config(name: str) -> dict[str, Any]:
    """Load and cache a YAML config by file name or path (e.g. ``"inference.yaml"``)."""
    if os.sep in name or name.endswith((".yaml", ".yml")) and not (CONFIG_DIR / name).exists():
        return load_yaml(name)
    return load_yaml(CONFIG_DIR / name)
