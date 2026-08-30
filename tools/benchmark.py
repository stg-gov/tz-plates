#!/usr/bin/env python3
"""Pipeline benchmark (spec §29).

Reports FPS, mean / P50 / P95 latency and per-stage timings for the whole
pipeline, plus CPU / RAM (and GPU / VRAM when pynvml is available).

    python tools/benchmark.py --images labeled_images --limit 300
    python tools/benchmark.py --synthetic 200            # generate throwaway inputs
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from tz_alpr.pipeline import get_pipeline
from tz_alpr.utils.timing import Stopwatch


def _resources() -> dict:
    out = {}
    try:
        import psutil

        out["cpu_percent"] = psutil.cpu_percent(interval=0.2)
        out["ram_mb"] = round(psutil.Process().memory_info().rss / 1e6, 1)
    except ImportError:
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        out["gpu_percent"] = util.gpu
        out["vram_mb"] = round(mem.used / 1e6, 1)
    except Exception:  # noqa: BLE001
        pass
    return out


def _load_images(args) -> list[np.ndarray]:
    if args.synthetic:
        rng = np.random.default_rng(0)
        return [rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(args.synthetic)]
    paths = sorted(Path(args.images).glob("*.jpg"))[: args.limit or None]
    imgs = [cv2.imread(str(p)) for p in paths]
    return [im for im in imgs if im is not None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, default=REPO_ROOT / "labeled_images")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    pipeline = get_pipeline()
    images = _load_images(args)
    if not images:
        raise SystemExit("No input images found.")
    print(f"Benchmarking on {len(images)} images | plate_detector={pipeline._plate_detector.name} "
          f"| ocr={pipeline._ocr.backend}")

    for im in images[: args.warmup]:
        pipeline.read_image(im)

    latencies: list[float] = []
    stage_totals: dict[str, float] = {}
    t0 = time.perf_counter()
    for im in images:
        sw = Stopwatch()
        with sw("total"):
            pipeline.read_image(im)
        latencies.append(sw.timings["total"])
        for stage, ms in getattr(pipeline, "last_stage_timings", {}).items():
            stage_totals[stage] = stage_totals.get(stage, 0.0) + ms
    wall = time.perf_counter() - t0

    latencies.sort()
    p = lambda q: latencies[min(len(latencies) - 1, int(q * len(latencies)))]
    print("\n=== Latency (ms) ===")
    print(f"  mean : {statistics.mean(latencies):8.2f}")
    print(f"  P50  : {p(0.50):8.2f}")
    print(f"  P95  : {p(0.95):8.2f}")
    print(f"  P99  : {p(0.99):8.2f}")
    print(f"  max  : {latencies[-1]:8.2f}")
    print(f"\n=== Throughput ===\n  {len(images) / wall:6.2f} FPS  ({len(images)} imgs / {wall:.1f}s)")
    print("\n=== Per-stage mean (ms) ===")
    for stage, total in sorted(stage_totals.items(), key=lambda kv: -kv[1]):
        print(f"  {stage:16s}: {total / len(images):7.2f}")
    print("\n=== Resources ===")
    for k, v in _resources().items():
        print(f"  {k:12s}: {v}")


if __name__ == "__main__":
    main()
