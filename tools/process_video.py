#!/usr/bin/env python3
"""Run the video ALPR pipeline on a file and print deduplicated events (spec §17, §18).

    python tools/process_video.py clip.mp4 --camera-id DODOMA_PARKING_01 --sample-fps 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tz_alpr.pipeline.video import VideoPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--camera-id", default="upload")
    ap.add_argument("--sample-fps", type=float, default=5.0)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    args = ap.parse_args()

    resp = VideoPipeline().process(
        args.video,
        camera_id=args.camera_id,
        sample_fps=args.sample_fps,
        max_seconds=args.max_seconds,
    )
    print(json.dumps(resp.model_dump(), indent=2))


if __name__ == "__main__":
    main()
