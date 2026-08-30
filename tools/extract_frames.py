#!/usr/bin/env python3
"""Sample frames from a video for annotation / dataset building (spec §10, §13).

Records a ``session_id`` per video so downstream splitting can keep frames from
one video out of multiple splits (leakage prevention).

    python tools/extract_frames.py --video clip.mp4 --fps 2 --out datasets/raw/images/clip
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2


def extract(video: Path, out_dir: Path, target_fps: float, max_frames: int) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / max(target_fps, 0.01))))
    session_id = hashlib.md5(str(video.resolve()).encode()).hexdigest()[:10]
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = out_dir / "frames.csv"
    idx = saved = 0
    with manifest.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_path", "session_id", "source_video", "src_frame_index", "timestamp_s"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                name = f"{session_id}_{saved:06d}.jpg"
                cv2.imwrite(str(out_dir / name), frame)
                w.writerow([str(out_dir / name), session_id, video.name, idx, round(idx / src_fps, 3)])
                saved += 1
                if max_frames and saved >= max_frames:
                    break
            idx += 1
    cap.release()
    print(f"Saved {saved} frames -> {out_dir}  (session_id={session_id})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    extract(args.video, args.out, args.fps, args.max_frames)


if __name__ == "__main__":
    main()
