#!/usr/bin/env python3
"""Generate OCR-ready crops with the arrangement used by Tanzanian plates.

The source photographs show the common *stacked* layout: ``T123`` over
``ABC`` (and ``MC123`` over ``ABC`` for motorcycles). Production rectification
de-stacks that physical plate into a one-line OCR crop, so this generator does
the same. It deliberately does not invent full vehicles or plate bounding
boxes: those must come from original, reviewed photographs.
"""

from __future__ import annotations

import argparse
import csv
import random
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
PLATE_W, PLATE_H = 260, 180
OCR_W, OCR_H = 256, 32
YELLOW, WHITE, BLACK = (238, 190, 28), (245, 245, 245), (18, 18, 18)
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def _private() -> str:
    return f"T{random.randint(0, 999):03d}" + "".join(random.choices(string.ascii_uppercase, k=3))


def _motorcycle() -> str:
    return f"MC{random.randint(100, 999):03d}" + "".join(random.choices(string.ascii_uppercase, k=3))


def _center(draw: ImageDraw.ImageDraw, text: str, font, y: int) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((PLATE_W - (box[2] - box[0])) / 2, y), text, font=font, fill=BLACK)


def _physical_plate(text: str, motorcycle: bool) -> np.ndarray:
    """Render the physical two-row plate before it is rectified for OCR."""
    image = Image.new("RGB", (PLATE_W, PLATE_H), YELLOW if random.random() < .94 else WHITE)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((3, 3, PLATE_W - 4, PLATE_H - 4), radius=8, outline=BLACK, width=3)
    top = text[:5] if motorcycle else text[:4]
    bottom = text[5:] if motorcycle else text[4:]
    _center(draw, top, _font(random.randint(52, 62) if motorcycle else random.randint(60, 70)), 14)
    _center(draw, bottom, _font(random.randint(62, 74)), 92)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _rectify_to_ocr(physical: np.ndarray) -> np.ndarray:
    """Mirror ``PlateRectifier._destack`` without clipping characters."""
    h, _ = physical.shape[:2]
    top = physical[int(.05 * h):int(.56 * h)]
    bottom = physical[int(.44 * h):int(.95 * h)]
    top = cv2.resize(top, (OCR_W // 2, OCR_H), interpolation=cv2.INTER_CUBIC)
    bottom = cv2.resize(bottom, (OCR_W // 2, OCR_H), interpolation=cv2.INTER_CUBIC)
    return np.hstack((top, bottom))


def _degrade(image: np.ndarray) -> np.ndarray:
    """Camera-like degradation; reflective borders preserve all plate text."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-2.5, 2.5), random.uniform(.96, 1.04))
    image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if random.random() < .35:
        image = cv2.GaussianBlur(image, random.choice([(3, 3), (5, 3)]), 0)
    if random.random() < .25:
        gradient = np.linspace(random.uniform(.55, .85), random.uniform(.9, 1.1), w, dtype=np.float32)
        image = np.clip(image * gradient[None, :, None], 0, 255).astype(np.uint8)
    image = np.clip(image.astype(np.float32) + np.random.normal(0, random.uniform(1, 7), image.shape), 0, 255)
    ok, encoded = cv2.imencode(".jpg", image.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, random.randint(65, 96)])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else image.astype(np.uint8)


def generate(count: int, out_dir: Path, moto_ratio: float, seed: int) -> Path:
    if not 0 <= moto_ratio <= 1:
        raise ValueError("--motorcycle-ratio must be between 0 and 1")
    random.seed(seed)
    np.random.seed(seed)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ocr_annotations.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "plate_text", "country", "plate_type", "split", "group_key", "source"])
        for index in range(count):
            motorcycle = random.random() < moto_ratio
            text = _motorcycle() if motorcycle else _private()
            crop = _degrade(_rectify_to_ocr(_physical_plate(text, motorcycle)))
            path = image_dir / f"{text}-{index:07d}.jpg"
            cv2.imwrite(str(path), crop)
            try:
                image_path = str(path.relative_to(REPO_ROOT))
            except ValueError:
                image_path = str(path.resolve())
            writer.writerow([image_path, text, "TZ", "MOTORCYCLE" if motorcycle else "PRIVATE", "train", text, "synthetic"])
    print(f"Wrote {count} layout-correct synthetic OCR crops -> {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=40000)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "datasets/synthetic/generated_plates_v2")
    parser.add_argument("--motorcycle-ratio", type=float, default=.062, help="matches the supplied real labels")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    generate(args.count, args.out, args.motorcycle_ratio, args.seed)


if __name__ == "__main__":
    main()
