#!/usr/bin/env python3
"""Export trained models to ONNX for the runtime (spec §26).

    python tools/export_onnx.py ocr    --ckpt checkpoints/ocr-best.ckpt --out models/ocr/v1/ocr_crnn.onnx
    python tools/export_onnx.py plate  --weights runs/plate_detector/v1/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))


def export_ocr(ckpt: Path, out: Path, opset: int) -> None:
    import numpy as np
    import torch

    from tz_alpr.config import load_yaml
    from tz_alpr.ocr.charset import load_charset
    from tz_alpr.ocr.model import build_crnn

    cfg = load_yaml("configs/ocr.yaml")
    charset = load_charset()
    model = build_crnn(cfg["model"], num_classes=charset.size)

    state = torch.load(str(ckpt), map_location="cpu")
    state = state.get("state_dict", state)
    cleaned = {k.removeprefix("model."): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    h, w = cfg["model"]["input_height"], cfg["model"]["input_width"]
    c = cfg["model"].get("input_channels", 1)
    dummy = torch.randn(1, c, h, w)

    class _InferWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.infer_logits(x)  # (B, T, C) logits

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _InferWrapper(model),
        dummy,
        str(out),
        input_names=["image"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes={"image": {0: "batch", 3: "width"}, "logits": {0: "batch", 1: "time"}},
    )
    print(f"Wrote {out}")

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        ref = model.infer_logits(dummy).detach().numpy()
        got = sess.run(None, {"image": dummy.numpy()})[0]
        max_err = float(np.abs(ref - got).max())
        print(f"onnxruntime parity check: max abs err = {max_err:.3e}")
    except ImportError:
        print("onnxruntime not installed; skipped parity check")


def export_plate(weights: Path, imgsz: int) -> None:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    path = model.export(format="onnx", imgsz=imgsz, opset=12, dynamic=True, simplify=True)
    print(f"Wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="target", required=True)

    p_ocr = sub.add_parser("ocr")
    p_ocr.add_argument("--ckpt", type=Path, required=True)
    p_ocr.add_argument("--out", type=Path, default=REPO_ROOT / "models/ocr/v1/ocr_crnn.onnx")
    p_ocr.add_argument("--opset", type=int, default=17)

    p_plate = sub.add_parser("plate")
    p_plate.add_argument("--weights", type=Path, required=True)
    p_plate.add_argument("--imgsz", type=int, default=960)

    args = ap.parse_args()
    if args.target == "ocr":
        export_ocr(args.ckpt, args.out, args.opset)
    else:
        export_plate(args.weights, args.imgsz)


if __name__ == "__main__":
    main()
