#!/usr/bin/env python3
"""Pre-flight convergence check for the OCR model (run this BEFORE a full train).

Overfits the CRNN+CTC model to a few hundred prepared crops with augmentation
off. A correct model+data pipeline reaches ~100% train sequence-accuracy within a
few hundred optimizer steps. If it stalls (char-acc plateaus < 0.6, seq-acc ~0),
something upstream is wrong (crop quality, charset, target encoding, model init)
and a full run will waste GPU hours.

    python tools/overfit_check.py                       # 512 samples, ~1500 steps
    python tools/overfit_check.py --samples 256 --steps 800 --device cuda

Exit code 0 if it converges (train seq-acc >= --target), 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from tz_alpr.config import load_yaml
from tz_alpr.ocr.charset import load_charset
from tz_alpr.ocr.ctc_decode import greedy_decode
from tz_alpr.ocr.dataset import OcrDataset, ctc_collate, load_samples_from_csv
from tz_alpr.ocr.metrics import score_batch
from tz_alpr.ocr.model import build_crnn
from tz_alpr.ocr.transforms import build_eval_transform


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs/ocr.yaml")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--min-det-conf", type=float, default=0.6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--target", type=float, default=0.95, help="train seq-acc to call success")
    args = ap.parse_args()

    device = (
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device={device}")

    cfg = load_yaml(args.config)
    charset = load_charset()
    samples = load_samples_from_csv(
        cfg["data"]["ocr_annotations_csv"], split="train", min_det_conf=args.min_det_conf
    )[: args.samples]
    if len(samples) < 32:
        sys.exit("Not enough prepared crops — run tools/prepare_dataset.py first.")

    ds = OcrDataset(
        samples,
        charset,
        build_eval_transform(cfg),  # NO augmentation
        image_root=cfg["data"].get("image_root", "."),
        min_len=cfg["data"]["min_text_len"],
        max_len=cfg["data"]["max_text_len"],
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=ctc_collate, num_workers=2,
        drop_last=True,
    )

    model = build_crnn(cfg["model"], charset.size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = torch.nn.CTCLoss(blank=charset.blank_index, zero_infinity=True)

    step = 0
    t0 = time.time()
    best_seq = 0.0
    model.train()
    while step < args.steps:
        for batch in loader:
            images = batch["images"].to(device)
            log_probs = model(images)
            tlen = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)
            loss = crit(log_probs, batch["targets"], tlen, batch["target_lengths"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1

            if step % 100 == 0 or step == args.steps:
                model.eval()
                preds: list[str] = []
                gts: list[str] = []
                with torch.no_grad():
                    for b in loader:
                        lp = model(b["images"].to(device)).permute(1, 0, 2).float().cpu().numpy()
                        for i, g in enumerate(b["texts"]):
                            preds.append(greedy_decode(lp[i], charset).text)
                            gts.append(g)
                s = score_batch(preds, gts)
                best_seq = max(best_seq, s.sequence_accuracy)
                print(
                    f"step {step:4d}  loss {loss.item():6.3f}  "
                    f"train char_acc {s.character_accuracy:.3f}  seq_acc {s.sequence_accuracy:.3f}  "
                    f"({time.time() - t0:.0f}s)  e.g. {preds[0]!r} / {gts[0]!r}",
                    flush=True,
                )
                model.train()
            if step >= args.steps:
                break

    ok = best_seq >= args.target
    print(f"\n{'PASS' if ok else 'FAIL'}: best train seq-acc {best_seq:.3f} (target {args.target})")
    if not ok:
        print(
            "The model did not overfit a small clean set. Do NOT start a full run.\n"
            "Check: crop quality (view datasets/processed/plate_crops/*.jpg), two-line\n"
            "de-stacking, charset/target encoding, model init."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
