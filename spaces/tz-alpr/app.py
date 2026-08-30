"""Tanzanian ALPR demo Space."""

from __future__ import annotations

import json

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from infer import recognize

CSS = """
.gradio-container {max-width: 1100px !important;}
#plate-hero textarea {
  font-size: 2.4rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.12em !important;
  text-align: center !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
}
#status-ok {color: #1a7f37;}
footer {visibility: hidden;}
"""


def _to_bgr(im: Image.Image | np.ndarray | None) -> np.ndarray | None:
    if im is None:
        return None
    if isinstance(im, Image.Image):
        rgb = np.array(im.convert("RGB"))
    else:
        rgb = np.asarray(im)
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def run(im):
    bgr = _to_bgr(im)
    if bgr is None:
        raise gr.Error("Upload a photo of a vehicle rear or a plate crop.")
    out = recognize(bgr)
    vis = cv2.cvtColor(out["annotated_bgr"], cv2.COLOR_BGR2RGB)
    plates = out["results"]
    top = plates[0] if plates else {}
    plate = top.get("plate") or "—"
    conf = float(top.get("confidence") or 0)
    review = top.get("review_status") or "—"
    ptype = top.get("plate_type") or "—"
    badge = {
        "auto_accept": "✅ auto accept",
        "review": "👀 review",
        "manual": "✋ manual",
    }.get(review, review)
    summary = (
        f"**Type:** {ptype}  **Confidence:** {conf:.0%}  **Route:** {badge}\n\n"
        f"Raw OCR: `{top.get('raw_ocr', '')}`"
    )
    return vis, plate, summary, json.dumps(plates, indent=2)


with gr.Blocks(
    title="tz-alpr — Tanzanian plates",
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="amber", font=gr.themes.GoogleFont("DM Sans")),
    css=CSS,
) as demo:
    gr.Markdown(
        """
# 🇹🇿 tz-alpr
Read **Tanzanian** number plates from a phone photo — private `T 331 EBG` and
motorcycle / bajaji `MC 102 BXK`.

Trained CRNN+CTC · YOLO plate finder · Tanzania rule engine
        """
    )
    with gr.Row():
        with gr.Column(scale=5):
            inp = gr.Image(type="pil", label="Vehicle rear or plate crop", height=420)
            btn = gr.Button("Read plate", variant="primary", size="lg")
        with gr.Column(scale=5):
            out_im = gr.Image(label="Detection", height=320)
            plate = gr.Textbox(label="Normalized plate", elem_id="plate-hero", lines=1)
            meta = gr.Markdown()
    with gr.Accordion("JSON result", open=False):
        raw = gr.Code(language="json")
    gr.Examples(
        examples=[
            "examples/T532QJN-0039557.jpg",
            "examples/T535JHO-0028902.jpg",
            "examples/MC102ITJ-0049394.jpg",
        ],
        inputs=inp,
        label="Try an example (synthetic layout-correct plates)",
    )
    btn.click(run, inputs=inp, outputs=[out_im, plate, meta, raw])
    inp.change(run, inputs=inp, outputs=[out_im, plate, meta, raw])
    gr.Markdown(
        "Not legal advice. Readings under 90% confidence should be reviewed. "
        "Source: [stg-gov/tz-plates](https://github.com/stg-gov/tz-plates)."
    )

if __name__ == "__main__":
    demo.queue().launch()
