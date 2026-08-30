"""Runtime OCR engines.

Three backends, chosen by ``build_ocr_engine``:
  * ``TorchOcrEngine``  — loads a .pt checkpoint (Lightning or bare state_dict).
  * ``OnnxOcrEngine``   — loads a .onnx export, needs only onnxruntime + numpy.
  * ``NullOcrEngine``   — no weights available; returns an empty reading and the
    pipeline attaches a warning. Keeps ``/v1/plate-reader`` responding on a clean
    checkout so the wiring is testable before training.

All three return the same :class:`OcrPrediction`, including per-character
posteriors, which the Tanzania-aware post-processor consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tz_alpr.config import get_config, get_settings, resolve_path
from tz_alpr.logging_conf import get_logger
from tz_alpr.ocr.charset import Charset, load_charset
from tz_alpr.ocr.ctc_decode import CharPosterior, DecodedSequence, greedy_decode
from tz_alpr.ocr.preprocess import preprocess_plate

log = get_logger(__name__)


@dataclass
class OcrPrediction:
    raw_text: str
    seq_confidence: float
    positions: list[CharPosterior] = field(default_factory=list)
    backend: str = "null"
    available: bool = True

    @classmethod
    def from_decoded(cls, dec: DecodedSequence, backend: str) -> OcrPrediction:
        return cls(
            raw_text=dec.text,
            seq_confidence=dec.seq_confidence,
            positions=dec.positions,
            backend=backend,
            available=True,
        )


class _BaseOcrEngine:
    backend = "base"
    available = True

    def __init__(self, charset: Charset, model_cfg: dict) -> None:
        self.charset = charset
        self.input_height = int(model_cfg.get("input_height", 48))
        self.input_width = int(model_cfg.get("input_width", 192))
        self.gray = int(model_cfg.get("input_channels", 1)) == 1

    def _decode(self, logits_1tc: np.ndarray) -> OcrPrediction:
        logits = np.asarray(logits_1tc)
        if logits.ndim == 3:
            logits = logits[0]
        dec = greedy_decode(logits, self.charset, topk=4)
        return OcrPrediction.from_decoded(dec, self.backend)

    def predict(self, plate_bgr: np.ndarray) -> OcrPrediction:  # pragma: no cover - abstract
        raise NotImplementedError


class TorchOcrEngine(_BaseOcrEngine):
    backend = "torch"

    def __init__(self, weights: Path, charset: Charset, model_cfg: dict, device: str = "cpu") -> None:
        super().__init__(charset, model_cfg)
        import torch

        from tz_alpr.ocr.model import build_crnn

        self._torch = torch
        self._device = torch.device(device if torch.cuda.is_available() or "cpu" in device else "cpu")
        self._model = build_crnn(model_cfg, num_classes=charset.size)
        state = torch.load(str(weights), map_location="cpu")
        state = state.get("state_dict", state)
        cleaned = { k.removeprefix("model."): v for k, v in state.items() }
        missing, unexpected = self._model.load_state_dict(cleaned, strict=False)
        if missing:
            log.warning("ocr.torch.load_partial", missing=len(missing), unexpected=len(unexpected))
        self._model.eval().to(self._device)

    def predict(self, plate_bgr: np.ndarray) -> OcrPrediction:
        arr = preprocess_plate(plate_bgr, self.input_height, self.input_width, self.gray)
        with self._torch.no_grad():
            x = self._torch.from_numpy(arr).to(self._device)
            logits = self._model.infer_logits(x).float().cpu().numpy()  # (1, T, C)
        return self._decode(logits)


class OnnxOcrEngine(_BaseOcrEngine):
    backend = "onnx"

    def __init__(self, onnx_path: Path, charset: Charset, model_cfg: dict, device: str = "cpu") -> None:
        super().__init__(charset, model_cfg)
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if "cuda" in device:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess = ort.InferenceSession(str(onnx_path), providers=providers)
        self._input_name = self._sess.get_inputs()[0].name

    def predict(self, plate_bgr: np.ndarray) -> OcrPrediction:
        arr = preprocess_plate(plate_bgr, self.input_height, self.input_width, self.gray).astype(
            np.float32
        )
        logits = self._sess.run(None, {self._input_name: arr})[0]  # (1, T, C)
        return self._decode(logits)


class NullOcrEngine(_BaseOcrEngine):
    backend = "null"
    available = False

    def predict(self, plate_bgr: np.ndarray) -> OcrPrediction:
        return OcrPrediction(raw_text="", seq_confidence=0.0, backend="null", available=False)


def build_ocr_engine(inference_config: str = "configs/inference.yaml") -> _BaseOcrEngine:
    inf = get_config(inference_config)
    settings = get_settings()
    ocr_cfg = get_config(inf["ocr"]["config"])
    model_cfg = ocr_cfg["model"]
    charset = load_charset(inf.get("country_rules_config", "configs/tanzania.yaml"))
    device = settings.device or inf.get("device", "cpu")

    onnx_path = resolve_path(settings.ocr_onnx)
    weights_path = resolve_path(settings.ocr_weights)
    prefer_onnx = (settings.runtime or inf.get("runtime", "torch")) == "onnx"

    if prefer_onnx and onnx_path and onnx_path.exists():
        try:
            log.info("ocr.engine", backend="onnx", path=str(onnx_path))
            return OnnxOcrEngine(onnx_path, charset, model_cfg, device)
        except Exception as exc:  # noqa: BLE001
            log.warning("ocr.onnx_failed", error=str(exc))

    if weights_path and weights_path.exists():
        try:
            log.info("ocr.engine", backend="torch", path=str(weights_path))
            return TorchOcrEngine(weights_path, charset, model_cfg, device)
        except Exception as exc:  # noqa: BLE001
            log.warning("ocr.torch_failed", error=str(exc))

    if onnx_path and onnx_path.exists():
        try:
            log.info("ocr.engine", backend="onnx", path=str(onnx_path))
            return OnnxOcrEngine(onnx_path, charset, model_cfg, device)
        except Exception as exc:  # noqa: BLE001
            log.warning("ocr.onnx_failed", error=str(exc))

    log.warning("ocr.engine", backend="null", reason="no OCR weights configured or loadable")
    return NullOcrEngine(charset, model_cfg)
