"""OCR subpackage.

Only numpy-light modules are imported here. Training modules (model,
lightning_module, datamodule, transforms) and the torch/onnx engines are
imported lazily by their callers so the inference API needs neither torch nor
albumentations to start.
"""

from tz_alpr.ocr.charset import Charset, load_charset
from tz_alpr.ocr.ctc_decode import CharPosterior, DecodedSequence, greedy_decode
from tz_alpr.ocr.metrics import OcrScores, score_batch

__all__ = [
    "CharPosterior",
    "Charset",
    "DecodedSequence",
    "OcrScores",
    "greedy_decode",
    "load_charset",
    "score_batch",
]
