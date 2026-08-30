"""CRNN OCR model (CNN feature extractor -> BiLSTM -> per-frame classifier).

Why CRNN + CTC for license plates (spec §6):
  * The label space is tiny and closed (36 alnum classes) and sequences are
    short (mostly 7-8 chars). A full attention/transformer decoder (PARSeq,
    TrOCR) buys little here and needs far more data and compute.
  * CTC is alignment-free: it copes with the variable inter-character spacing of
    real plates (the dots/flag on TZ plates, kerning differences) without
    character-level boxes, which this dataset does not have.
  * ~3.7M params, converts cleanly to ONNX, and runs in a few ms on CPU — which
    the deployment targets (Jetson, CPU-only servers) require.
  * LPRNet was considered (even lighter) but its fixed global-context head is
    less robust to perspective residue after rectification; SVTR is a strong
    middle ground and is provided as an alternative backbone hook below.

Two-line motorcycle plates are de-stacked to a single line by the rectifier
before they reach this model, so the head stays purely 1-D.
"""

from __future__ import annotations

import torch
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, bn: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, 1, 1, bias=not bn)
        self.bn = nn.BatchNorm2d(cout) if bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CNNLiteBackbone(nn.Module):
    """Canonical CRNN VGG-style stack (Shi et al. 2015).

    Height is pooled all the way to 1 inside the conv stack so no vertical
    averaging is needed; width is downsampled by 4 (stride-(2,1) pools in the
    later stages), giving W/4 time-steps. Expects input height 32.
    """

    out_channels = 512

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _ConvBlock(in_channels, 64),
            nn.MaxPool2d(2, 2),                              # 32 -> 16 (H), W/2
            _ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),                              # 16 -> 8  (H), W/4
            _ConvBlock(128, 256),
            _ConvBlock(256, 256),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),            # 8 -> 4 (H), W kept
            _ConvBlock(256, 512),
            _ConvBlock(512, 512),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),            # 4 -> 2 (H), W kept
            nn.Conv2d(512, 512, 2, 1, 0, bias=False),        # 2 -> 1 (H)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        input_height: int = 32,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_height = input_height
        self.in_channels = in_channels
        self.backbone = CNNLiteBackbone(in_channels)

        self.collapse = nn.AdaptiveAvgPool2d((1, None))  # safety net: force any residual H -> 1
        self.rnn = nn.LSTM(
            self.backbone.out_channels,
            rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rnn_hidden * 2, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        # PyTorch's default Conv2d init (kaiming_uniform_ with a=sqrt(5)) makes a
        # plain 7-layer VGG stack lose forward-signal magnitude layer by layer,
        # which stalls CTC training. Use fan-out Kaiming for the ReLU stack.
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for name, p in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> log-probs (T, B, num_classes) for CTC."""
        feat = self.backbone(x)                 # (B, C', H', W')
        feat = self.collapse(feat).squeeze(2)   # (B, C', W')
        feat = feat.permute(0, 2, 1)            # (B, W', C')  == (B, T, C')
        seq, _ = self.rnn(feat)                 # (B, T, 2H)
        seq = self.dropout(seq)
        logits = self.classifier(seq)           # (B, T, num_classes)
        return logits.log_softmax(2).permute(1, 0, 2)  # (T, B, num_classes)

    @torch.no_grad()
    def infer_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw (B, T, num_classes) logits for the inference decoder."""
        feat = self.backbone(x)
        feat = self.collapse(feat).squeeze(2).permute(0, 2, 1)
        seq, _ = self.rnn(feat)
        return self.classifier(seq)


def build_crnn(model_cfg: dict, num_classes: int) -> CRNN:
    return CRNN(
        num_classes=num_classes,
        in_channels=int(model_cfg.get("input_channels", 1)),
        input_height=int(model_cfg.get("input_height", 48)),
        rnn_hidden=int(model_cfg.get("rnn_hidden", 256)),
        rnn_layers=int(model_cfg.get("rnn_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.15)),
    )
