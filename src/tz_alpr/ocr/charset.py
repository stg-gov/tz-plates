"""Charset shared by the OCR head and the CTC decoder.

Index 0 is reserved for the CTC blank. Characters come from the active country's
rule config so the OCR alphabet and the validation alphabet can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

from tz_alpr.config import load_yaml

BLANK = "<blank>"


@dataclass(frozen=True)
class Charset:
    chars: str                       # ordered, index 1..N (index 0 is blank)

    @property
    def size(self) -> int:
        return len(self.chars) + 1   # + blank

    @property
    def blank_index(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        return [self.chars.index(c) + 1 for c in text]

    def decode_indices(self, indices: list[int]) -> str:
        out = []
        for i in indices:
            if i == 0:
                continue
            out.append(self.chars[i - 1])
        return "".join(out)

    def char_at(self, index: int) -> str:
        return BLANK if index == 0 else self.chars[index - 1]


def load_charset(tanzania_config: str = "configs/tanzania.yaml") -> Charset:
    cfg = load_yaml(tanzania_config)
    raw = cfg["charset"]
    seen: list[str] = []
    for c in raw.upper():
        if c not in seen:
            seen.append(c)
    return Charset(chars="".join(seen))
