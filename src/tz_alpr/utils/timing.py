"""Lightweight timing helpers used by the pipeline and the benchmark tool."""

from __future__ import annotations

import time
from contextlib import ContextDecorator


def now_ms() -> float:
    return time.perf_counter() * 1000.0


class Stopwatch(ContextDecorator):
    """Accumulates named stage timings.

    Usage::

        sw = Stopwatch()
        with sw("detect"):
            ...
        sw.timings["detect"]  # milliseconds
    """

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._stack: list[tuple[str, float]] = []

    def __call__(self, name: str) -> Stopwatch:
        self._pending = name
        return self

    def __enter__(self) -> Stopwatch:
        self._stack.append((self._pending, time.perf_counter()))
        return self

    def __exit__(self, *exc) -> bool:
        name, start = self._stack.pop()
        self.timings[name] = self.timings.get(name, 0.0) + (time.perf_counter() - start) * 1000.0
        return False

    @property
    def total_ms(self) -> float:
        return sum(self.timings.values())
