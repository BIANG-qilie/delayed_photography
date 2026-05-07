"""Pixel-wise frame stacking strategies."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

DEFAULT_MEDIAN_MAX_BYTES = 512 * 1024 * 1024


class MeanAccumulator:
    """Online mean in float64 for numerical stability."""

    def __init__(self) -> None:
        self._mean: NDArray[np.float64] | None = None
        self._n = 0

    def push(self, frame: NDArray[np.uint8]) -> None:
        x = frame.astype(np.float64, copy=False)
        self._n += 1
        if self._mean is None:
            self._mean = x.copy()
        else:
            self._mean += (x - self._mean) / self._n

    def result(self) -> NDArray[np.uint8]:
        if self._mean is None or self._n == 0:
            msg = "MeanAccumulator has no frames"
            raise ValueError(msg)
        return np.clip(np.round(self._mean), 0, 255).astype(np.uint8)


class MaxAccumulator:
    """Per-channel maximum across frames."""

    def __init__(self) -> None:
        self._max: NDArray[np.uint8] | None = None

    def push(self, frame: NDArray[np.uint8]) -> None:
        if self._max is None:
            self._max = frame.copy()
        else:
            np.maximum(self._max, frame, out=self._max)

    def result(self) -> NDArray[np.uint8]:
        if self._max is None:
            msg = "MaxAccumulator has no frames"
            raise ValueError(msg)
        return self._max


class MedianAccumulator:
    """Full-frame buffer then ``np.median`` (high memory)."""

    def __init__(self, *, max_bytes: int = DEFAULT_MEDIAN_MAX_BYTES) -> None:
        if max_bytes < 1:
            msg = f"max_bytes must be >= 1, got {max_bytes}"
            raise ValueError(msg)
        self._max_bytes = max_bytes
        self._frames: list[NDArray[np.uint8]] = []

    def push(self, frame: NDArray[np.uint8]) -> None:
        next_n = len(self._frames) + 1
        estimated = next_n * int(frame.nbytes)
        if estimated > self._max_bytes:
            msg = (
                "Estimated memory for median stacking exceeds max_bytes="
                f"{self._max_bytes}. Use sampling='step' or sampling='count' with a "
                "smaller frame set, or increase max_median_bytes."
            )
            raise MemoryError(msg)
        self._frames.append(frame.copy())

    def result(self) -> NDArray[np.uint8]:
        if not self._frames:
            msg = "MedianAccumulator has no frames"
            raise ValueError(msg)
        stacked = np.stack(self._frames, axis=0)
        med = np.median(stacked, axis=0)
        return cast(NDArray[np.uint8], med.astype(np.uint8, copy=False))
