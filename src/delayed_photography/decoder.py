"""Decode video frames to RGB uint8 arrays using PyAV."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import av
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def count_frames(video_path: str | Path) -> int:
    """Count decoded video frames (reliable, at cost of one full decode pass)."""
    path = Path(video_path)
    n = 0
    for _ in iter_frames(path):
        n += 1
    return n


def iter_frames(video_path: str | Path) -> Iterator[NDArray[np.uint8]]:
    """Yield each video frame as ``uint8`` RGB with shape ``(H, W, 3)``.

    Raises:
        ValueError: If frame dimensions change mid-stream.
    """
    path = Path(video_path)
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        expected_shape: tuple[int, int, int] | None = None
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format="rgb24")
            if expected_shape is None:
                expected_shape = (arr.shape[0], arr.shape[1], arr.shape[2])
            elif arr.shape != expected_shape:
                msg = (
                    f"Inconsistent frame shape: expected {expected_shape}, got {arr.shape}. "
                    "All frames must share the same resolution."
                )
                raise ValueError(msg)
            yield arr.astype(np.uint8, copy=False)
