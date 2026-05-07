"""Frame index selection for sampling strategies."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

Sampling = Literal["all", "step", "count"]


def frame_indices(*, total_frames: int, sampling: Sampling, sampling_param: int) -> list[int]:
    """Return sorted unique frame indices to use for stacking.

    Args:
        total_frames: Total frames in the video.
        sampling: ``all`` uses every frame; ``step`` takes every Nth frame starting at 0;
            ``count`` takes ``sampling_param`` frames spread uniformly (inclusive ends).
        sampling_param: For ``step``, interval (>=1). For ``count``, target number of frames
            (>=1). Ignored for ``all``.

    Raises:
        ValueError: If arguments are inconsistent (e.g. non-positive counts).
    """
    if total_frames < 0:
        msg = f"total_frames must be non-negative, got {total_frames}"
        raise ValueError(msg)
    if total_frames == 0:
        return []

    if sampling == "all":
        return list(range(total_frames))

    if sampling == "step":
        if sampling_param < 1:
            msg = f"step must be >= 1 for sampling='step', got {sampling_param}"
            raise ValueError(msg)
        return list(range(0, total_frames, sampling_param))

    if sampling == "count":
        if sampling_param < 1:
            msg = f"count must be >= 1 for sampling='count', got {sampling_param}"
            raise ValueError(msg)
        k = min(sampling_param, total_frames)
        idx = np.linspace(0, total_frames - 1, num=k, dtype=np.int64)
        return sorted({int(i) for i in idx})

    msg = f"Unknown sampling mode: {sampling!r}"
    raise ValueError(msg)


def iter_selected_frames(
    frames: Iterator[NDArray[np.uint8]],
    *,
    indices: list[int],
) -> Iterator[NDArray[np.uint8]]:
    """Yield frames whose zero-based index appears in ``indices`` (must be sorted)."""
    if not indices:
        return
    want = set(indices)
    for i, frame in enumerate(frames):
        if i in want:
            yield frame
            want.discard(i)
            if not want:
                return
