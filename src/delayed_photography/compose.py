"""Public API: compose a single stacked image from a video."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from delayed_photography.accumulators import (
    DEFAULT_MEDIAN_MAX_BYTES,
    MaxAccumulator,
    MeanAccumulator,
    MedianAccumulator,
)
from delayed_photography.decoder import count_frames, iter_frames
from delayed_photography.io import save_image
from delayed_photography.samplers import frame_indices, iter_selected_frames

if TYPE_CHECKING:
    from numpy.typing import NDArray

Mode = Literal["mean", "max", "median"]
Sampling = Literal["all", "step", "count"]


def _iter_frames_for_sampling(
    video_path: Path,
    *,
    sampling: Sampling,
    sampling_param: int,
) -> Iterator[NDArray[np.uint8]]:
    if sampling == "all":
        yield from iter_frames(video_path)
        return

    if sampling == "step":
        if sampling_param < 1:
            msg = f"sampling_param must be >= 1 for sampling='step', got {sampling_param}"
            raise ValueError(msg)
        for i, frame in enumerate(iter_frames(video_path)):
            if i % sampling_param == 0:
                yield frame
        return

    if sampling == "count":
        total = count_frames(video_path)
        indices = frame_indices(
            total_frames=total,
            sampling="count",
            sampling_param=sampling_param,
        )
        yield from iter_selected_frames(iter_frames(video_path), indices=indices)
        return

    msg = f"Unknown sampling: {sampling!r}"
    raise ValueError(msg)


def compose(
    video_path: str | Path,
    *,
    mode: Mode = "mean",
    sampling: Sampling = "all",
    sampling_param: int = 1,
    output_path: str | Path | None = None,
    max_median_bytes: int = DEFAULT_MEDIAN_MAX_BYTES,
) -> NDArray[np.uint8]:
    """Stack frames from ``video_path`` into one RGB image.

    Args:
        video_path: Input video file path.
        mode: ``mean`` (long-exposure look), ``max`` (light trails), ``median`` (remove tourists).
        sampling: ``all`` uses every frame; ``step`` every ``sampling_param``-th frame;
            ``count`` uses ``sampling_param`` frames spread uniformly across the clip.
        sampling_param: Interval for ``step`` or target count for ``count``; ignored for ``all``.
        output_path: If set, write JPEG/PNG based on file suffix.
        max_median_bytes: Hard cap on buffered bytes for ``median`` mode.

    Returns:
        ``uint8`` RGB array with shape ``(H, W, 3)``.

    Note:
        ``median`` buffers every selected frame in RAM; long clips can exceed memory.
        Prefer ``sampling='count'`` or ``'step'`` for long videos.
    """
    path = Path(video_path)
    if not path.is_file():
        msg = f"Video path does not exist or is not a file: {path}"
        raise FileNotFoundError(msg)

    if mode == "mean":
        acc: MeanAccumulator | MaxAccumulator | MedianAccumulator = MeanAccumulator()
    elif mode == "max":
        acc = MaxAccumulator()
    elif mode == "median":
        acc = MedianAccumulator(max_bytes=max_median_bytes)
    else:
        msg = f"Unknown mode: {mode!r}"
        raise ValueError(msg)

    frame_iter = _iter_frames_for_sampling(
        path,
        sampling=sampling,
        sampling_param=sampling_param,
    )
    for frame in frame_iter:
        acc.push(frame)

    out = acc.result()
    if output_path is not None:
        save_image(out, output_path)
    return out
