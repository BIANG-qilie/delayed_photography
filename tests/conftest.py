"""Synthetic MP4 fixtures for tests (PyAV, no external assets)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import av
import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


def write_rgb_mp4(frames: list[NDArray[np.uint8]], path: Path, *, fps: int = 10) -> None:
    """Write a small MPEG-4 file from RGB uint8 frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        msg = "frames must be non-empty"
        raise ValueError(msg)
    h, w, _ = frames[0].shape
    for f in frames[1:]:
        if f.shape != frames[0].shape:
            msg = "All frames must share shape"
            raise ValueError(msg)
    with av.open(path, mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        for img in frames:
            vf = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(vf):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


@pytest.fixture
def video_mean_three_grays(tmp_path: Path) -> Path:
    """Three solid frames (60, 120, 180) -> mean ~120."""
    h, w = 16, 16
    frames = [np.full((h, w, 3), v, dtype=np.uint8) for v in (60, 120, 180)]
    path = tmp_path / "mean_three.mp4"
    write_rgb_mp4(frames, path)
    return path


@pytest.fixture
def video_moving_white_dot(tmp_path: Path) -> Path:
    """Black frames with a single white pixel moving along the diagonal."""
    h, w = 16, 16
    frames: list[NDArray[np.uint8]] = []
    for t in range(5):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[t, t, :] = 255
        frames.append(img)
    path = tmp_path / "moving_dot.mp4"
    write_rgb_mp4(frames, path)
    return path


@pytest.fixture
def video_static_gray_with_red_flash(tmp_path: Path) -> Path:
    """Mostly gray; one frame has a red square in the center."""
    h, w = 32, 32
    gray = np.full((h, w, 3), 100, dtype=np.uint8)
    frames = [gray.copy() for _ in range(6)]
    red_frame = gray.copy()
    red_frame[12:20, 12:20, 0] = 255
    red_frame[12:20, 12:20, 1] = 0
    red_frame[12:20, 12:20, 2] = 0
    frames[2] = red_frame
    path = tmp_path / "median_red.mp4"
    write_rgb_mp4(frames, path)
    return path
