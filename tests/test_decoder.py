"""Tests for PyAV frame decoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from delayed_photography.decoder import count_frames, iter_frames


def test_iter_frames_yields_rgb_uint8_shape(
    video_mean_three_grays: Path,
) -> None:
    frames = list(iter_frames(video_mean_three_grays))
    assert len(frames) == 3
    assert frames[0].dtype == np.uint8
    assert frames[0].shape == (16, 16, 3)


def test_count_frames_matches_iter(video_mean_three_grays: Path) -> None:
    assert count_frames(video_mean_three_grays) == 3


def test_iter_frames_raises_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.mp4"
    with pytest.raises(OSError):
        next(iter_frames(missing))
