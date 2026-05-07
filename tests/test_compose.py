"""End-to-end compose() tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from delayed_photography.compose import compose
from delayed_photography.decoder import iter_frames


def test_compose_mean_all_close_to_expected(video_mean_three_grays: Path) -> None:
    out = compose(video_mean_three_grays, mode="mean", sampling="all")
    assert out.shape == (16, 16, 3)
    # Lossy MPEG-4 may shift values slightly
    assert np.all(np.abs(out.astype(np.int16) - 120) <= 8)


def test_compose_max_step_preserves_trail(video_moving_white_dot: Path) -> None:
    out = compose(video_moving_white_dot, mode="max", sampling="step", sampling_param=1)
    frames = list(iter_frames(video_moving_white_dot))
    expected = np.maximum.reduce(frames)
    assert np.array_equal(out, expected)


def test_compose_median_count_removes_red_blob(
    video_static_gray_with_red_flash: Path,
) -> None:
    out = compose(
        video_static_gray_with_red_flash,
        mode="median",
        sampling="count",
        sampling_param=6,
    )
    # Center block should return to gray median (~100), not red
    block = out[12:20, 12:20, 0]
    assert int(block.mean()) < 200


def test_compose_writes_output_png(
    video_mean_three_grays: Path,
    tmp_path: Path,
) -> None:
    outp = tmp_path / "stack.png"
    compose(video_mean_three_grays, mode="mean", sampling="all", output_path=outp)
    assert outp.is_file()


def test_compose_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compose(tmp_path / "missing.mp4")


def test_compose_mean_step_subsamples(video_mean_three_grays: Path) -> None:
    out = compose(video_mean_three_grays, mode="mean", sampling="step", sampling_param=2)
    # Frames 0 and 2 only -> mean of 60 and 180 is 120
    assert np.all(np.abs(out.astype(np.int16) - 120) <= 8)


def test_compose_max_count_matches_manual(video_moving_white_dot: Path) -> None:
    out = compose(video_moving_white_dot, mode="max", sampling="count", sampling_param=3)
    total = len(list(iter_frames(video_moving_white_dot)))
    idx = [0, total // 2, total - 1] if total >= 3 else list(range(total))
    frames = list(iter_frames(video_moving_white_dot))
    subset = [frames[i] for i in idx]
    expected = np.maximum.reduce(subset)
    assert np.array_equal(out, expected)


def test_compose_median_all_small_clip(video_mean_three_grays: Path) -> None:
    out = compose(video_mean_three_grays, mode="median", sampling="all")
    # Solid-ish frames -> median near per-pixel medians of the three grays
    assert out.shape == (16, 16, 3)
    assert np.all(np.abs(out.astype(np.int16) - 120) <= 8)
