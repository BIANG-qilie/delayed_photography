"""Sampling index utilities."""

from __future__ import annotations

import numpy as np
import pytest

from delayed_photography.samplers import frame_indices, iter_selected_frames


@pytest.mark.parametrize(
    ("total", "sampling", "param", "expected"),
    [
        (5, "all", 1, [0, 1, 2, 3, 4]),
        (10, "step", 3, [0, 3, 6, 9]),
        (10, "count", 3, [0, 4, 9]),
    ],
)
def test_frame_indices(
    total: int,
    sampling: str,
    param: int,
    expected: list[int],
) -> None:
    assert frame_indices(total_frames=total, sampling=sampling, sampling_param=param) == expected  # type: ignore[arg-type]


def test_frame_indices_count_clamps_to_total() -> None:
    assert frame_indices(total_frames=3, sampling="count", sampling_param=100) == [0, 1, 2]


def test_frame_indices_invalid_step() -> None:
    with pytest.raises(ValueError):
        frame_indices(total_frames=5, sampling="step", sampling_param=0)


def test_iter_selected_frames_subset() -> None:
    frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(6)]
    out = list(iter_selected_frames(iter(frames), indices=[0, 2, 5]))
    assert len(out) == 3
    assert int(out[0][0, 0, 0]) == 0
    assert int(out[1][0, 0, 0]) == 2
    assert int(out[2][0, 0, 0]) == 5
