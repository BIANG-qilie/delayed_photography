"""Accumulator unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from delayed_photography.accumulators import MaxAccumulator, MeanAccumulator, MedianAccumulator


def test_mean_accumulator_three_values() -> None:
    acc = MeanAccumulator()
    acc.push(np.full((2, 2, 3), 60, dtype=np.uint8))
    acc.push(np.full((2, 2, 3), 120, dtype=np.uint8))
    acc.push(np.full((2, 2, 3), 180, dtype=np.uint8))
    out = acc.result()
    assert out.shape == (2, 2, 3)
    assert np.all(out == 120)


def test_max_accumulator_takes_elementwise_max() -> None:
    acc = MaxAccumulator()
    a = np.zeros((3, 3, 3), dtype=np.uint8)
    a[0, 0, :] = 10
    b = np.zeros((3, 3, 3), dtype=np.uint8)
    b[1, 1, :] = 200
    acc.push(a)
    acc.push(b)
    out = acc.result()
    assert int(out[0, 0, 0]) == 10
    assert int(out[1, 1, 0]) == 200


def test_median_accumulator_ignores_single_outlier_channel() -> None:
    acc = MedianAccumulator(max_bytes=10 * 1024 * 1024)
    base = np.full((4, 4, 3), 100, dtype=np.uint8)
    for _ in range(4):
        acc.push(base.copy())
    spike = base.copy()
    spike[1, 1, :] = 255
    acc.push(spike)
    out = acc.result()
    assert int(out[1, 1, 0]) == 100


def test_median_accumulator_memory_guard() -> None:
    # Each frame is 10*10*3 = 300 bytes; allow 3 buffered frames but not 4.
    acc = MedianAccumulator(max_bytes=1000)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    for _ in range(3):
        acc.push(frame)
    with pytest.raises(MemoryError):
        acc.push(frame)
