"""Image IO tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from delayed_photography.io import save_image


def test_save_png_roundtrip(tmp_path: Path) -> None:
    arr = np.zeros((4, 5, 3), dtype=np.uint8)
    arr[:, :, 0] = 255
    path = tmp_path / "out.png"
    save_image(arr, path)
    loaded = np.array(Image.open(path).convert("RGB"))
    assert loaded.shape == arr.shape
    assert np.array_equal(loaded, arr)


def test_save_jpeg(tmp_path: Path) -> None:
    arr = np.full((8, 8, 3), 128, dtype=np.uint8)
    path = tmp_path / "out.jpg"
    save_image(arr, path)
    assert path.is_file()


def test_save_unsupported_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_image(np.zeros((2, 2, 3), dtype=np.uint8), tmp_path / "x.bmp")
