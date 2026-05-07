"""Save RGB uint8 images."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray


def save_image(array: NDArray[np.uint8], path: str | Path) -> None:
    """Save ``array`` (H, W, 3) RGB ``uint8`` to ``path`` as JPEG or PNG."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        msg = f"Unsupported image extension {suffix!r}; use .png, .jpg, or .jpeg"
        raise ValueError(msg)
    image = Image.fromarray(array, mode="RGB")
    image.save(p)
