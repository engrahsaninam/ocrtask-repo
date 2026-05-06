from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from section1.preprocessing import preprocess_image


def _save_text_image(path: Path, fill: int = 255, text_fill: int = 0, rotate: float = 0) -> None:
    img = Image.new("L", (160, 80), fill)
    draw = ImageDraw.Draw(img)
    draw.text((20, 28), "hello", fill=text_fill)
    if rotate:
        img = img.rotate(rotate, expand=True, fillcolor=fill)
    img.save(path)


def test_preprocess_returns_binary_uint8_array(tmp_path: Path) -> None:
    path = tmp_path / "clean.png"
    _save_text_image(path)

    result = preprocess_image(str(path))

    assert result.dtype == np.uint8
    assert result.ndim == 2
    assert set(np.unique(result)).issubset({0, 255})


def test_preprocess_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        preprocess_image("missing.png")


def test_preprocess_handles_skewed_image(tmp_path: Path) -> None:
    path = tmp_path / "skewed.png"
    _save_text_image(path, rotate=8)

    result = preprocess_image(str(path))

    assert result.size > 0
    assert 0 in np.unique(result)


def test_preprocess_enhances_low_contrast_image(tmp_path: Path) -> None:
    path = tmp_path / "low_contrast.png"
    _save_text_image(path, fill=180, text_fill=135)

    result = preprocess_image(str(path))

    assert set(np.unique(result)).issubset({0, 255})
    assert len(np.unique(result)) == 2


def test_preprocess_suppresses_blue_notebook_lines(tmp_path: Path) -> None:
    path = tmp_path / "ruled.png"
    img = Image.new("RGB", (220, 100), "white")
    draw = ImageDraw.Draw(img)
    draw.line((0, 50, 219, 50), fill=(120, 175, 220), width=2)
    draw.text((40, 42), "hello", fill=(20, 20, 20))
    img.save(path)

    result = preprocess_image(str(path))

    dark_pixels_on_rule = int((result[50, :] == 0).sum())
    assert dark_pixels_on_rule < 80
