import importlib

import pytest
from PIL import Image, ImageDraw


def test_detect_border_success():
    mod = importlib.import_module("think.detect_border")
    img = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 199, 199], outline=(0, 0, 255), width=3)
    y_min, x_min, y_max, x_max = mod.detect_border(img, (0, 0, 255))
    assert (y_min, x_min, y_max, x_max) == (0, 0, 199, 199)


def test_detect_border_failure():
    mod = importlib.import_module("think.detect_border")
    img = Image.new("RGB", (100, 100), "white")
    with pytest.raises(ValueError):
        mod.detect_border(img, (0, 0, 255))
