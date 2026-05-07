# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe.see utilities."""

import pytest
from PIL import Image

from solstone.observe.see import (
    decode_frames,
    draw_bounding_box,
    image_to_jpeg_bytes,
)


def test_draw_bounding_box():
    """Test drawing bounding box on image."""
    img = Image.new("RGB", (100, 100), color="white")

    # Draw a box - should not crash
    box_2d = [10, 20, 30, 40]  # y_min, x_min, y_max, x_max
    draw_bounding_box(img, box_2d, color="red", width=3)

    # Image should still be same size (mutated in place)
    assert img.size == (100, 100)


def test_image_to_jpeg_bytes():
    """Test converting image to JPEG bytes."""
    img = Image.new("RGB", (50, 50), color="green")

    # Convert to JPEG
    jpeg_bytes = image_to_jpeg_bytes(img, quality=85)

    assert isinstance(jpeg_bytes, bytes)
    assert len(jpeg_bytes) > 0
    # JPEG files start with FF D8 magic bytes
    assert jpeg_bytes[:2] == b"\xff\xd8"


def test_decode_frames_empty_list():
    """Test decode_frames with empty frame list."""
    result = decode_frames("dummy.mp4", [])
    assert result == []


def test_decode_frames_missing_frame_id():
    """Test decode_frames raises error when frame_id is missing."""
    frames = [{"timestamp": 1.0}]

    with pytest.raises(ValueError, match="must have 'frame_id' field"):
        decode_frames("dummy.mp4", frames)


def test_decode_frames_uses_one_based_frame_ids(monkeypatch):
    """Test decode_frames maps 1-based frame_id values to decoded frames."""
    import numpy as np

    class FakeFrame:
        def __init__(self, color: int):
            self.pts = 1
            self._color = color

        def to_ndarray(self, format: str):
            assert format == "rgb24"
            return np.full((2, 2, 3), self._color, dtype=np.uint8)

    class FakeContainer:
        def __init__(self):
            self.streams = type("Streams", (), {"video": [object()]})
            self._frames = [FakeFrame(10), FakeFrame(20)]

        def decode(self, stream):
            return iter(self._frames)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeAv:
        @staticmethod
        def open(path):
            return FakeContainer()

    monkeypatch.setitem(__import__("sys").modules, "av", FakeAv)

    frames = [{"frame_id": 1}, {"frame_id": 2}]
    images = decode_frames("dummy.mp4", frames, annotate_boxes=False)

    assert images[0].getpixel((0, 0)) == (10, 10, 10)
    assert images[1].getpixel((0, 0)) == (20, 20, 20)


def test_decode_frames_stops_after_highest_requested_frame(monkeypatch):
    """Test decode_frames exits once it has filled the highest requested frame."""
    import numpy as np

    seen = {"count": 0}

    class FakeFrame:
        def __init__(self, color: int):
            self.pts = 1
            self._color = color

        def to_ndarray(self, format: str):
            assert format == "rgb24"
            return np.full((2, 2, 3), self._color, dtype=np.uint8)

    class FakeContainer:
        def __init__(self):
            self.streams = type("Streams", (), {"video": [object()]})

        def decode(self, stream):
            for index in range(30):
                seen["count"] += 1
                yield FakeFrame(index)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeAv:
        @staticmethod
        def open(path):
            return FakeContainer()

    monkeypatch.setitem(__import__("sys").modules, "av", FakeAv)

    frames = [{"frame_id": 7}, {"frame_id": 12}, {"frame_id": 23}]
    images = decode_frames("dummy.mp4", frames, annotate_boxes=False)

    assert seen["count"] == 23
    assert all(image is not None for image in images)
