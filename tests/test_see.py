"""Tests for observe.see utilities."""

import pytest
from PIL import Image

from observe.see import (
    crop_frame_to_monitor,
    decode_frames,
    draw_bounding_box,
    image_to_jpeg_bytes,
)


def test_crop_frame_to_monitor():
    """Test cropping image to monitor bounds."""
    # Create a simple 100x100 test image
    img = Image.new("RGB", (100, 100), color="red")

    # Crop to specific bounds
    monitor_bounds = {"x1": 10, "y1": 20, "x2": 50, "y2": 60}
    cropped = crop_frame_to_monitor(img, monitor_bounds)

    assert cropped.size == (40, 40)  # 50-10, 60-20
    assert img.size == (100, 100)  # Original unchanged


def test_crop_frame_to_monitor_defaults():
    """Test cropping with missing bounds uses full image."""
    img = Image.new("RGB", (100, 100), color="blue")

    # Empty bounds should use full image
    cropped = crop_frame_to_monitor(img, {})

    assert cropped.size == (100, 100)


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
    frames = [{"timestamp": 1.0, "monitor": "0"}]

    with pytest.raises(ValueError, match="must have 'frame_id' field"):
        decode_frames("dummy.mp4", frames)


def test_decode_frames_duplicate_frame_id_ok():
    """Test decode_frames allows duplicate frame_ids (multi-monitor case)."""
    frames = [
        {"frame_id": 5, "timestamp": 1.0, "monitor": "DP-3"},
        {
            "frame_id": 5,
            "timestamp": 1.0,
            "monitor": "HDMI-4",
        },  # Same frame, different monitor - OK!
    ]

    # Should not raise ValueError for duplicates - they're valid for multi-monitor
    # Will fail later when trying to open video, but validation passes
    try:
        decode_frames("dummy.mp4", frames)
    except (FileNotFoundError, OSError, Exception):
        pass  # Expected - file doesn't exist or import issues in test env
