# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe.aruco ArUco marker detection and masking."""

import cv2
import numpy as np
from PIL import Image

from solstone.observe.aruco import (
    CORNER_TAG_IDS,
    detect_markers,
    mask_convey_region,
    polygon_area,
)


def test_corner_tag_ids():
    """Test that corner tag IDs match expected values."""
    assert CORNER_TAG_IDS == {2, 4, 6, 7}


def test_polygon_area_square():
    """Test polygon area calculation for a square."""
    # 100x100 square
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert polygon_area(polygon) == 10000.0


def test_polygon_area_triangle():
    """Test polygon area calculation for a triangle."""
    # Right triangle with legs 10 and 20
    polygon = [(0, 0), (10, 0), (0, 20)]
    assert polygon_area(polygon) == 100.0  # (10 * 20) / 2


def test_polygon_area_empty():
    """Test polygon area with insufficient points."""
    assert polygon_area([]) == 0.0
    assert polygon_area([(0, 0)]) == 0.0
    assert polygon_area([(0, 0), (1, 1)]) == 0.0


def test_detect_markers_no_markers():
    """Test detect_markers returns None when no markers are present."""
    img = Image.new("RGB", (640, 480), color="white")
    result = detect_markers(img)
    assert result is None


def test_detect_markers_grayscale():
    """Test detect_markers works with grayscale input."""
    img = Image.new("L", (640, 480), color=128)
    result = detect_markers(img)
    assert result is None  # No markers, but shouldn't crash


def test_mask_convey_region():
    """Test masking fills polygon with black."""
    img = Image.new("RGB", (100, 100), color="white")

    # Define a square polygon in the center
    polygon = [(25, 25), (75, 25), (75, 75), (25, 75)]
    mask_convey_region(img, polygon)

    # Check corners are still white
    assert img.getpixel((0, 0)) == (255, 255, 255)
    assert img.getpixel((99, 99)) == (255, 255, 255)

    # Check center is black
    assert img.getpixel((50, 50)) == (0, 0, 0)


def test_mask_convey_region_triangle():
    """Test masking works with non-rectangular polygon."""
    img = Image.new("RGB", (100, 100), color="white")

    # Triangle
    polygon = [(50, 10), (90, 90), (10, 90)]
    mask_convey_region(img, polygon)

    # Center should be black (inside triangle)
    assert img.getpixel((50, 60)) == (0, 0, 0)

    # Top corners should still be white (outside triangle)
    assert img.getpixel((5, 5)) == (255, 255, 255)
    assert img.getpixel((95, 5)) == (255, 255, 255)


def test_detect_markers_with_all_corners():
    """Test detect_markers returns full result with all 4 corner markers."""
    # Create a test image
    img_array = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # Generate and place the 4 corner markers
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 50
    pad = 20

    # Generate and place markers
    for tag_id in [6, 7, 4, 2]:
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, marker_size)
        marker_rgb = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)

        if tag_id == 6:  # TL
            img_array[pad : pad + marker_size, pad : pad + marker_size] = marker_rgb
        elif tag_id == 7:  # TR
            img_array[pad : pad + marker_size, 640 - pad - marker_size : 640 - pad] = (
                marker_rgb
            )
        elif tag_id == 4:  # BL
            img_array[480 - pad - marker_size : 480 - pad, pad : pad + marker_size] = (
                marker_rgb
            )
        elif tag_id == 2:  # BR
            img_array[
                480 - pad - marker_size : 480 - pad,
                640 - pad - marker_size : 640 - pad,
            ] = marker_rgb

    pil_img = Image.fromarray(img_array)

    result = detect_markers(pil_img)

    # Should return dict with markers and polygon
    assert result is not None
    assert "markers" in result
    assert "polygon" in result

    # Should have 4 markers
    assert len(result["markers"]) == 4

    # Each marker should have id and corners
    marker_ids = {m["id"] for m in result["markers"]}
    assert marker_ids == {2, 4, 6, 7}

    for marker in result["markers"]:
        assert "id" in marker
        assert "corners" in marker
        assert len(marker["corners"]) == 4
        for corner in marker["corners"]:
            assert len(corner) == 2
            assert isinstance(corner[0], (int, float))
            assert isinstance(corner[1], (int, float))

    # Polygon should be present (all 4 corners detected)
    assert result["polygon"] is not None
    assert len(result["polygon"]) == 4


def test_detect_markers_partial():
    """Test detect_markers returns markers but no polygon with partial detection."""
    # Create a test image
    img_array = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # Generate and place only 2 corner markers
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 50
    pad = 20

    # Only place TL (6) and TR (7) markers
    for tag_id, pos in [(6, (pad, pad)), (7, (pad, 640 - pad - marker_size))]:
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, marker_size)
        marker_rgb = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
        y, x = pos
        img_array[y : y + marker_size, x : x + marker_size] = marker_rgb

    pil_img = Image.fromarray(img_array)

    result = detect_markers(pil_img)

    # Should return dict with markers but no polygon
    assert result is not None
    assert "markers" in result
    assert "polygon" in result

    # Should have 2 markers
    assert len(result["markers"]) == 2
    marker_ids = {m["id"] for m in result["markers"]}
    assert marker_ids == {6, 7}

    # Polygon should be None (only 2 of 4 corners)
    assert result["polygon"] is None


def test_detect_markers_extrapolated_tl():
    """Test detect_markers extrapolates missing TL corner from 3 present markers."""
    img_array = np.ones((480, 640, 3), dtype=np.uint8) * 255
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 50
    pad = 20

    # Place TR (7), BR (2), BL (4) — omit TL (6)
    for tag_id, (y, x) in [
        (7, (pad, 640 - pad - marker_size)),
        (2, (480 - pad - marker_size, 640 - pad - marker_size)),
        (4, (480 - pad - marker_size, pad)),
    ]:
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, marker_size)
        marker_rgb = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
        img_array[y : y + marker_size, x : x + marker_size] = marker_rgb

    result = detect_markers(Image.fromarray(img_array))

    assert result is not None
    assert result["polygon"] is not None
    assert len(result["polygon"]) == 4
    assert result.get("extrapolated") == 6

    # Extrapolated TL should be within 2px of expected position
    tl = result["polygon"][0]
    assert abs(tl[0] - pad) <= 2
    assert abs(tl[1] - pad) <= 2


def test_detect_markers_two_missing_no_extrapolation():
    """Test detect_markers returns no polygon when only 2 corner tags present."""
    img_array = np.ones((480, 640, 3), dtype=np.uint8) * 255
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 50
    pad = 20

    # Place only TL (6) and BR (2)
    for tag_id, (y, x) in [
        (6, (pad, pad)),
        (2, (480 - pad - marker_size, 640 - pad - marker_size)),
    ]:
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, marker_size)
        marker_rgb = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
        img_array[y : y + marker_size, x : x + marker_size] = marker_rgb

    result = detect_markers(Image.fromarray(img_array))

    assert result is not None
    assert result["polygon"] is None
