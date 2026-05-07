# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""
ArUco marker detection for Convey UI masking.

Detects the 4 corner fiducial tags (ArUco DICT_4X4_50, IDs 2,4,6,7) used in the
Convey web interface to identify and mask self-referential UI regions in
screencast frames before vision processing.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Corner tag IDs from convey/static/tags/
# Tag positions: 6=TL, 7=TR, 4=BL, 2=BR
CORNER_TAG_IDS = {6, 7, 4, 2}

# Singleton detector instance (created on first use)
_detector: Optional[cv2.aruco.ArucoDetector] = None

# Per-tag corner extraction index: which corner of each marker gives the outer bounding point
# ArUco corner order within each marker: [TL(0), TR(1), BR(2), BL(3)]
_CORNER_IDX = {6: 0, 7: 1, 2: 2, 4: 3}


def _get_detector() -> cv2.aruco.ArucoDetector:
    """Get or create the ArUco detector singleton."""
    global _detector
    if _detector is None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        # Tuned for 16px CSS tags (~0.011 rate on 1080p, ~0.005 on 4K).
        # Values below 0.003 trigger an OpenCV 4.13 perf cliff (ms → seconds).
        params.minMarkerPerimeterRate = 0.003
        params.maxMarkerPerimeterRate = 8.0
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        _detector = cv2.aruco.ArucoDetector(dictionary, params)
    return _detector


def _extrapolate_corner(id_to_corners: dict, missing_id: int) -> list:
    known = {
        tag_id: id_to_corners[tag_id].reshape(4, 2)[_CORNER_IDX[tag_id]]
        for tag_id in CORNER_TAG_IDS
        if tag_id in id_to_corners
    }
    # Parallelogram rule: TL + BR = TR + BL (diagonals share midpoint)
    if missing_id == 6:  # TL = TR + BL - BR
        pt = known[7] + known[4] - known[2]
    elif missing_id == 7:  # TR = TL + BR - BL
        pt = known[6] + known[2] - known[4]
    elif missing_id == 2:  # BR = TR + BL - TL
        pt = known[7] + known[4] - known[6]
    else:  # BL (4) = TL + BR - TR
        pt = known[6] + known[2] - known[7]
    return pt.tolist()


def detect_markers(image: Image.Image) -> Optional[dict]:
    """
    Detect ArUco markers in an image and return raw detection data.

    Parameters
    ----------
    image : Image.Image
        PIL Image to scan for ArUco markers

    Returns
    -------
    Optional[dict]
        Detection result with keys:
        - markers: list of {id: int, corners: [[x,y], ...]} for each detected marker
        - polygon: [[x,y], ...] bounding polygon if all 4 corner tags found, else None
        Returns None if no markers detected.
    """
    # Convert PIL to numpy array
    img_array = np.array(image)

    # Convert to grayscale for detection
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Detect markers
    detector = _get_detector()
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None

    # Build raw markers list
    markers = []
    id_to_corners = {}
    for tag_id, pts in zip(ids.flatten().tolist(), corners):
        id_to_corners[tag_id] = pts
        # Convert corners to list of [x, y] pairs
        corner_list = pts.reshape(4, 2).tolist()
        markers.append({"id": tag_id, "corners": corner_list})

    result: dict = {"markers": markers, "polygon": None}

    # Check if all 4 corner tags are present for bounding polygon
    if CORNER_TAG_IDS.issubset(id_to_corners.keys()):
        # Extract outer corners from each tag to form the bounding polygon
        # ArUco corner order within each marker: [TL, TR, BR, BL]
        tl = id_to_corners[6].reshape(4, 2)[0]  # TL tag, TL corner
        tr = id_to_corners[7].reshape(4, 2)[1]  # TR tag, TR corner
        br = id_to_corners[2].reshape(4, 2)[2]  # BR tag, BR corner
        bl = id_to_corners[4].reshape(4, 2)[3]  # BL tag, BL corner
        result["polygon"] = [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]
    elif len(CORNER_TAG_IDS & id_to_corners.keys()) == 3:
        found = CORNER_TAG_IDS & id_to_corners.keys()
        missing_id = next(iter(CORNER_TAG_IDS - found))
        corners = {
            tag_id: id_to_corners[tag_id].reshape(4, 2)[_CORNER_IDX[tag_id]].tolist()
            for tag_id in found
        }
        corners[missing_id] = _extrapolate_corner(id_to_corners, missing_id)
        result["polygon"] = [corners[6], corners[7], corners[2], corners[4]]
        result["extrapolated"] = missing_id

    return result


def mask_convey_region(image: Image.Image, polygon: list[tuple[float, float]]) -> None:
    """
    Mask Convey UI region by filling polygon with black.

    Mutates the image in place.

    Parameters
    ----------
    image : Image.Image
        PIL Image to mask (modified in place)
    polygon : list[tuple[float, float]]
        Polygon coordinates [(x,y), ...] defining the region to mask
    """
    draw = ImageDraw.Draw(image)
    draw.polygon(polygon, fill=(0, 0, 0))


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    """
    Calculate area of a polygon using the shoelace formula.

    Parameters
    ----------
    polygon : list[tuple[float, float]]
        List of (x, y) coordinates

    Returns
    -------
    float
        Area in square pixels
    """
    n = len(polygon)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


__all__ = [
    "CORNER_TAG_IDS",
    "detect_markers",
    "mask_convey_region",
    "polygon_area",
]
