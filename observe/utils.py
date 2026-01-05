# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Utilities for working with media files (audio and video)."""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".webm", ".mp4", ".mov")
AUDIO_EXTENSIONS = (".flac", ".ogg", ".m4a")


def get_segment_key(media_path: Path) -> str | None:
    """
    Extract segment key from a media file path.

    For the new model, files are always in segment directories (HHMMSS_LEN/).
    The segment key is the parent directory name.

    Parameters
    ----------
    media_path : Path
        Path to media file (audio or video)

    Returns
    -------
    str or None
        Segment key in HHMMSS_LEN format, or None if not found

    Examples
    --------
    >>> get_segment_key(Path("/journal/20250101/143022_300/audio.flac"))
    "143022_300"
    >>> get_segment_key(Path("/journal/20250101/random.txt"))
    None
    """
    from think.utils import segment_key

    # Segment key is the parent directory name
    return segment_key(media_path.parent.name)


def segment_and_suffix(media_path: Path) -> tuple[str, str]:
    """
    Extract segment key and descriptive suffix from a media file path.

    For the new model, files are always in segment directories.
    The segment key is the parent directory name, suffix is the file stem.

    Parameters
    ----------
    media_path : Path
        Path to media file (audio or video) in a segment directory

    Returns
    -------
    tuple[str, str]
        (segment_key, suffix) - e.g., ("143022_300", "audio")

    Raises
    ------
    ValueError
        If the parent directory is not a valid segment

    Examples
    --------
    >>> segment_and_suffix(Path("/journal/20250101/143022_300/audio.flac"))
    ("143022_300", "audio")
    >>> segment_and_suffix(Path("/journal/20250101/143022_300/center_DP-3_screen.webm"))
    ("143022_300", "center_DP-3_screen")
    """
    from think.utils import segment_key

    # Segment key is the parent directory name
    segment = segment_key(media_path.parent.name)
    if segment is None:
        raise ValueError(
            f"File not in segment directory: {media_path} "
            f"(parent {media_path.parent.name} is not HHMMSS_LEN format)"
        )

    # Suffix is the file stem
    return segment, media_path.stem


def parse_screen_filename(filename: str) -> tuple[str, str]:
    """
    Parse position and connector/displayID from a per-monitor screen filename.

    Files are in segment directories with format: position_connector_screen.ext
    Works with both GNOME connector IDs (e.g., "DP-3") and macOS displayIDs (e.g., "1").

    Parameters
    ----------
    filename : str
        Filename stem (without extension), e.g.:
        - "center_DP-3_screen" (GNOME)
        - "center_1_screen" (macOS)

    Returns
    -------
    tuple[str, str]
        (position, connector) tuple, e.g., ("center", "DP-3") or ("center", "1")
        Returns ("unknown", "unknown") if pattern doesn't match

    Examples
    --------
    >>> parse_screen_filename("center_DP-3_screen")
    ("center", "DP-3")
    >>> parse_screen_filename("center_1_screen")
    ("center", "1")
    >>> parse_screen_filename("left_HDMI-1_screen")
    ("left", "HDMI-1")
    """
    # Pattern: position_connector_screen
    # Connector can be alphanumeric with hyphens (GNOME: DP-3) or just numeric (macOS: 1)
    match = re.match(r"^([a-z-]+)_([A-Za-z0-9-]+)_screen$", filename)
    if match:
        return match.group(1), match.group(2)

    return "unknown", "unknown"


def assign_monitor_positions(monitors: list[dict]) -> list[dict]:
    """
    Assign position labels to monitors based on relative positions.

    Uses pairwise comparison to determine positions. Vertical labels (top/bottom)
    are only assigned when monitors actually overlap horizontally, avoiding
    phantom relationships from offset monitors.

    Parameters
    ----------
    monitors : list[dict]
        List of monitor dicts, each with keys:
        - id: Monitor identifier (e.g., "DP-3", "HDMI-1")
        - box: [x1, y1, x2, y2] coordinates

    Returns
    -------
    list[dict]
        Same monitors with "position" key added to each:
        - "center": No monitors on both sides
        - "left"/"right": Horizontal position
        - "top"/"bottom": Vertical position (only with horizontal overlap)
        - "left-top", "right-bottom", etc.: Corner positions

    Examples
    --------
    >>> monitors = [
    ...     {"id": "DP-1", "box": [0, 0, 1920, 1080]},
    ...     {"id": "DP-2", "box": [1920, 0, 3840, 1080]},
    ... ]
    >>> result = assign_monitor_positions(monitors)
    >>> result[0]["position"]
    'left'
    >>> result[1]["position"]
    'right'
    """
    if not monitors:
        return []

    if len(monitors) == 1:
        monitors[0]["position"] = "center"
        return monitors

    # Tolerance for center classification
    epsilon = 1

    for m in monitors:
        x1, y1, x2, y2 = m["box"]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        has_left = False
        has_right = False
        has_above = False
        has_below = False

        for other in monitors:
            if other is m:
                continue

            ox1, oy1, ox2, oy2 = other["box"]
            other_center_x = (ox1 + ox2) / 2
            other_center_y = (oy1 + oy2) / 2

            # Horizontal relationship (always check)
            if other_center_x < center_x - epsilon:
                has_left = True
            elif other_center_x > center_x + epsilon:
                has_right = True

            # Vertical relationship only if horizontal overlap exists
            # Overlap means ranges intersect (not just touch)
            h_overlap = (x1 < ox2) and (x2 > ox1)
            if h_overlap:
                if other_center_y < center_y - epsilon:
                    has_above = True
                elif other_center_y > center_y + epsilon:
                    has_below = True

        # Determine horizontal label
        if has_left and has_right:
            h_pos = "center"
        elif has_left:
            h_pos = "right"
        elif has_right:
            h_pos = "left"
        else:
            h_pos = "center"

        # Determine vertical label (only if monitors above/below with overlap)
        if has_above and has_below:
            v_pos = "middle"
        elif has_above:
            v_pos = "bottom"
        elif has_below:
            v_pos = "top"
        else:
            v_pos = None

        # Combine positions
        if v_pos is None:
            position = h_pos
        elif h_pos == "center":
            position = v_pos
        else:
            position = f"{h_pos}-{v_pos}"

        m["position"] = position

    return monitors


def load_analysis_frames(jsonl_path: Path) -> list[dict]:
    """
    Load and parse analysis JSONL, filtering out error frames.

    The first line is a header with metadata (e.g., {"raw": "path"}).
    Subsequent frames are sorted by frame_id before being returned.

    Parameters
    ----------
    jsonl_path : Path
        Path to analysis JSONL file

    Returns
    -------
    list[dict]
        List of valid frame analysis results, with header first and frames sorted by frame_id
    """
    header = None
    frames = []
    try:
        with open(jsonl_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    frame = json.loads(line)
                    # Skip frames with errors
                    if "error" not in frame:
                        # First line without frame_id is the header
                        if "frame_id" not in frame and header is None:
                            header = frame
                        else:
                            frames.append(frame)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Invalid JSON at line {line_num} in {jsonl_path}: {e}"
                    )
    except FileNotFoundError:
        logger.error(f"Analysis file not found: {jsonl_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading {jsonl_path}: {e}")
        return []

    # Sort frames by frame_id for sequential video decoding
    frames.sort(key=lambda f: f.get("frame_id", 0))

    # Return header first, then sorted frames
    if header:
        return [header] + frames
    return frames
