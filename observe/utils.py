"""Utilities for working with screencasts and video files."""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".webm", ".mp4", ".mov")


def extract_descriptive_suffix(filename: str) -> str:
    """
    Extract descriptive suffix from media filename.

    Returns the portion after the segment (HHMMSS_LEN), preserving
    the descriptive information for the final filename in the segment directory.

    Parameters
    ----------
    filename : str
        Filename stem (without extension), e.g., "143022_300_audio"

    Returns
    -------
    str
        Descriptive suffix (e.g., "audio", "screen", "mic_sys"), or "raw" if none

    Examples
    --------
    >>> extract_descriptive_suffix("143022_300_audio")
    "audio"
    >>> extract_descriptive_suffix("143022_300_screen")
    "screen"
    >>> extract_descriptive_suffix("143022_300_mic_sys")
    "mic_sys"
    >>> extract_descriptive_suffix("143022_300")
    "raw"
    """
    parts = filename.split("_")

    # Filename format: HHMMSS_LEN[_descriptive_text...]
    # First part must be 6-digit timestamp
    if not parts or not parts[0].isdigit() or len(parts[0]) != 6:
        raise ValueError(
            f"Invalid filename format: {filename} (must start with HHMMSS)"
        )

    # Second part must be numeric duration suffix
    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(
            f"Invalid filename format: {filename} (must have HHMMSS_LEN format)"
        )

    # HHMMSS_LEN_suffix... - join remaining parts as descriptive suffix
    if len(parts) > 2:
        return "_".join(parts[2:])
    else:
        return "raw"


def get_segment_key(media_path: Path) -> str | None:
    """
    Extract segment key from a media file path.

    Checks parent directory first (for files already in segment dirs),
    then falls back to filename stem (for files in day root).

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
    >>> get_segment_key(Path("/journal/20250101/143022_300_audio.flac"))
    "143022_300"
    >>> get_segment_key(Path("/journal/20250101/random.txt"))
    None
    """
    from think.utils import segment_key

    # Check if parent directory is a segment (file already moved)
    parent_segment = segment_key(media_path.parent.name)
    if parent_segment:
        return parent_segment

    # Check if filename contains segment (file in day root)
    return segment_key(media_path.stem)


def segment_and_suffix(media_path: Path) -> tuple[str, str]:
    """
    Extract segment key and descriptive suffix from a media file path.

    Handles both files in day root (YYYYMMDD/HHMMSS_LEN_suffix.ext) and
    files already in segment directories (YYYYMMDD/HHMMSS_LEN/suffix.ext).

    Parameters
    ----------
    media_path : Path
        Path to media file (audio or video)

    Returns
    -------
    tuple[str, str]
        (segment_key, suffix) - e.g., ("143022_300", "audio")

    Raises
    ------
    ValueError
        If the path doesn't contain a valid segment key

    Examples
    --------
    >>> segment_and_suffix(Path("/journal/20250101/143022_300_audio.flac"))
    ("143022_300", "audio")
    >>> segment_and_suffix(Path("/journal/20250101/143022_300/audio.flac"))
    ("143022_300", "audio")
    """
    from think.utils import segment_key

    # Check if parent directory is a segment (file already moved)
    parent_segment = segment_key(media_path.parent.name)
    if parent_segment:
        # File is in segment dir - stem is the suffix
        return parent_segment, media_path.stem

    # File is in day root - extract segment from filename
    segment = segment_key(media_path.stem)
    if segment is None:
        raise ValueError(
            f"Invalid media filename: {media_path.stem} (must contain HHMMSS_LEN)"
        )

    suffix = extract_descriptive_suffix(media_path.stem)
    return segment, suffix


def parse_screen_filename(filename: str) -> tuple[str, str]:
    """
    Parse position and connector/displayID from a per-monitor screen filename.

    Handles both pre-move filenames (with segment prefix) and post-move filenames
    (in segment directory without prefix). Works with both GNOME connector IDs
    (e.g., "DP-3") and macOS displayIDs (e.g., "1").

    Parameters
    ----------
    filename : str
        Filename stem (without extension), e.g.:
        - "143022_300_center_DP-3_screen" (GNOME pre-move)
        - "143022_300_center_1_screen" (macOS pre-move)
        - "center_DP-3_screen" (GNOME post-move)
        - "center_1_screen" (macOS post-move)

    Returns
    -------
    tuple[str, str]
        (position, connector) tuple, e.g., ("center", "DP-3") or ("center", "1")
        Returns ("unknown", "unknown") if pattern doesn't match

    Examples
    --------
    >>> parse_screen_filename("143022_300_center_DP-3_screen")
    ("center", "DP-3")
    >>> parse_screen_filename("143022_300_center_1_screen")
    ("center", "1")
    >>> parse_screen_filename("center_DP-3_screen")
    ("center", "DP-3")
    >>> parse_screen_filename("center_1_screen")
    ("center", "1")
    >>> parse_screen_filename("143022_300_screen")
    ("unknown", "unknown")
    """
    # Pattern 1: HHMMSS_LEN_position_connector_screen (pre-move)
    # Connector can be alphanumeric with hyphens (GNOME: DP-3) or just numeric (macOS: 1)
    match = re.match(r"^\d{6}_\d+_([a-z-]+)_([A-Za-z0-9-]+)_screen$", filename)
    if match:
        return match.group(1), match.group(2)

    # Pattern 2: position_connector_screen (post-move, in segment directory)
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
