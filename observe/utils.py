# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Utilities for working with media files (audio and video) and shared observer helpers."""

import datetime
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from think.utils import day_path

logger = logging.getLogger(__name__)

# Standard sample rate for audio processing
SAMPLE_RATE = 16000

VIDEO_EXTENSIONS = (".webm", ".mp4", ".mov")
AUDIO_EXTENSIONS = (".flac", ".ogg", ".m4a")


def get_output_dir(audio_path: Path) -> Path:
    """Get output directory for audio processing artifacts.

    For audio files in segment directories, returns <segment>/<stem>/ folder.
    E.g., 20250101/120000_300/audio.flac -> 20250101/120000_300/audio/

    Parameters
    ----------
    audio_path : Path
        Path to audio file in segment directory

    Returns
    -------
    Path
        Output directory path (parent / stem)
    """
    return audio_path.parent / audio_path.stem


def prepare_audio_file(raw_path: Path, sample_rate: int = SAMPLE_RATE) -> Path:
    """Prepare audio file for processing, converting M4A if needed.

    Returns path to a file suitable for transcription/embedding (mono FLAC).
    For M4A files, converts to temporary FLAC, mixing all audio streams.

    M4A files from sck-cli contain two mono streams: track 0 = system audio,
    track 1 = microphone. Both are decoded and mixed together.

    Parameters
    ----------
    raw_path : Path
        Path to audio file (.flac or .m4a)
    sample_rate : int
        Target sample rate (default: 16000)

    Returns
    -------
    Path
        Path to audio file ready for processing. For .flac files, returns
        the original path. For .m4a files, returns path to temporary .flac
        file that caller should delete after use.

    Raises
    ------
    ValueError
        If no audio streams found in M4A file
    """
    import av

    if raw_path.suffix.lower() != ".m4a":
        return raw_path

    logger.info(f"Converting m4a to FLAC: {raw_path}")

    # First pass: count streams
    container = av.open(str(raw_path))
    num_streams = len(list(container.streams.audio))
    container.close()

    if num_streams == 0:
        raise ValueError(f"No audio streams found in {raw_path}")

    # Decode each stream separately (PyAV requires fresh container per stream)
    # sck-cli produces: track 0 = system audio, track 1 = microphone
    stream_data = []
    for stream_idx in range(num_streams):
        container = av.open(str(raw_path))
        stream = list(container.streams.audio)[stream_idx]

        resampler = av.audio.resampler.AudioResampler(
            format="flt", layout="mono", rate=sample_rate
        )
        chunks = []
        for frame in container.decode(stream):
            for out_frame in resampler.resample(frame):
                arr = out_frame.to_ndarray()
                chunks.append(arr)

        container.close()

        if chunks:
            combined = np.concatenate(chunks, axis=1).flatten()
            stream_data.append(combined)
            logger.info(
                f"  Stream {stream_idx}: {len(combined)} samples "
                f"({len(combined) / sample_rate:.1f}s)"
            )

    if not stream_data:
        raise ValueError(f"No audio data decoded from {raw_path}")

    # Mix all streams together
    if len(stream_data) == 1:
        mixed = stream_data[0]
    else:
        # Pad shorter streams to match longest
        max_len = max(len(s) for s in stream_data)
        padded = []
        for s in stream_data:
            if len(s) < max_len:
                s = np.pad(s, (0, max_len - len(s)), mode="constant")
            padded.append(s)
        # Average all streams
        mixed = np.mean(padded, axis=0)
        logger.info(f"  Mixed {len(stream_data)} streams -> {len(mixed)} samples")

    # Write to temporary FLAC in same directory
    temp_path = raw_path.with_suffix(".tmp.flac")
    audio_int16 = (np.clip(mixed, -1.0, 1.0) * 32767).astype(np.int16)
    sf.write(temp_path, audio_int16, sample_rate, format="FLAC")

    return temp_path


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


# -----------------------------------------------------------------------------
# Observer utilities (shared between Linux and macOS observers)
# -----------------------------------------------------------------------------


def get_timestamp_parts(timestamp: float | None = None) -> tuple[str, str]:
    """Get date and time parts from timestamp.

    Args:
        timestamp: Unix timestamp (default: current time)

    Returns:
        Tuple of (date_part, time_part) like ("20250101", "143022")
    """
    if timestamp is None:
        timestamp = time.time()
    dt = datetime.datetime.fromtimestamp(timestamp)
    date_part = dt.strftime("%Y%m%d")
    time_part = dt.strftime("%H%M%S")
    return date_part, time_part


def create_draft_folder(start_at: float) -> str:
    """Create a draft folder for the current segment.

    Args:
        start_at: Segment start timestamp (wall-clock time)

    Returns:
        Path to the draft folder (YYYYMMDD/HHMMSS_draft/)
    """
    date_part, time_part = get_timestamp_parts(start_at)
    day_dir = day_path(date_part)

    # Create draft folder: YYYYMMDD/HHMMSS_draft/
    draft_name = f"{time_part}_draft"
    draft_path = str(day_dir / draft_name)
    os.makedirs(draft_path, exist_ok=True)

    return draft_path
