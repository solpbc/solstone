"""Utilities for working with screencasts and video files."""

import json
import logging
import re
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Dict

import av
import numpy as np
from skimage.metrics import structural_similarity as ssim
from think.utils import period_key

logger = logging.getLogger(__name__)


def extract_descriptive_suffix(filename: str) -> str:
    """
    Extract descriptive suffix from media filename.

    Returns the portion after the period (HHMMSS or HHMMSS_LEN), preserving
    the descriptive information for the final filename in the period directory.

    Parameters
    ----------
    filename : str
        Filename stem (without extension), e.g., "143022_300_audio", "143022_screen"

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
    >>> extract_descriptive_suffix("143022_audio")
    "audio"
    >>> extract_descriptive_suffix("143022")
    "raw"
    """
    parts = filename.split("_")

    # Filename format: HHMMSS[_LEN][_descriptive_text...]
    # First part must be 6-digit timestamp
    if not parts or not parts[0].isdigit() or len(parts[0]) != 6:
        raise ValueError(f"Invalid filename format: {filename} (must start with HHMMSS)")

    # Check if second part is numeric duration suffix
    if len(parts) >= 2 and parts[1].isdigit():
        # Has duration suffix: HHMMSS_LEN_suffix...
        # Join remaining parts as descriptive suffix
        if len(parts) > 2:
            return "_".join(parts[2:])
        else:
            return "raw"
    else:
        # No duration suffix: HHMMSS_suffix...
        # Join remaining parts as descriptive suffix
        if len(parts) > 1:
            return "_".join(parts[1:])
        else:
            return "raw"


def parse_monitor_metadata(
    title: str, video_width: int, video_height: int
) -> Dict[str, dict]:
    """
    Parse monitor metadata from video title string.

    Parameters
    ----------
    title : str
        Video title metadata (e.g., "DP-3:center,1920,0,5360,1440 HDMI-4:right,5360,219,7280,1299")
    video_width : int
        Video frame width in pixels
    video_height : int
        Video frame height in pixels

    Returns
    -------
    Dict[str, dict]
        Mapping of monitor_id to monitor info with keys:
        - name: Monitor identifier
        - position: Position label (e.g., "center", "right", "unknown")
        - x1, y1: Top-left coordinates
        - x2, y2: Bottom-right coordinates

        If title is empty or unparseable, returns single monitor covering full frame.

    Examples
    --------
    >>> parse_monitor_metadata("DP-3:center,0,0,1920,1080", 1920, 1080)
    {'DP-3': {'name': 'DP-3', 'position': 'center', 'x1': 0, 'y1': 0, 'x2': 1920, 'y2': 1080}}

    >>> parse_monitor_metadata("", 1920, 1080)
    {'0': {'name': '0', 'position': 'unknown', 'x1': 0, 'y1': 0, 'x2': 1920, 'y2': 1080}}
    """
    if not title:
        # No metadata - return single monitor covering full frame
        return {
            "0": {
                "name": "0",
                "position": "unknown",
                "x1": 0,
                "y1": 0,
                "x2": video_width,
                "y2": video_height,
            }
        }

    monitors = {}
    # Parse space-separated monitor entries
    for entry in title.split():
        # Format: "DP-3:center,1920,0,5360,1440"
        # Monitor name can be any character except ':' or whitespace
        match = re.match(r"([^:\s]+):([^,]+),(\d+),(\d+),(\d+),(\d+)", entry.strip())
        if match:
            monitor_name, position, x1, y1, x2, y2 = match.groups()
            monitors[monitor_name] = {
                "name": monitor_name,
                "position": position,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            }

    if not monitors:
        logger.warning(f"Could not parse monitor metadata from title: {title}")
        # Return single monitor covering full frame
        return {
            "0": {
                "name": "0",
                "position": "unknown",
                "x1": 0,
                "y1": 0,
                "x2": video_width,
                "y2": video_height,
            }
        }

    logger.info(f"Parsed {len(monitors)} monitors from metadata")
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
    # Note: Duplicate frame_ids are normal when multiple monitors qualify the same frame
    frames.sort(key=lambda f: f.get("frame_id", 0))

    # Return header first, then sorted frames
    if header:
        return [header] + frames
    return frames


def get_frames(container: av.container.Container) -> list[tuple[float, int]]:
    """
    Get frames sorted by compressed packet size from a video container.

    Larger packets typically indicate more complex/detailed frames with more
    visual change. This method is fast since it reads packets without decoding.
    Samples frames at 1.0 second intervals.

    Args:
        container: PyAV container opened for reading

    Returns:
        List of (timestamp, packet_size) tuples sorted by packet size descending
    """
    # Scan video packets and collect frame data
    frame_data = []  # List of (timestamp, packet_size)
    sample_interval = 1.0
    last_sampled = -sample_interval

    for packet in container.demux(video=0):
        if packet.pts is None:
            continue

        timestamp = float(packet.pts * packet.time_base)

        # Sample at 1 second intervals
        if timestamp - last_sampled >= sample_interval:
            frame_data.append((timestamp, packet.size))
            last_sampled = timestamp

    # Sort by packet size descending
    frame_data.sort(key=lambda x: x[1], reverse=True)

    return frame_data


def compare_frames(
    frame1: av.VideoFrame,
    frame2: av.VideoFrame,
    block_size: int = 64,
    ssim_threshold: float = 0.90,
    margin: int = 5,
) -> list[dict]:
    """
    Compare two PyAV video frames and return bounding boxes of changed regions.

    Uses block-based SSIM on Y-plane (luma) for efficient detection of perceptual
    changes without full RGB decoding. Optimized to work directly with PyAV frames
    for ~3-5x faster performance.

    Args:
        frame1: First video frame
        frame2: Second video frame
        block_size: Size of comparison blocks in pixels (default 64)
        ssim_threshold: SSIM threshold below which blocks are marked as changed (default 0.90)
        margin: Pixel margin to add around bounding boxes (default 5)

    Returns:
        List of dicts with 'box_2d' key containing [y_min, x_min, y_max, x_max] coordinates
    """
    # Extract Y-plane (luma) directly - equivalent to LAB L-channel
    y_plane1 = frame1.to_ndarray(format="gray")
    y_plane2 = frame2.to_ndarray(format="gray")

    height, width = y_plane1.shape
    grid_rows = ceil(height / block_size)
    grid_cols = ceil(width / block_size)
    changed = [[False] * grid_cols for _ in range(grid_rows)]

    # Compute SSIM for each block
    for i in range(grid_rows):
        for j in range(grid_cols):
            y0 = i * block_size
            x0 = j * block_size
            y1 = min(y0 + block_size, height)
            x1 = min(x0 + block_size, width)
            block1 = y_plane1[y0:y1, x0:x1]
            block2 = y_plane2[y0:y1, x0:x1]
            score, _ = ssim(block1, block2, full=True)
            if score < ssim_threshold:
                changed[i][j] = True

    # Group contiguous changed blocks using DFS
    groups = _group_changed_blocks(changed, grid_rows, grid_cols)

    # Convert groups to bounding boxes
    boxes = _blocks_to_boxes(groups, block_size, width, height, margin)

    return boxes


def _group_changed_blocks(changed, grid_rows, grid_cols):
    """Group contiguous changed blocks using iterative DFS."""
    groups = []
    visited = [[False] * grid_cols for _ in range(grid_rows)]

    def dfs(i, j, group):
        stack = [(i, j)]
        while stack:
            ci, cj = stack.pop()
            if ci < 0 or ci >= grid_rows or cj < 0 or cj >= grid_cols:
                continue
            if visited[ci][cj] or not changed[ci][cj]:
                continue
            visited[ci][cj] = True
            group.append((ci, cj))
            for ni, nj in [(ci - 1, cj), (ci + 1, cj), (ci, cj - 1), (ci, cj + 1)]:
                stack.append((ni, nj))

    for i in range(grid_rows):
        for j in range(grid_cols):
            if changed[i][j] and not visited[i][j]:
                group = []
                dfs(i, j, group)
                groups.append(group)

    return groups


def _blocks_to_boxes(groups, block_size, width, height, margin):
    """Convert groups of changed blocks to bounding boxes."""
    boxes = []
    for group in groups:
        # Calculate bounding box in pixel coordinates
        min_x = width
        min_y = height
        max_x = 0
        max_y = 0
        for i, j in group:
            x0 = j * block_size
            y0 = i * block_size
            x1 = min(x0 + block_size, width)
            y1 = min(y0 + block_size, height)
            min_x = min(min_x, x0)
            min_y = min(min_y, y0)
            max_x = max(max_x, x1)
            max_y = max(max_y, y1)
        # Add margin
        min_x = max(0, min_x - margin)
        min_y = max(0, min_y - margin)
        max_x = min(width, max_x + margin)
        max_y = min(height, max_y + margin)
        # Format as [y_min, x_min, y_max, x_max]
        boxes.append({"box_2d": [min_y, min_x, max_y, max_x]})
    return boxes
