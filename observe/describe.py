#!/usr/bin/env python3
"""
Describe screencast videos by detecting significant frame changes per monitor.

Processes .webm screencast files, detects per-monitor changes using block-based SSIM,
and qualifies frames that meet the 400x400 threshold for Gemini processing.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import av
import numpy as np
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from observe.utils import compare_frames
from think.utils import day_path, setup_cli

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process screencast videos and detect significant frame changes per monitor."""

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.monitors = self._parse_monitor_metadata()
        # Store qualified frames per monitor: {monitor_id: [frame_data, ...]}
        self.qualified_frames: Dict[str, List[dict]] = {
            monitor_id: [] for monitor_id in self.monitors.keys()
        }

    def _parse_monitor_metadata(self) -> Dict[str, dict]:
        """
        Parse monitor metadata from video title.

        Expected format: "0:primary,0,0,1920,1080 1:secondary,1920,0,3840,1080"
        Returns: {monitor_id: {position, x1, y1, x2, y2}}
        """
        try:
            with av.open(str(self.video_path)) as container:
                title = container.metadata.get("title", "")

            if not title:
                logger.warning(f"No metadata in {self.video_path}, using full frame")
                return self._get_default_monitor(container)

            monitors = {}
            # Parse space-separated monitor entries
            for entry in title.split():
                # Format: "0:primary,0,0,1920,1080"
                match = re.match(
                    r"(\d+):([^,]+),(\d+),(\d+),(\d+),(\d+)", entry.strip()
                )
                if match:
                    monitor_id, position, x1, y1, x2, y2 = match.groups()
                    monitors[monitor_id] = {
                        "position": position,
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                    }

            if not monitors:
                logger.warning(f"Could not parse monitor metadata from title: {title}")
                with av.open(str(self.video_path)) as container:
                    return self._get_default_monitor(container)

            logger.info(f"Parsed {len(monitors)} monitors from metadata")
            return monitors

        except Exception as e:
            logger.warning(f"Error parsing monitor metadata: {e}")
            with av.open(str(self.video_path)) as container:
                return self._get_default_monitor(container)

    def _get_default_monitor(self, container) -> Dict[str, dict]:
        """Create default single monitor covering full frame."""
        stream = container.streams.video[0]
        width = stream.width
        height = stream.height
        return {
            "0": {
                "position": "unknown",
                "x1": 0,
                "y1": 0,
                "x2": width,
                "y2": height,
            }
        }

    def process(self) -> Dict[str, List[dict]]:
        """
        Process video and return qualified frames per monitor.

        Returns:
            Dict mapping monitor_id to list of qualified frames with timestamp,
            frame data, and change boxes.
        """
        # Track last qualified frame per monitor
        last_qualified: Dict[str, Optional[av.VideoFrame]] = {
            monitor_id: None for monitor_id in self.monitors.keys()
        }

        try:
            with av.open(str(self.video_path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                stream.codec_context.thread_count = 0

                frame_count = 0
                for frame in container.decode(video=0):
                    if frame.pts is None:
                        continue

                    timestamp = frame.time if frame.time is not None else 0.0
                    frame_count += 1

                    # Process each monitor independently
                    for monitor_id, monitor_info in self.monitors.items():
                        x1, y1 = monitor_info["x1"], monitor_info["y1"]
                        x2, y2 = monitor_info["x2"], monitor_info["y2"]

                        # Slice frame to monitor region (will be done in compare_frames via to_ndarray)
                        # For now, we need to work with the full frame and slice during comparison

                        # First frame: always qualify with full monitor bounds
                        if last_qualified[monitor_id] is None:
                            # Box coordinates relative to monitor slice
                            monitor_width = x2 - x1
                            monitor_height = y2 - y1
                            box_2d = [0, 0, monitor_height, monitor_width]

                            self.qualified_frames[monitor_id].append(
                                {
                                    "frame_id": frame_count,
                                    "timestamp": timestamp,
                                    "frame": frame,
                                    "monitor_bounds": (x1, y1, x2, y2),
                                    "box_2d": box_2d,
                                }
                            )

                            last_qualified[monitor_id] = frame
                            logger.debug(
                                f"Monitor {monitor_id}: First frame at {timestamp:.2f}s"
                            )
                            continue

                        # Compare current frame slice with last qualified
                        # We need to slice both frames for comparison
                        boxes = self._compare_monitor_regions(
                            last_qualified[monitor_id],
                            frame,
                            x1,
                            y1,
                            x2,
                            y2,
                        )

                        if not boxes:
                            continue

                        # Find largest box by area
                        largest_box = max(
                            boxes,
                            key=lambda b: (b["box_2d"][2] - b["box_2d"][0])
                            * (b["box_2d"][3] - b["box_2d"][1]),
                        )

                        y_min, x_min, y_max, x_max = largest_box["box_2d"]
                        width = x_max - x_min
                        height = y_max - y_min

                        # Qualify if largest box meets threshold
                        if width >= 400 and height >= 400:
                            self.qualified_frames[monitor_id].append(
                                {
                                    "frame_id": frame_count,
                                    "timestamp": timestamp,
                                    "frame": frame,
                                    "monitor_bounds": (x1, y1, x2, y2),
                                    "box_2d": largest_box["box_2d"],
                                }
                            )

                            last_qualified[monitor_id] = frame
                            logger.debug(
                                f"Monitor {monitor_id}: Qualified frame at {timestamp:.2f}s "
                                f"(box: {width}x{height})"
                            )

                logger.info(
                    f"Processed {frame_count} frames from {self.video_path.name}"
                )
                for monitor_id, frames in self.qualified_frames.items():
                    logger.info(
                        f"  Monitor {monitor_id}: {len(frames)} qualified frames"
                    )

        except Exception as e:
            logger.error(
                f"Error processing video {self.video_path}: {e}", exc_info=True
            )
            raise

        return self.qualified_frames

    def _compare_monitor_regions(
        self,
        frame1: av.VideoFrame,
        frame2: av.VideoFrame,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> List[dict]:
        """
        Compare monitor regions between two frames.

        Returns boxes relative to the monitor slice coordinates.
        """
        # Create sliced frame views for comparison
        # PyAV frames need to be converted to numpy, then sliced
        arr1 = frame1.to_ndarray(format="gray")
        arr2 = frame2.to_ndarray(format="gray")

        # Slice to monitor region
        slice1 = arr1[y1:y2, x1:x2]
        slice2 = arr2[y1:y2, x1:x2]

        # Create temporary VideoFrames from slices for compare_frames
        # Actually, compare_frames expects VideoFrames, but we can adapt it
        # Let's use the direct numpy comparison approach from observe/utils.py

        return self._compare_slices(slice1, slice2)

    def _compare_slices(
        self,
        slice1: np.ndarray,
        slice2: np.ndarray,
        block_size: int = 64,
        ssim_threshold: float = 0.90,
        margin: int = 5,
    ) -> List[dict]:
        """
        Compare two numpy array slices using block-based SSIM.

        Adapted from observe/utils.py compare_frames to work with numpy arrays.
        """
        from math import ceil
        from skimage.metrics import structural_similarity as ssim

        height, width = slice1.shape
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
                block1 = slice1[y0:y1, x0:x1]
                block2 = slice2[y0:y1, x0:x1]
                score, _ = ssim(block1, block2, full=True)
                if score < ssim_threshold:
                    changed[i][j] = True

        # Group contiguous changed blocks
        groups = self._group_changed_blocks(changed, grid_rows, grid_cols)

        # Convert groups to bounding boxes
        boxes = self._blocks_to_boxes(groups, block_size, width, height, margin)

        return boxes

    def _group_changed_blocks(self, changed, grid_rows, grid_cols):
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

    def _blocks_to_boxes(self, groups, block_size, width, height, margin):
        """Convert groups of changed blocks to bounding boxes."""
        boxes = []
        for group in groups:
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
            boxes.append({"box_2d": [min_y, min_x, max_y, max_x]})
        return boxes


class Describer:
    """Watch for new screencast videos and process them."""

    def __init__(self, journal_dir: Path):
        self.journal_dir = journal_dir
        self.watch_dir: Optional[Path] = None
        self.observer: Optional[Observer] = None

    def _handle_video(self, video_path: Path) -> None:
        """Process a screencast video file."""
        logger.info(f"Processing video: {video_path}")

        try:
            processor = VideoProcessor(video_path)
            qualified_frames = processor.process()

            # For now, output as JSON (stub for Gemini processing)
            self._output_qualified_frames(processor, qualified_frames)

        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}", exc_info=True)

    def _output_qualified_frames(
        self, processor: VideoProcessor, qualified_frames: Dict[str, List[dict]]
    ) -> None:
        """Output qualified frames as JSON for testing."""
        output = {
            "video": str(processor.video_path.name),
            "monitors": [],
        }

        for monitor_id, frames in qualified_frames.items():
            monitor_info = processor.monitors[monitor_id]
            monitor_data = {
                "monitor_id": monitor_id,
                "position": monitor_info.get("position"),
                "bounds": [
                    monitor_info["x1"],
                    monitor_info["y1"],
                    monitor_info["x2"],
                    monitor_info["y2"],
                ],
                "frames": [
                    {
                        "frame_id": frame["frame_id"],
                        "timestamp": frame["timestamp"],
                        "box_2d": frame["box_2d"],
                    }
                    for frame in frames
                ],
            }
            output["monitors"].append(monitor_data)

        print(json.dumps(output, indent=2))

    def start(self):
        """Start watching for new video files."""
        handler = PatternMatchingEventHandler(
            patterns=["*.webm"],
            ignore_directories=True,
            ignore_patterns=["*_live.webm"],
        )

        def on_created(event):
            video_path = Path(event.src_path)
            logger.info(f"New video detected: {video_path}")
            self._handle_video(video_path)

        handler.on_created = on_created

        self.observer = None
        current_day: Optional[str] = None

        try:
            while True:
                today_str = datetime.datetime.now().strftime("%Y%m%d")
                day_dir = day_path()

                if day_dir.exists() and (current_day != today_str):
                    if self.observer:
                        self.observer.stop()
                        self.observer.join()
                    self.observer = Observer()
                    self.observer.schedule(handler, str(day_dir), recursive=False)
                    self.observer.start()
                    self.watch_dir = day_dir
                    current_day = today_str
                    logger.info(f"Watching {day_dir}")

                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            if self.observer:
                self.observer.stop()
                self.observer.join()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Describe screencast videos by detecting significant frame changes"
    )
    parser.add_argument(
        "--process",
        type=str,
        help="Process a specific video file (for testing)",
    )
    args = setup_cli(parser)

    journal = Path(os.getenv("JOURNAL_PATH", ""))
    if not journal.is_dir():
        parser.error("JOURNAL_PATH not set or invalid")

    if args.process:
        # Process a specific video file
        video_path = Path(args.process)
        if not video_path.exists():
            parser.error(f"Video file not found: {video_path}")

        describer = Describer(journal)
        describer._handle_video(video_path)
    else:
        # Start watching for new videos
        describer = Describer(journal)
        describer.start()


if __name__ == "__main__":
    main()
