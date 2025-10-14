#!/usr/bin/env python3
"""
Describe screencast videos by detecting significant frame changes per monitor.

Processes .webm screencast files, detects per-monitor changes using block-based SSIM,
and qualifies frames that meet the 400x400 threshold for Gemini processing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import av
import numpy as np
from PIL import Image, ImageDraw

from think.utils import setup_cli

logger = logging.getLogger(__name__)


class RequestType(Enum):
    """Type of vision analysis request."""

    DESCRIBE_JSON = "describe_json"
    DESCRIBE_TEXT = "describe_text"
    DESCRIBE_MEETING = "describe_meeting"


def _load_config() -> dict:
    """
    Load describe.json configuration file.

    Returns
    -------
    dict
        Configuration dictionary

    Raises
    ------
    SystemExit
        If config file is missing or invalid
    """
    config_path = Path(__file__).parent / "describe.json"
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise SystemExit(1)

    try:
        with open(config_path) as f:
            config = json.load(f)
        logger.debug(f"Loaded configuration from {config_path}")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file {config_path}: {e}")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise SystemExit(1)


# Load configuration at module level
CONFIG = _load_config()


class VideoProcessor:
    """Process screencast videos and detect significant frame changes per monitor."""

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.monitors = self._parse_monitor_metadata()
        # Store qualified frames per monitor: {monitor_id: [frame_data, ...]}
        self.qualified_frames: Dict[str, List[dict]] = {
            monitor_id: [] for monitor_id in self.monitors.keys()
        }
        # Load entity names for vision analysis context
        from think.utils import load_entity_names

        self.entity_names = load_entity_names()

    def _parse_monitor_metadata(self) -> Dict[str, dict]:
        """
        Parse monitor metadata from video title.

        Expected format: "DP-3:center,1920,0,5360,1440 HDMI-4:right,5360,219,7280,1299"
        Returns: {monitor_name: {position, x1, y1, x2, y2}}
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
                # Format: "DP-3:center,1920,0,5360,1440"
                # Monitor name can be any character except ':' or whitespace
                match = re.match(
                    r"([^:\s]+):([^,]+),(\d+),(\d+),(\d+),(\d+)", entry.strip()
                )
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

    def _extract_frame_images(self, frame_data: dict) -> Image.Image:
        """
        Extract frame and create cropped image for vision analysis.

        Parameters
        ----------
        frame_data : dict
            Frame data containing frame, monitor_bounds, and box_2d

        Returns
        -------
        Image.Image
            Cropped region expanded by 50px in all directions where possible
        """
        frame = frame_data["frame"]
        x1, y1, x2, y2 = frame_data["monitor_bounds"]
        box_2d = frame_data["box_2d"]  # Relative to monitor

        # Convert PyAV frame to PIL Image
        arr = frame.to_ndarray(format="rgb24")
        full_image = Image.fromarray(arr)

        # Convert box_2d (relative to monitor) to absolute coordinates
        abs_y_min = y1 + box_2d[0]
        abs_x_min = x1 + box_2d[1]
        abs_y_max = y1 + box_2d[2]
        abs_x_max = x1 + box_2d[3]

        # Expand bounds by 50px in all directions where possible
        img_width, img_height = full_image.size
        expanded_x_min = max(0, abs_x_min - 50)
        expanded_y_min = max(0, abs_y_min - 50)
        expanded_x_max = min(img_width, abs_x_max + 50)
        expanded_y_max = min(img_height, abs_y_max + 50)

        # Crop to expanded region
        cropped = full_image.crop(
            (expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_max)
        )

        return cropped

    def _user_contents(self, prompt: str, image, entities: bool = False) -> list:
        """Build contents list with optional entity context."""
        contents = [prompt]
        if entities and self.entity_names:
            contents.append(
                f"These are some frequently used names that you may encounter "
                f"and can be helpful when transcribing for accuracy: {self.entity_names}"
            )
        contents.append(image)
        return contents

    def _move_to_seen(self, media_path: Path) -> Path:
        """Move processed media file to seen/ subdirectory."""
        seen_dir = media_path.parent / "seen"
        try:
            seen_dir.mkdir(exist_ok=True)
            new_path = seen_dir / media_path.name
            media_path.rename(new_path)
            logger.info(f"Moved {media_path} to {seen_dir}")
            return new_path
        except Exception as exc:
            logger.error(f"Failed to move {media_path} to seen: {exc}")
            return media_path

    def _create_crumb(
        self,
        output_path: Path,
        moved_video_path: Path,
        used_prompts: set,
        used_models: set,
    ) -> None:
        """Create crumb file for the analysis output."""
        from think.crumbs import CrumbBuilder

        crumb_builder = CrumbBuilder()
        crumb_builder.add_file(moved_video_path)

        # Add prompt files that were used
        observe_dir = Path(__file__).parent
        for prompt_file in sorted(used_prompts):
            crumb_builder.add_file(observe_dir / prompt_file)

        # Add models that were used
        for model in sorted(used_models):
            crumb_builder.add_model(model)

        crumb_path = crumb_builder.commit(str(output_path))
        logger.info(f"Crumb saved to {crumb_path}")

    async def process_with_vision(
        self,
        use_prompt: str = "describe_json.txt",
        max_concurrent: int = 5,
        output_path: Optional[Path] = None,
    ) -> None:
        """
        Process video and write vision analysis results to file.

        Parameters
        ----------
        use_prompt : str
            Prompt template filename to use (default: describe_json.txt)
        max_concurrent : int
            Maximum number of concurrent API requests (default: 5)
        output_path : Optional[Path]
            Path to write JSONL output (default: {video_stem}.jsonl)
        """
        from think.batch import GeminiBatch
        from think.models import GEMINI_FLASH, GEMINI_LITE

        # Track prompts and models used for crumb file
        used_prompts = {use_prompt}
        used_models = set()

        # Load prompt templates
        prompt_path = Path(__file__).parent / use_prompt
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

        system_instruction = prompt_path.read_text()

        # Load text extraction prompt
        text_prompt_path = Path(__file__).parent / "describe_text.txt"
        if not text_prompt_path.exists():
            raise FileNotFoundError(f"Text prompt not found: {text_prompt_path}")

        text_system_instruction = text_prompt_path.read_text()

        # Load meeting analysis prompt
        meeting_prompt_path = Path(__file__).parent / "describe_meeting.txt"
        if not meeting_prompt_path.exists():
            raise FileNotFoundError(f"Meeting prompt not found: {meeting_prompt_path}")

        meeting_system_instruction = meeting_prompt_path.read_text()

        # Process video to get qualified frames (synchronous)
        qualified_frames = self.process()

        # Create batch processor
        batch = GeminiBatch(max_concurrent=max_concurrent)

        # Open output file if specified
        output_file = open(output_path, "w") if output_path else None

        # Create vision requests for all qualified frames
        for monitor_id, frames in qualified_frames.items():
            for frame_data in frames:
                cropped_img = self._extract_frame_images(frame_data)

                req = batch.create(
                    contents=self._user_contents(
                        "Analyze this screenshot frame from a screencast recording.",
                        cropped_img,
                    ),
                    model=GEMINI_LITE,
                    system_instruction=system_instruction,
                    json_output=True,
                    temperature=0.7,
                    max_output_tokens=3072,
                    thinking_budget=2048,
                )

                # Attach metadata for tracking
                req.frame_id = frame_data["frame_id"]
                req.timestamp = frame_data["timestamp"]
                req.monitor = monitor_id
                req.monitor_position = self.monitors[monitor_id].get("position")
                req.box_2d = frame_data["box_2d"]
                req.retry_count = 0
                req.frame_data = frame_data  # Store for potential full frame access
                req.cropped_img = cropped_img  # Store for potential text extraction
                req.request_type = RequestType.DESCRIBE_JSON
                req.json_analysis = None  # Will store the JSON analysis result
                req.meeting_analysis = None  # Will store meeting analysis if applicable
                req.requests = []  # Track all requests for this frame

                batch.add(req)

        # Stream results as they complete, with retry logic
        async for req in batch.drain_batch():
            # Check for errors
            has_error = bool(req.error)
            error_msg = req.error

            # Handle based on request type
            if not has_error:
                if req.request_type == RequestType.DESCRIBE_JSON:
                    # Parse JSON analysis
                    try:
                        analysis = json.loads(req.response)
                        req.json_analysis = analysis  # Store for follow-up analysis
                    except json.JSONDecodeError as e:
                        has_error = True
                        error_msg = f"Invalid JSON response: {e}"
                elif req.request_type == RequestType.DESCRIBE_MEETING:
                    # Parse meeting analysis
                    try:
                        meeting_data = json.loads(req.response)
                        req.meeting_analysis = meeting_data  # Store meeting analysis
                    except json.JSONDecodeError as e:
                        has_error = True
                        error_msg = f"Invalid JSON response: {e}"

            # Retry logic (up to 5 attempts total, so 4 retries)
            if has_error and req.retry_count < 4:
                req.retry_count += 1
                batch.add(req)
                logger.info(
                    f"Retrying frame {req.frame_id} (attempt {req.retry_count + 1}/5): {error_msg}"
                )
                continue  # Don't output, wait for retry result

            # Record this request's result (after retries are done)
            request_record = {
                "type": req.request_type.value,
                "model": req.model_used,
                "duration": req.duration,
            }
            if req.retry_count > 0:
                request_record["retries"] = req.retry_count

            req.requests.append(request_record)

            # Check if we should trigger follow-up analysis
            should_process_further = (
                not has_error
                and req.request_type == RequestType.DESCRIBE_JSON
                and req.json_analysis
            )

            if should_process_further:
                visible_category = req.json_analysis.get("visible", "")

                # Check for meeting analysis
                if visible_category == "meeting":
                    logger.info(f"Frame {req.frame_id}: Triggering meeting analysis")
                    used_prompts.add("describe_meeting.txt")
                    # Need full frame for meeting analysis (not cropped)
                    frame = req.frame_data["frame"]
                    arr = frame.to_ndarray(format="rgb24")
                    full_image = Image.fromarray(arr)

                    batch.update(
                        req,
                        contents=self._user_contents(
                            "Analyze this meeting screenshot.",
                            full_image,
                            entities=True,
                        ),
                        model=GEMINI_FLASH,
                        system_instruction=meeting_system_instruction,
                        json_output=True,
                        max_output_tokens=10240,
                        thinking_budget=6144,
                    )
                    req.request_type = RequestType.DESCRIBE_MEETING
                    req.retry_count = 0
                    continue  # Don't output yet, wait for meeting analysis

                # Check for text extraction
                text_categories = CONFIG.get("text_extraction_categories", [])
                if visible_category in text_categories:
                    logger.info(
                        f"Frame {req.frame_id}: Triggering text extraction for category '{visible_category}'"
                    )
                    used_prompts.add("describe_text.txt")
                    # Update request for text extraction and re-add
                    batch.update(
                        req,
                        contents=self._user_contents(
                            "Extract text from this screenshot frame.",
                            req.cropped_img,
                            entities=True,
                        ),
                        model=GEMINI_FLASH,
                        system_instruction=text_system_instruction,
                        json_output=False,
                        max_output_tokens=8192,
                        thinking_budget=4096,
                    )
                    req.request_type = RequestType.DESCRIBE_TEXT
                    req.retry_count = 0
                    continue  # Don't output yet, wait for text extraction

            # Final output - this frame is complete
            result = {
                "frame_id": req.frame_id,
                "timestamp": req.timestamp,
                "monitor": req.monitor,
                "box_2d": req.box_2d,
                "requests": req.requests,
            }

            # Add monitor position if available and not "unknown"
            if req.monitor_position and req.monitor_position != "unknown":
                result["monitor_position"] = req.monitor_position

            # Add error at top level if any request failed
            if has_error:
                result["error"] = error_msg

            # Add analysis if we have it
            if req.json_analysis:
                result["analysis"] = req.json_analysis

            # Add meeting analysis if we have it (from DESCRIBE_MEETING)
            if req.meeting_analysis:
                result["meeting_analysis"] = req.meeting_analysis

            # Add extracted text if we have it (from DESCRIBE_TEXT)
            if req.request_type == RequestType.DESCRIBE_TEXT and req.response:
                result["extracted_text"] = req.response

            # Track model usage
            if req.model_used:
                used_models.add(req.model_used)

            # Write to file and optionally to stdout
            result_line = json.dumps(result)
            if output_file:
                output_file.write(result_line + "\n")
                output_file.flush()
            if logger.isEnabledFor(logging.DEBUG):
                print(result_line, flush=True)

        # Close output file, move media, and create crumb
        if output_file:
            output_file.close()
            moved_path = self._move_to_seen(self.video_path)
            self._create_crumb(output_path, moved_path, used_prompts, used_models)


def output_qualified_frames(
    processor: VideoProcessor, qualified_frames: Dict[str, List[dict]]
) -> None:
    """Output qualified frames as JSON."""
    output = {
        "video": str(processor.video_path.name),
        "monitors": [],
    }

    for monitor_id, frames in qualified_frames.items():
        monitor_info = processor.monitors[monitor_id]
        monitor_data = {
            "name": monitor_info.get("name", monitor_id),
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
        # Only include position if it's not "unknown"
        position = monitor_info.get("position")
        if position and position != "unknown":
            monitor_data["position"] = position
        output["monitors"].append(monitor_data)

    print(json.dumps(output, indent=2))


async def async_main():
    """Async CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Describe screencast videos with vision analysis"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to video file to process",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="describe_json.txt",
        help="Prompt template to use (default: describe_json.txt)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=5,
        help="Max concurrent vision API requests (default: 5)",
    )
    parser.add_argument(
        "--frames-only",
        action="store_true",
        help="Only output frame metadata without vision analysis",
    )
    args = setup_cli(parser)

    video_path = Path(args.video_path)
    if not video_path.exists():
        parser.error(f"Video file not found: {video_path}")

    # Determine output path and warn if overwriting
    output_path = None
    if not args.frames_only:
        output_path = video_path.parent / f"{video_path.stem}.jsonl"
        if output_path.exists():
            logger.warning(f"Overwriting existing analysis file: {output_path}")

    logger.info(f"Processing video: {video_path}")

    try:
        processor = VideoProcessor(video_path)

        if args.frames_only:
            # Original behavior: just output frame metadata
            qualified_frames = processor.process()
            output_qualified_frames(processor, qualified_frames)
        else:
            # New behavior: process with vision analysis
            await processor.process_with_vision(
                use_prompt=args.prompt,
                max_concurrent=args.jobs,
                output_path=output_path,
            )
    except Exception as e:
        logger.error(f"Failed to process {video_path}: {e}", exc_info=True)
        raise


def main():
    """CLI entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
