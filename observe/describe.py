#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""
Describe screencast videos by detecting significant frame changes.

Processes per-monitor screencast files (.webm/.mp4/.mov), detects changes using
RMS-based comparison, and sends frames for multi-stage LLM analysis:

1. Phase 1: Categorization - All frames get initial category analysis
2. Phase 2: Selection - AI/fallback selects which frames get detailed extraction
3. Phase 3: Extraction - Selected frames get category-specific content extraction

Uses Batch for async batch processing with provider routing via context.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional

import av
from PIL import Image, ImageChops, ImageStat

from observe.aruco import detect_markers, mask_convey_region, polygon_area
from observe.extract import DEFAULT_MAX_EXTRACTIONS, select_frames_for_extraction
from observe.utils import get_segment_key
from think.callosum import callosum_send
from think.muse import load_prompt
from think.utils import get_config, get_journal, setup_cli

logger = logging.getLogger(__name__)


class RequestType(Enum):
    """Type of vision analysis request."""

    DESCRIBE = "describe"  # Initial categorization
    CATEGORY = "category"  # Category-specific follow-up


def _discover_categories() -> dict[str, dict]:
    """
    Discover all categories from categories/ directory.

    Each category is a .md file with JSON frontmatter containing:
    - description (required): Single-line description for categorization prompt
    - output (optional, default: "markdown"): Response format for extraction
    - tier (optional, default: 2): Model tier for this category (1=pro, 2=flash, 3=lite)
    - label (optional): Human-readable name for settings UI
    - group (optional, default: "Screen Analysis"): Category for grouping in settings UI

    Categories with content in the .md file (after frontmatter) get detailed
    extraction analysis using that content as the extraction prompt template.

    Returns
    -------
    dict[str, dict]
        Mapping of category name to metadata (including 'prompt' if extractable)
    """
    categories_dir = Path(__file__).parent / "categories"
    if not categories_dir.exists():
        logger.warning(f"Categories directory not found: {categories_dir}")
        return {}

    categories = {}
    for md_path in categories_dir.glob("*.md"):
        category = md_path.stem

        try:
            prompt_content = load_prompt(category, base_dir=categories_dir)
            metadata = dict(prompt_content.metadata)

            # Validate required field
            if "description" not in metadata:
                logger.warning(f"Category {category} missing 'description' field")
                continue

            # Apply defaults for observation settings
            metadata.setdefault("output", "markdown")

            # Apply defaults for tier routing
            # tier: 1=pro, 2=flash, 3=lite (default: flash)
            metadata.setdefault("tier", 2)
            # label: Human-readable name (default: title-cased category name)
            metadata.setdefault("label", category.replace("_", " ").title())
            # group: Settings UI grouping (default: Screen Analysis)
            metadata.setdefault("group", "Screen Analysis")

            # Store the category context for later resolution
            # The model will be resolved at runtime via generate()
            metadata["context"] = f"observe.describe.{category}"

            # Use content as extraction prompt if non-empty
            if prompt_content.text.strip():
                metadata["prompt"] = prompt_content.text

            categories[category] = metadata
            extractable = "prompt" in metadata
            logger.debug(f"Loaded category: {category} (extractable={extractable})")

        except Exception as e:
            logger.warning(f"Failed to load category {category}: {e}")

    return categories


def _build_categorization_prompt() -> str:
    """
    Build the categorization prompt from template and discovered categories.

    Returns
    -------
    str
        Complete prompt with category list substituted
    """
    # Build category list (alphabetical order)
    category_lines = []
    for name in sorted(CATEGORIES.keys()):
        description = CATEGORIES[name]["description"]
        category_lines.append(f"- {name}: {description}")

    category_list = "\n".join(category_lines)

    return load_prompt(
        "describe",
        base_dir=Path(__file__).parent,
        context={"categories": category_list},
    ).text


# Discover categories at module level
CATEGORIES = _discover_categories()

# Build categorization prompt from template
CATEGORIZATION_PROMPT = _build_categorization_prompt()


class VideoProcessor:
    """Process per-monitor screencast videos and detect significant frame changes."""

    # RMS threshold for frame qualification (5% difference)
    RMS_THRESHOLD = 0.05
    # Downsample size for RMS comparison
    COMPARE_SIZE = (160, 90)
    # Skip frame if Convey UI covers more than this fraction of the frame
    MASK_SKIP_THRESHOLD = 0.8

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        # Store qualified frames as simple list
        self.qualified_frames: List[dict] = []
        # Load entity names for vision analysis context
        from think.entities import load_entity_names

        self.entity_names = load_entity_names()

    def process(self) -> List[dict]:
        """
        Process video and return qualified frames.

        Uses RMS-based comparison on downsampled frames to detect significant
        changes. Caches the downsampled version of the last qualified frame
        to avoid repeated resizing.

        Returns:
            List of qualified frames with timestamp and frame_bytes.
        """
        # Cache for downsampled last qualified frame
        last_qualified_small: Optional[Image.Image] = None

        try:
            with av.open(str(self.video_path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                stream.codec_context.thread_count = 0
                self.width = stream.width
                self.height = stream.height

                frame_count = 0
                for frame in container.decode(video=0):
                    if frame.pts is None:
                        continue

                    timestamp = frame.time if frame.time is not None else 0.0
                    frame_count += 1

                    # Convert to PIL for comparison and bytes conversion
                    arr_rgb = frame.to_ndarray(format="rgb24")
                    pil_img = Image.fromarray(arr_rgb)
                    del arr_rgb

                    # Detect ArUco markers (fiducial corner tags)
                    aruco_result = detect_markers(pil_img)
                    aruco_masked = False
                    if aruco_result is not None and aruco_result["polygon"] is not None:
                        # All 4 corner tags detected - check coverage
                        polygon = [tuple(pt) for pt in aruco_result["polygon"]]
                        mask_area = polygon_area(polygon)
                        frame_area = pil_img.width * pil_img.height
                        if mask_area / frame_area > self.MASK_SKIP_THRESHOLD:
                            # Skip frame entirely - Convey UI dominates
                            pil_img.close()
                            logger.debug(
                                f"Skipping frame at {timestamp:.2f}s "
                                f"(Convey UI covers {mask_area/frame_area:.0%})"
                            )
                            continue
                        # Mask the Convey region with black
                        mask_convey_region(pil_img, polygon)
                        aruco_masked = True

                    # Downsample for comparison
                    current_small = self._downsample(pil_img)

                    # Build frame data dict
                    frame_data: dict = {
                        "frame_id": frame_count,
                        "timestamp": timestamp,
                    }
                    # Include aruco detection result if markers were found
                    if aruco_result is not None:
                        frame_data["aruco"] = {
                            "markers": aruco_result["markers"],
                            "masked": aruco_masked,
                        }

                    # First frame: always qualify (RMS vs nothing = 100% different)
                    if last_qualified_small is None:
                        frame_data["frame_bytes"] = self._frame_to_bytes(pil_img)
                        pil_img.close()

                        self.qualified_frames.append(frame_data)

                        last_qualified_small = current_small
                        logger.debug(f"First frame at {timestamp:.2f}s")
                        continue

                    # Compare current frame with last qualified using RMS
                    rms = self._rms_diff(last_qualified_small, current_small)

                    if rms < self.RMS_THRESHOLD:
                        # Not enough change - skip this frame
                        current_small.close()
                        pil_img.close()
                        continue

                    # Qualified - convert full frame to bytes
                    frame_data["frame_bytes"] = self._frame_to_bytes(pil_img)
                    pil_img.close()

                    self.qualified_frames.append(frame_data)

                    # Update cached downsampled frame
                    last_qualified_small.close()
                    last_qualified_small = current_small

                    logger.debug(
                        f"Qualified frame at {timestamp:.2f}s (RMS: {rms:.3f})"
                    )

                # Clean up last cached frame
                if last_qualified_small is not None:
                    last_qualified_small.close()

                logger.info(
                    f"Processed {frame_count} frames from {self.video_path.name}, "
                    f"{len(self.qualified_frames)} qualified"
                )

        except Exception as e:
            logger.error(
                f"Error processing video {self.video_path}: {e}", exc_info=True
            )
            raise

        return self.qualified_frames

    def _downsample(self, img: Image.Image) -> Image.Image:
        """Downsample image to comparison size."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.resize(self.COMPARE_SIZE, Image.BILINEAR)

    def _rms_diff(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Compute RMS difference between two images, normalized to [0, 1].

        Both images should already be downsampled to COMPARE_SIZE.
        """
        diff = ImageChops.difference(img1, img2)
        stat = ImageStat.Stat(diff)
        # Normalize RMS to [0, 1] by dividing by 255 per channel
        rms = sum(stat.rms) / (len(stat.rms) * 255.0)
        return float(rms)

    def _frame_to_bytes(self, img: Image.Image) -> bytes:
        """
        Convert full frame to PNG bytes.

        Parameters
        ----------
        img : Image.Image
            PIL Image to convert

        Returns
        -------
        bytes
            Image as PNG bytes
        """
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=1)
        return buf.getvalue()

    def _get_category_metadata(self, category: str) -> Optional[dict]:
        """
        Get category metadata if extraction prompt is available.

        Parameters
        ----------
        category : str
            Category from initial analysis

        Returns
        -------
        Optional[dict]
            Category metadata with 'prompt', 'output', 'context' keys,
            or None if no extraction prompt available
        """
        cat_meta = CATEGORIES.get(category)
        if cat_meta and cat_meta.get("prompt"):
            return cat_meta
        return None

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

    async def process_with_vision(
        self,
        max_concurrent: int = 10,
        output_path: Optional[Path] = None,
    ) -> None:
        """
        Process video and write vision analysis results to file.

        Three-phase pipeline:
        1. Categorization: All frames get initial category analysis
        2. Selection: Determine which frames get detailed extraction
        3. Extraction: Selected frames get category-specific content extraction

        Parameters
        ----------
        max_concurrent : int
            Maximum number of concurrent API requests (default: 10)
        output_path : Optional[Path]
            Path to write JSONL output (when None, no output file is written)
        """
        from think.batch import Batch
        from think.models import resolve_provider

        # Load config for max_extractions
        config = get_config()
        max_extractions = config.get("describe", {}).get(
            "max_extractions", DEFAULT_MAX_EXTRACTIONS
        )

        # Use dynamically built categorization prompt
        system_instruction = CATEGORIZATION_PROMPT

        # Process video to get qualified frames (synchronous)
        qualified_frames = self.process()

        # Create batch processor
        batch = Batch(max_concurrent=max_concurrent)

        # Open output file if specified
        output_file = open(output_path, "w") if output_path else None

        try:
            # Write metadata header to JSONL file with actual video filename
            if output_file:
                # Files are in segment directories, filename is simple (e.g., center_DP-3_screen.webm)
                metadata = {"raw": self.video_path.name}

                # Add remote origin if set (from sense.py for remote observer uploads)
                remote = os.getenv("REMOTE_NAME")
                if remote:
                    metadata["remote"] = remote

                # Add segment metadata (from sense.py via SEGMENT_META env var)
                segment_meta_str = os.getenv("SEGMENT_META")
                if segment_meta_str:
                    try:
                        segment_meta = json.loads(segment_meta_str)
                        for key, value in segment_meta.items():
                            metadata[key] = value
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Invalid SEGMENT_META JSON: {segment_meta_str[:100]}"
                        )

                output_file.write(json.dumps(metadata) + "\n")
                output_file.flush()

            # Resolve model for frame description (tier from describe.md frontmatter)
            _, frame_model = resolve_provider("observe.describe.frame")

            # Create vision requests for all qualified frames
            for frame_data in qualified_frames:
                # Load frame image from bytes - keep it open until request completes
                frame_img = Image.open(io.BytesIO(frame_data["frame_bytes"]))

                req = batch.create(
                    contents=self._user_contents(
                        "Analyze this screenshot frame from a screencast recording.",
                        frame_img,
                    ),
                    context="observe.describe.frame",
                    model=frame_model,
                    system_instruction=system_instruction,
                    json_output=True,
                    temperature=0.7,
                    max_output_tokens=1024,
                    thinking_budget=1024,
                )

                # Attach metadata for tracking (store bytes, not PIL images)
                req.frame_id = frame_data["frame_id"]
                req.timestamp = frame_data["timestamp"]
                req.retry_count = 0
                req.frame_bytes = frame_data["frame_bytes"]  # Store bytes for reuse
                req.aruco = frame_data.get(
                    "aruco"
                )  # ArUco detection result (may be None)
                req.request_type = RequestType.DESCRIBE
                req.json_analysis = None  # Will store the JSON analysis result
                req.requests = []  # Track all requests for this frame
                req.initial_image = (
                    frame_img  # Keep reference to close after completion
                )

                batch.add(req)

            # Clear qualified_frames now that all requests are created
            self.qualified_frames.clear()

            # =================================================================
            # PHASE 1: Collect all categorization results
            # =================================================================
            categorized: dict = {}  # frame_id -> request
            total_frames = 0
            failed_frames = 0

            async for req in batch.drain_batch():
                total_frames += 1

                # Check for errors
                has_error = bool(req.error)
                error_msg = req.error

                # Parse JSON analysis
                if not has_error:
                    try:
                        analysis = json.loads(req.response)
                        req.json_analysis = analysis
                    except json.JSONDecodeError as e:
                        has_error = True
                        error_msg = f"Invalid JSON response: {e}"

                # Retry logic (up to 5 attempts total, so 4 retries)
                if has_error and req.retry_count < 4:
                    req.retry_count += 1
                    total_frames -= 1  # Don't count retries
                    batch.add(req)
                    logger.info(
                        f"Retrying frame {req.frame_id} "
                        f"(attempt {req.retry_count + 1}/5): {error_msg}"
                    )
                    continue

                # Track failure after all retries exhausted
                if has_error:
                    failed_frames += 1

                # Record categorization request result
                request_record = {
                    "type": req.request_type.value,
                    "model": req.model_used,
                    "duration": req.duration,
                }
                if req.retry_count > 0:
                    request_record["retries"] = req.retry_count
                req.requests.append(request_record)

                # Store error on request for later output
                if has_error:
                    req.error_msg = error_msg

                # Close initial image - no longer needed for categorization
                if hasattr(req, "initial_image") and req.initial_image:
                    req.initial_image.close()
                    req.initial_image = None

                # Store in categorized dict (keep frame_bytes for extraction)
                categorized[req.frame_id] = req

            logger.info(
                f"Phase 1 complete: {len(categorized)} frames categorized "
                f"({failed_frames} failed)"
            )

            # Check if all frames failed
            if total_frames > 0 and failed_frames == total_frames:
                error_detail = (
                    f"Error details in {output_path}"
                    if output_path
                    else "No output file"
                )
                logger.error(
                    f"All {total_frames} frame(s) failed categorization. "
                    f"Video left in place for retry. {error_detail}"
                )
                raise RuntimeError(
                    f"All {total_frames} frame(s) failed vision analysis after retries"
                )

            # =================================================================
            # PHASE 2: Select frames for extraction
            # =================================================================
            # Build input for selection (only successfully categorized frames)
            categorized_list = [
                {
                    "frame_id": req.frame_id,
                    "timestamp": req.timestamp,
                    "analysis": req.json_analysis,
                }
                for req in categorized.values()
                if req.json_analysis is not None
            ]
            # Sort by frame_id for consistent ordering
            categorized_list.sort(key=lambda x: x["frame_id"])

            # Run selection (pass CATEGORIES for AI-based selection)
            selected_ids = set(
                select_frames_for_extraction(
                    categorized_list, max_extractions, categories=CATEGORIES
                )
            )

            logger.info(
                f"Phase 2 complete: {len(selected_ids)} of {len(categorized_list)} "
                f"frames selected for extraction (max: {max_extractions})"
            )

            # =================================================================
            # PHASE 3: Extract content from selected frames
            # =================================================================
            # Track frames with pending extractions for merging
            frame_results: dict = {}  # frame_id -> result dict
            frame_images: dict = {}  # frame_id -> PIL Image (for cleanup)
            extraction_count = 0

            for frame_id, req in categorized.items():
                has_error = hasattr(req, "error_msg")
                error_msg = getattr(req, "error_msg", None)

                # Build base result
                result = {
                    "frame_id": req.frame_id,
                    "timestamp": req.timestamp,
                    "requests": req.requests,
                }

                if req.aruco:
                    result["aruco"] = req.aruco

                if has_error:
                    result["error"] = error_msg

                if req.json_analysis:
                    result["analysis"] = req.json_analysis

                # Check if this frame is selected for extraction
                if frame_id not in selected_ids or req.json_analysis is None:
                    # Not selected or failed - output immediately with enhanced=false
                    result["enhanced"] = False

                    result_line = json.dumps(result)
                    if output_file:
                        output_file.write(result_line + "\n")
                        output_file.flush()
                    if logger.isEnabledFor(logging.DEBUG):
                        print(result_line, flush=True)

                    # Release frame bytes
                    req.frame_bytes = None
                    req.json_analysis = None
                    continue

                # Frame is selected - determine extractions based on overlap logic
                primary = req.json_analysis.get("primary", "")
                secondary = req.json_analysis.get("secondary", "none")
                overlap = req.json_analysis.get("overlap", True)

                extractions = []

                # Check primary category
                primary_meta = self._get_category_metadata(primary)
                if primary_meta:
                    extractions.append((primary, primary_meta))
                else:
                    logger.warning(
                        f"Frame {frame_id}: category '{primary}' has no extraction prompt"
                    )

                # Check secondary category if no overlap
                if not overlap and secondary != "none":
                    secondary_meta = self._get_category_metadata(secondary)
                    if secondary_meta:
                        extractions.append((secondary, secondary_meta))

                # If no extractions possible, output without enhancement
                if not extractions:
                    result["enhanced"] = False

                    result_line = json.dumps(result)
                    if output_file:
                        output_file.write(result_line + "\n")
                        output_file.flush()
                    if logger.isEnabledFor(logging.DEBUG):
                        print(result_line, flush=True)

                    req.frame_bytes = None
                    req.json_analysis = None
                    continue

                # Queue extraction request(s)
                full_img = Image.open(io.BytesIO(req.frame_bytes))
                frame_images[frame_id] = full_img

                # Store result for merging when extractions complete
                result["enhanced"] = True
                result["pending"] = len(extractions)
                result["content"] = {}
                frame_results[frame_id] = result

                for i, (category, cat_meta) in enumerate(extractions):
                    extraction_count += 1

                    if i == 0:
                        extract_req = req
                        extract_req.category_results = {}
                    else:
                        # Create new request for secondary extraction
                        extract_req = batch.create(
                            contents=[], context=cat_meta["context"]
                        )
                        extract_req.frame_id = req.frame_id
                        extract_req.timestamp = req.timestamp
                        extract_req.aruco = req.aruco
                        extract_req.json_analysis = req.json_analysis
                        extract_req.category_results = {}
                        extract_req.requests = result["requests"]  # Share list

                    extract_req.extraction_category = category
                    extract_req.retry_count = 0
                    extract_req.request_type = RequestType.CATEGORY

                    # Determine output format from metadata
                    is_json = cat_meta.get("output") == "json"

                    # Resolve model for this category context
                    _, cat_model = resolve_provider(cat_meta["context"])

                    batch.update(
                        extract_req,
                        contents=self._user_contents(
                            f"Analyze this {category} screenshot.",
                            full_img,
                            entities=True,
                        ),
                        model=cat_model,
                        system_instruction=cat_meta["prompt"],
                        json_output=is_json,
                        max_output_tokens=10240 if is_json else 8192,
                        thinking_budget=6144 if is_json else 4096,
                        context=cat_meta["context"],
                    )

                logger.info(
                    f"Frame {frame_id}: {len(extractions)} extraction(s) - "
                    f"{', '.join(cat for cat, _ in extractions)}"
                )

            logger.info(f"Phase 3: {extraction_count} extraction request(s) queued")

            # Drain extraction results
            async for req in batch.drain_batch():
                has_error = bool(req.error)
                error_msg = req.error

                # Parse extraction result
                if not has_error:
                    category = req.extraction_category
                    cat_meta = self._get_category_metadata(category)
                    if cat_meta and cat_meta.get("output") == "json":
                        try:
                            result_data = json.loads(req.response)
                            req.category_results[category] = result_data
                        except json.JSONDecodeError as e:
                            has_error = True
                            error_msg = f"Invalid JSON response for {category}: {e}"
                    else:
                        # Markdown output - store as-is
                        req.category_results[category] = req.response

                # Retry logic
                if has_error and req.retry_count < 4:
                    req.retry_count += 1
                    batch.add(req)
                    logger.info(
                        f"Retrying extraction {req.frame_id}/{req.extraction_category} "
                        f"(attempt {req.retry_count + 1}/5): {error_msg}"
                    )
                    continue

                # Record extraction request result
                request_record = {
                    "type": req.request_type.value,
                    "model": req.model_used,
                    "duration": req.duration,
                    "category": req.extraction_category,
                }
                if req.retry_count > 0:
                    request_record["retries"] = req.retry_count
                req.requests.append(request_record)

                # Get the frame result we're merging into
                result = frame_results.get(req.frame_id)
                if result is None:
                    logger.error(f"Extraction result for unknown frame {req.frame_id}")
                    continue

                # Merge extraction result
                if has_error:
                    if "error" not in result:
                        result["error"] = error_msg
                else:
                    for category, cat_result in req.category_results.items():
                        result["content"][category] = cat_result

                # Decrement pending count
                result["pending"] -= 1

                # If all extractions complete, output the result
                if result["pending"] <= 0:
                    del result["pending"]

                    result_line = json.dumps(result)
                    if output_file:
                        output_file.write(result_line + "\n")
                        output_file.flush()
                    if logger.isEnabledFor(logging.DEBUG):
                        print(result_line, flush=True)

                    # Clean up
                    del frame_results[req.frame_id]
                    if req.frame_id in frame_images:
                        frame_images[req.frame_id].close()
                        del frame_images[req.frame_id]

        finally:
            # Always close output file
            if output_file:
                output_file.close()

            # Clean up any remaining frame images (in case of exception)
            if "frame_images" in locals():
                for img in frame_images.values():
                    try:
                        img.close()
                    except Exception:
                        pass
                frame_images.clear()

        # Report any failures
        if failed_frames > 0:
            logger.warning(
                f"{failed_frames}/{total_frames} frame(s) failed categorization."
            )


def output_qualified_frames(
    processor: VideoProcessor, qualified_frames: List[dict]
) -> None:
    """Output qualified frames as JSON."""
    output = {
        "video": str(processor.video_path.name),
        "width": processor.width,
        "height": processor.height,
        "frames": [
            {
                "frame_id": frame["frame_id"],
                "timestamp": frame["timestamp"],
            }
            for frame in qualified_frames
        ],
    }

    print(json.dumps(output, indent=2))


async def async_main():
    """Async CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Describe screencast videos with vision analysis"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to video file in segment directory",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=10,
        help="Max concurrent vision API requests (default: 10)",
    )
    parser.add_argument(
        "--frames-only",
        action="store_true",
        help="Only output frame metadata without vision analysis",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Reprocess file, overwriting existing outputs",
    )
    args = setup_cli(parser)

    video_path = Path(args.video_path)
    if not video_path.exists():
        parser.error(f"Video file not found: {video_path}")

    # Files must be in segment directories (YYYYMMDD/HHMMSS_LEN/)
    segment = get_segment_key(video_path)
    if segment is None:
        parser.error(
            f"Video file must be in a segment directory (HHMMSS_LEN/), "
            f"but parent is: {video_path.parent.name}"
        )

    # Determine output path
    output_path = None
    if not args.frames_only:
        # Output JSONL in same directory, same stem (e.g., center_DP-3_screen.jsonl)
        output_path = video_path.with_suffix(".jsonl")

        # Skip if already processed (unless redo mode)
        if not args.redo and output_path.exists():
            logger.info(f"Already processed: {video_path}")
            return

        if output_path.exists():
            logger.warning(f"Overwriting existing analysis file: {output_path}")

    logger.info(f"Processing video: {video_path}")

    start_time = time.time()

    try:
        processor = VideoProcessor(video_path)

        if args.frames_only:
            # Original behavior: just output frame metadata
            qualified_frames = processor.process()
            output_qualified_frames(processor, qualified_frames)
        else:
            # New behavior: process with vision analysis
            await processor.process_with_vision(
                max_concurrent=args.jobs,
                output_path=output_path,
            )

            # Emit completion event
            if output_path and output_path.exists():
                journal_path = Path(get_journal())

                try:
                    rel_input = video_path.relative_to(journal_path)
                    rel_output = output_path.relative_to(journal_path)
                except ValueError:
                    rel_input = video_path
                    rel_output = output_path

                duration_ms = int((time.time() - start_time) * 1000)

                # Extract day from video path (grandparent is day dir)
                day = video_path.parent.parent.name

                event_fields = {
                    "input": str(rel_input),
                    "output": str(rel_output),
                    "duration_ms": duration_ms,
                }
                if day:
                    event_fields["day"] = day
                if segment:
                    event_fields["segment"] = segment
                remote = os.getenv("REMOTE_NAME")
                if remote:
                    event_fields["remote"] = remote
                callosum_send("observe", "described", **event_fields)
    except Exception as e:
        logger.error(f"Failed to process {video_path}: {e}", exc_info=True)
        raise


def main():
    """CLI entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
