#!/usr/bin/env python3
"""
Describe screencast videos by detecting significant frame changes.

Processes per-monitor screencast files (.webm/.mp4/.mov), detects changes using
RMS-based comparison, and sends frames to Gemini for multi-stage analysis:

1. Initial categorization identifies primary/secondary app categories
2. Follow-up analysis (text extraction or meeting analysis) based on category
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

from think.callosum import callosum_send
from think.utils import setup_cli

logger = logging.getLogger(__name__)


class RequestType(Enum):
    """Type of vision analysis request."""

    DESCRIBE = "describe"  # Initial categorization
    CATEGORY = "category"  # Category-specific follow-up


def _segment_and_suffix(media_path: Path) -> tuple[str, str]:
    """Return segment key and descriptive suffix for a media path."""
    from observe.utils import extract_descriptive_suffix
    from think.utils import segment_key

    segment = segment_key(media_path.stem)
    if segment is None:
        raise ValueError(
            f"Invalid video filename: {media_path.stem} (must be HHMMSS_LEN format)"
        )
    try:
        suffix = extract_descriptive_suffix(media_path.stem)
    except ValueError as exc:
        raise ValueError(
            f"Invalid video filename: {media_path.stem} (must be HHMMSS_LEN format)"
        ) from exc
    return segment, suffix


def _discover_categories() -> dict[str, dict]:
    """
    Discover all categories from categories/ directory.

    Each category has a .json metadata file with:
    - description (required): Single-line description for categorization prompt
    - followup (optional, default: false): Whether to run follow-up analysis
    - output (optional, default: "markdown"): Response format if followup=true
    - iq (optional, default: "lite"): Model tier for follow-up ("lite", "flash", "pro")

    If followup=true, a matching .txt file contains the follow-up prompt.

    Returns
    -------
    dict[str, dict]
        Mapping of category name to metadata (including 'prompt' if followup=true)
    """
    from think.models import GEMINI_FLASH, GEMINI_LITE, GEMINI_PRO

    # Map iq values to model constants
    iq_to_model = {
        "lite": GEMINI_LITE,
        "flash": GEMINI_FLASH,
        "pro": GEMINI_PRO,
    }

    categories_dir = Path(__file__).parent / "categories"
    if not categories_dir.exists():
        logger.warning(f"Categories directory not found: {categories_dir}")
        return {}

    categories = {}
    for json_path in categories_dir.glob("*.json"):
        category = json_path.stem

        try:
            with open(json_path) as f:
                metadata = json.load(f)

            # Validate required field
            if "description" not in metadata:
                logger.warning(f"Category {category} missing 'description' field")
                continue

            # Apply defaults
            metadata.setdefault("followup", False)
            metadata.setdefault("output", "markdown")
            metadata.setdefault("iq", "lite")

            # Map iq to model constant
            iq = metadata["iq"]
            if iq not in iq_to_model:
                logger.warning(
                    f"Category {category} has invalid iq '{iq}', using 'lite'"
                )
                iq = "lite"
            metadata["model"] = iq_to_model[iq]

            # Load prompt if followup is enabled
            if metadata["followup"]:
                txt_path = categories_dir / f"{category}.txt"
                if not txt_path.exists():
                    logger.warning(
                        f"Category {category} has followup=true but no {category}.txt"
                    )
                    continue
                metadata["prompt"] = txt_path.read_text()

            categories[category] = metadata
            logger.debug(
                f"Loaded category: {category} (followup={metadata['followup']})"
            )

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
    template_path = Path(__file__).parent / "describe.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    template = template_path.read_text()

    # Build category list (alphabetical order)
    category_lines = []
    for name in sorted(CATEGORIES.keys()):
        description = CATEGORIES[name]["description"]
        category_lines.append(f"- {name}: {description}")

    category_list = "\n".join(category_lines)

    return template.replace("${CATEGORIES}", category_list)


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

                    # Downsample for comparison
                    current_small = self._downsample(pil_img)

                    # First frame: always qualify (RMS vs nothing = 100% different)
                    if last_qualified_small is None:
                        frame_bytes = self._frame_to_bytes(pil_img)
                        pil_img.close()

                        self.qualified_frames.append(
                            {
                                "frame_id": frame_count,
                                "timestamp": timestamp,
                                "frame_bytes": frame_bytes,
                            }
                        )

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
                    frame_bytes = self._frame_to_bytes(pil_img)
                    pil_img.close()

                    self.qualified_frames.append(
                        {
                            "frame_id": frame_count,
                            "timestamp": timestamp,
                            "frame_bytes": frame_bytes,
                        }
                    )

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
        Get category metadata if follow-up is enabled.

        Parameters
        ----------
        category : str
            Category from initial analysis

        Returns
        -------
        Optional[dict]
            Category metadata with 'prompt', 'output', 'model' keys, or None if no follow-up
        """
        cat_meta = CATEGORIES.get(category)
        if cat_meta and cat_meta.get("followup"):
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

    def _move_to_segment(self, media_path: Path) -> Path:
        """Move media file to its segment and return new path."""
        segment, suffix = _segment_and_suffix(media_path)
        segment_dir = media_path.parent / segment
        try:
            segment_dir.mkdir(exist_ok=True)
            # Preserve the original extension
            ext = media_path.suffix
            new_path = segment_dir / f"{suffix}{ext}"
            media_path.rename(new_path)
            logger.info(f"Moved {media_path} to {segment_dir}")
            return new_path
        except Exception as exc:
            logger.error(f"Failed to move {media_path} to segment: {exc}")
            return media_path

    async def process_with_vision(
        self,
        max_concurrent: int = 10,
        output_path: Optional[Path] = None,
    ) -> None:
        """
        Process video and write vision analysis results to file.

        Parameters
        ----------
        max_concurrent : int
            Maximum number of concurrent API requests (default: 10)
        output_path : Optional[Path]
            Path to write JSONL output (when None, no output file is written)
        """
        from think.batch import GeminiBatch
        from think.models import GEMINI_LITE

        # Use dynamically built categorization prompt
        system_instruction = CATEGORIZATION_PROMPT

        # Process video to get qualified frames (synchronous)
        qualified_frames = self.process()

        # Create batch processor
        batch = GeminiBatch(max_concurrent=max_concurrent)

        # Open output file if specified
        output_file = open(output_path, "w") if output_path else None

        # Write metadata header to JSONL file with actual video filename
        if output_file:
            from observe.utils import extract_descriptive_suffix

            suffix = extract_descriptive_suffix(self.video_path.stem)
            metadata = {"raw": f"{suffix}{self.video_path.suffix}"}
            output_file.write(json.dumps(metadata) + "\n")
            output_file.flush()

        # Create vision requests for all qualified frames
        for frame_data in qualified_frames:
            # Load frame image from bytes - keep it open until request completes
            frame_img = Image.open(io.BytesIO(frame_data["frame_bytes"]))

            req = batch.create(
                contents=self._user_contents(
                    "Analyze this screenshot frame from a screencast recording.",
                    frame_img,
                ),
                model=GEMINI_LITE,
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
            req.request_type = RequestType.DESCRIBE
            req.json_analysis = None  # Will store the JSON analysis result
            req.category_results = {}  # Will store category-specific results
            req.requests = []  # Track all requests for this frame
            req.initial_image = frame_img  # Keep reference to close after completion
            req.pending_follow_ups = 0  # Track how many follow-ups are pending
            req.follow_up_category = None  # Category name for follow-up requests

            batch.add(req)

        # Clear qualified_frames now that all requests are created
        # Bytes are already referenced in request objects, so this allows them
        # to be freed incrementally as requests complete rather than all at the end
        self.qualified_frames.clear()

        # Track success/failure for all frames
        total_frames = 0
        failed_frames = 0

        # Track frames by frame_id for merging follow-up results
        frame_results = {}  # frame_id -> result dict

        # Stream results as they complete, with retry logic
        async for req in batch.drain_batch():
            # Only count initial DESCRIBE requests as frames (not follow-ups)
            if req.request_type == RequestType.DESCRIBE:
                total_frames += 1

            # Check for errors
            has_error = bool(req.error)
            error_msg = req.error

            # Handle based on request type
            if not has_error:
                if req.request_type == RequestType.DESCRIBE:
                    # Parse JSON analysis
                    try:
                        analysis = json.loads(req.response)
                        req.json_analysis = analysis  # Store for follow-up analysis
                    except json.JSONDecodeError as e:
                        has_error = True
                        error_msg = f"Invalid JSON response: {e}"
                elif req.request_type == RequestType.CATEGORY:
                    # Handle category-specific follow-up result
                    category = req.follow_up_category
                    cat_meta = self._get_category_metadata(category)
                    if cat_meta and cat_meta.get("output") == "json":
                        try:
                            result = json.loads(req.response)
                            req.category_results[category] = result
                        except json.JSONDecodeError as e:
                            has_error = True
                            error_msg = f"Invalid JSON response for {category}: {e}"
                    else:
                        # Markdown output - store as-is
                        req.category_results[category] = req.response

            # Retry logic (up to 5 attempts total, so 4 retries)
            if has_error and req.retry_count < 4:
                req.retry_count += 1
                batch.add(req)
                logger.info(
                    f"Retrying frame {req.frame_id} (attempt {req.retry_count + 1}/5): {error_msg}"
                )
                continue  # Don't output, wait for retry result

            # Track failure after all retries exhausted (only for initial requests)
            if has_error and req.request_type == RequestType.DESCRIBE:
                failed_frames += 1

            # Record this request's result (after retries are done)
            request_record = {
                "type": req.request_type.value,
                "model": req.model_used,
                "duration": req.duration,
            }
            if req.retry_count > 0:
                request_record["retries"] = req.retry_count
            if req.follow_up_category:
                request_record["category"] = req.follow_up_category

            req.requests.append(request_record)

            # Check if we should trigger follow-up analysis
            should_process_further = (
                not has_error
                and req.request_type == RequestType.DESCRIBE
                and req.json_analysis
            )

            if should_process_further:
                # Extract categories from analysis
                primary = req.json_analysis.get("primary", "")
                secondary = req.json_analysis.get("secondary", "none")
                overlap = req.json_analysis.get("overlap", True)

                # Determine which categories have follow-up enabled
                primary_meta = self._get_category_metadata(primary)
                secondary_meta = (
                    self._get_category_metadata(secondary)
                    if secondary != "none"
                    else None
                )

                # Build follow-up list: each category with followup=true gets analyzed
                # Primary always triggers if followup is enabled
                # Secondary triggers only if overlap=false
                follow_ups = []

                if primary_meta:
                    follow_ups.append((primary, primary_meta))

                if not overlap and secondary_meta:
                    follow_ups.append((secondary, secondary_meta))

                # Create follow-up requests
                if follow_ups:
                    full_img = Image.open(io.BytesIO(req.frame_bytes))
                    req.pending_follow_ups = len(follow_ups)

                    # Close initial image since DESCRIBE is complete
                    if hasattr(req, "initial_image") and req.initial_image:
                        req.initial_image.close()
                        req.initial_image = None

                    for i, (category, cat_meta) in enumerate(follow_ups):
                        if i == 0:
                            follow_req = req
                        else:
                            follow_req = batch.create(contents=[])
                            follow_req.frame_id = req.frame_id
                            follow_req.timestamp = req.timestamp
                            follow_req.frame_bytes = req.frame_bytes
                            follow_req.json_analysis = req.json_analysis
                            follow_req.category_results = req.category_results
                            follow_req.requests = req.requests
                            follow_req.pending_follow_ups = req.pending_follow_ups

                        follow_req.follow_up_category = category
                        follow_req.retry_count = 0
                        follow_req.request_type = RequestType.CATEGORY

                        # Determine output format from metadata
                        is_json = cat_meta.get("output") == "json"

                        batch.update(
                            follow_req,
                            contents=self._user_contents(
                                f"Analyze this {category} screenshot.",
                                full_img,
                                entities=True,
                            ),
                            model=cat_meta["model"],
                            system_instruction=cat_meta["prompt"],
                            json_output=is_json,
                            max_output_tokens=10240 if is_json else 8192,
                            thinking_budget=6144 if is_json else 4096,
                        )

                    logger.info(
                        f"Frame {req.frame_id}: {len(follow_ups)} follow-up(s) - "
                        f"{', '.join(cat for cat, _ in follow_ups)}"
                    )

                    full_img.close()
                    continue  # Don't output yet, wait for follow-ups

            # Handle follow-up completion for parallel requests
            if req.request_type == RequestType.CATEGORY:
                # Store result in frame_results for merging
                if req.frame_id not in frame_results:
                    frame_results[req.frame_id] = {
                        "frame_id": req.frame_id,
                        "timestamp": req.timestamp,
                        "requests": req.requests,
                        "analysis": req.json_analysis,
                        "pending": req.pending_follow_ups,
                    }
                    if has_error:
                        frame_results[req.frame_id]["error"] = error_msg

                result = frame_results[req.frame_id]

                # Merge this follow-up's category result
                for category, cat_result in req.category_results.items():
                    result[category] = cat_result

                # Update requests list (avoid duplicates by using shared list)
                result["requests"] = req.requests

                # Decrement pending count
                result["pending"] -= 1

                # If all follow-ups complete, output the result
                if result["pending"] <= 0:
                    del result["pending"]  # Remove internal tracking field

                    # Write to file and optionally to stdout
                    result_line = json.dumps(result)
                    if output_file:
                        output_file.write(result_line + "\n")
                        output_file.flush()
                    if logger.isEnabledFor(logging.DEBUG):
                        print(result_line, flush=True)

                    # Clean up frame_results entry
                    del frame_results[req.frame_id]

                    # Aggressively clear heavy fields
                    req.frame_bytes = None
                    req.json_analysis = None
                    req.category_results = None

                continue

            # Final output for frames with no follow-ups (DESCRIBE only)
            result = {
                "frame_id": req.frame_id,
                "timestamp": req.timestamp,
                "requests": req.requests,
            }

            # Add error at top level if any request failed
            if has_error:
                result["error"] = error_msg

            # Add analysis if we have it
            if req.json_analysis:
                result["analysis"] = req.json_analysis

            # Write to file and optionally to stdout
            result_line = json.dumps(result)
            if output_file:
                output_file.write(result_line + "\n")
                output_file.flush()
            if logger.isEnabledFor(logging.DEBUG):
                print(result_line, flush=True)

            # Close all PIL Images associated with this request
            if hasattr(req, "initial_image") and req.initial_image:
                req.initial_image.close()
                req.initial_image = None

            # Aggressively clear heavy fields now that request is finalized
            req.frame_bytes = None
            req.json_analysis = None
            req.category_results = None

        # Close output file
        if output_file:
            output_file.close()

        # Check if all frames failed
        all_failed = total_frames > 0 and failed_frames == total_frames

        if all_failed:
            # Don't move video to segment - leave for retry
            error_detail = (
                f"Error details in {output_path}" if output_path else "No output file"
            )
            logger.error(
                f"All {total_frames} frame(s) failed processing. "
                f"Video left in place for retry. {error_detail}"
            )
            # Clear qualified_frames to free memory before raising
            self.qualified_frames.clear()
            raise RuntimeError(
                f"All {total_frames} frame(s) failed vision analysis after retries"
            )
        else:
            # At least some frames succeeded - move to segment
            if failed_frames > 0:
                logger.warning(
                    f"{failed_frames}/{total_frames} frame(s) failed processing. "
                    f"Moving video to segment anyway."
                )
            if output_path:
                self._move_to_segment(self.video_path)

        # Clear qualified_frames to free memory
        self.qualified_frames.clear()


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
        help="Path to video file to process",
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
    args = setup_cli(parser)

    video_path = Path(args.video_path)
    if not video_path.exists():
        parser.error(f"Video file not found: {video_path}")

    # Determine output path and warn if overwriting
    output_path = None
    segment = None
    suffix = None
    if not args.frames_only:
        # Extract segment and suffix for output naming
        try:
            segment, suffix = _segment_and_suffix(video_path)
        except ValueError as exc:
            parser.error(str(exc))
        segment_dir = video_path.parent / segment
        segment_dir.mkdir(exist_ok=True)
        # Output JSONL matches input filename pattern (e.g., center_DP-3_screen.jsonl)
        output_path = segment_dir / f"{suffix}.jsonl"
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
                journal_path = Path(os.getenv("JOURNAL_PATH", ""))
                # Moved path is in segment: YYYYMMDD/HHMMSS_LEN/suffix.webm
                moved_path = (
                    video_path.parent / segment / f"{suffix}{video_path.suffix}"
                )

                try:
                    rel_input = moved_path.relative_to(journal_path)
                    rel_output = output_path.relative_to(journal_path)
                except ValueError:
                    rel_input = moved_path
                    rel_output = output_path

                duration_ms = int((time.time() - start_time) * 1000)

                callosum_send(
                    "observe",
                    "described",
                    input=str(rel_input),
                    output=str(rel_output),
                    duration_ms=duration_ms,
                )
    except Exception as e:
        logger.error(f"Failed to process {video_path}: {e}", exc_info=True)
        raise


def main():
    """CLI entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
