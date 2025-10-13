#!/usr/bin/env python3
"""
Reduce screencast analysis JSONL to articulate markdown summary.

Takes a video path, loads the corresponding .jsonl file, and generates
a comprehensive markdown summary using Gemini.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from think.crumbs import CrumbBuilder
from think.models import GEMINI_FLASH, gemini_generate
from think.utils import load_entity_names, setup_cli

logger = logging.getLogger(__name__)


def load_analysis_frames(jsonl_path: Path) -> list[dict]:
    """
    Load and parse analysis JSONL, filtering out error frames.

    Parameters
    ----------
    jsonl_path : Path
        Path to analysis JSONL file

    Returns
    -------
    list[dict]
        List of valid frame analysis results
    """
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

    return frames


def assemble_markdown(frames: list[dict], entity_names: str, video_path: Path) -> str:
    """
    Assemble markdown document from frame analyses.

    Parameters
    ----------
    frames : list[dict]
        Frame analysis results
    entity_names : str
        Comma-separated entity names for context
    video_path : Path
        Path to video file (for extracting base timestamp)

    Returns
    -------
    str
        Markdown document for Gemini
    """
    lines = []

    # Add entity context at the top
    if entity_names:
        lines.append("# Entity Context")
        lines.append("")
        lines.append(
            f"Frequently used names that may appear: {entity_names}"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("# Frame Analyses")
    lines.append("")

    # Extract base timestamp from video filename (HHMMSS)
    # Expected format: HHMMSS_screen.ext
    try:
        parts = video_path.stem.split("_")
        if len(parts) >= 2:
            base_time_str = parts[1]  # HHMMSS
            base_hour = int(base_time_str[0:2])
            base_minute = int(base_time_str[2:4])
            base_second = int(base_time_str[4:6])
        else:
            base_hour = base_minute = base_second = 0
    except (ValueError, IndexError):
        base_hour = base_minute = base_second = 0

    # Check if multiple monitors present
    monitors_present = set(frame.get("monitor", "0") for frame in frames)
    multiple_monitors = len(monitors_present) > 1

    # Sort all frames chronologically
    sorted_frames = sorted(frames, key=lambda f: f.get("timestamp", 0))

    for frame in sorted_frames:
        # Calculate absolute time
        frame_offset = frame.get("timestamp", 0)
        total_seconds = base_hour * 3600 + base_minute * 60 + base_second + int(frame_offset)
        abs_hour = (total_seconds // 3600) % 24
        abs_minute = (total_seconds // 60) % 60
        abs_second = total_seconds % 60

        # Build header with timestamp and optional monitor info
        header = f"### {abs_hour:02d}:{abs_minute:02d}:{abs_second:02d}"

        if multiple_monitors:
            monitor_id = frame.get("monitor", "0")
            monitor_position = frame.get("monitor_position")

            if monitor_position:
                header += f" (Monitor {monitor_id} - {monitor_position})"
            else:
                header += f" (Monitor {monitor_id})"

        lines.append(header)
        lines.append("")

        # Add analysis if present
        analysis = frame.get("analysis", {})
        if analysis:
            category = analysis.get("visible", "unknown")
            description = analysis.get("visual_description", "")

            lines.append(f"**Category:** {category}")
            lines.append("")
            if description:
                lines.append(description)
                lines.append("")

        # Add extracted text if present
        extracted_text = frame.get("extracted_text")
        if extracted_text:
            lines.append("**Extracted Text:**")
            lines.append("")
            lines.append("```")
            lines.append(extracted_text.strip())
            lines.append("```")
            lines.append("")

        # Add meeting analysis if present
        meeting = frame.get("meeting_analysis")
        if meeting:
            lines.append("**Meeting Analysis:**")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(meeting, indent=2))
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def call_gemini_with_retry(
    markdown: str, prompt: str, max_retries: int = 3
) -> str | None:
    """
    Call Gemini with retry logic.

    Parameters
    ----------
    markdown : str
        Markdown content to send
    prompt : str
        System instruction
    max_retries : int
        Maximum retry attempts (default: 3)

    Returns
    -------
    str | None
        Generated summary or None on failure
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Calling Gemini (attempt {attempt + 1}/{max_retries})")
            response = gemini_generate(
                contents=markdown,
                model=GEMINI_FLASH,
                system_instruction=prompt,
                temperature=0.5,
                max_output_tokens=16384,
                thinking_budget=8192,
            )
            return response
        except Exception as e:
            logger.warning(f"Gemini call failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} Gemini attempts failed")
                return None

    return None


def reduce_analysis(video_path: Path) -> int:
    """
    Reduce analysis JSONL to markdown summary.

    Parameters
    ----------
    video_path : Path
        Path to original video file

    Returns
    -------
    int
        Exit code (0 success, 1 error)
    """
    # Derive paths
    analysis_path = video_path.parent / f"{video_path.stem}.jsonl"
    summary_path = video_path.parent / f"{video_path.stem}.md"
    prompt_path = Path(__file__).parent / "reduce.txt"

    # Check analysis file exists
    if not analysis_path.exists():
        logger.error(f"Analysis file not found: {analysis_path}")
        return 1

    # Check prompt exists
    if not prompt_path.exists():
        logger.error(f"Prompt file not found: {prompt_path}")
        return 1

    # Load frames
    logger.info(f"Loading frames from {analysis_path}")
    frames = load_analysis_frames(analysis_path)

    if not frames:
        logger.error("No valid frames found in analysis file")
        return 1

    logger.info(f"Loaded {len(frames)} valid frames")

    # Load entity names
    entity_names = load_entity_names()

    # Assemble markdown
    logger.info("Assembling markdown for Gemini")
    markdown = assemble_markdown(frames, entity_names, video_path)

    # Load prompt
    prompt_text = prompt_path.read_text()

    # Call Gemini with retry
    logger.info("Generating summary with Gemini")
    summary = call_gemini_with_retry(markdown, prompt_text, max_retries=3)

    if summary is None:
        logger.error("Failed to generate summary")
        return 1

    # Write summary
    logger.info(f"Writing summary to {summary_path}")
    try:
        summary_path.write_text(summary)
    except Exception as e:
        logger.error(f"Failed to write summary: {e}")
        return 1

    # Create crumb
    logger.info("Creating crumb")
    try:
        crumb_builder = CrumbBuilder()
        crumb_builder.add_file(analysis_path)
        crumb_builder.add_file(prompt_path)
        crumb_builder.add_model(GEMINI_FLASH)
        crumb_path = crumb_builder.commit(str(summary_path))
        logger.info(f"Crumb saved to {crumb_path}")
    except Exception as e:
        logger.warning(f"Failed to create crumb: {e}")
        # Don't fail on crumb creation

    logger.info(f"Summary complete: {summary_path}")
    return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Reduce screencast analysis to markdown summary"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to video file (will look for {stem}.jsonl)",
    )
    args = setup_cli(parser)

    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    exit_code = reduce_analysis(video_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
