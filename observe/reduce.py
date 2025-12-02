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
import os
import sys
import time
from pathlib import Path
from typing import Any

from observe.utils import load_analysis_frames
from think.callosum import callosum_send
from think.entities import load_entity_names
from think.models import GEMINI_FLASH, gemini_generate
from think.utils import setup_cli

logger = logging.getLogger(__name__)


def format_screen(
    entries: list[dict],
    context: dict | None = None,
) -> tuple[list[dict], dict]:
    """Format screen.jsonl entries to markdown chunks.

    This is the formatter function used by the formatters registry.

    Args:
        entries: Raw JSONL entries (first line is metadata, rest are frames)
        context: Optional context with:
            - file_path: Path to JSONL file (for extracting base timestamp)
            - entity_names: Comma-separated entity names for context
            - include_entity_context: Whether to include entity header

    Returns:
        Tuple of (chunks, meta) where:
            - chunks: List of {"timestamp": int, "markdown": str} dicts, one per frame
            - meta: Dict with optional "header" and "error" keys
    """
    ctx = context or {}
    file_path = ctx.get("file_path")
    entity_names = ctx.get("entity_names", "")
    include_entity_context = ctx.get("include_entity_context", True)

    # Separate metadata from frame entries
    # Only first entry can be metadata (has "raw" key but no "timestamp" key)
    metadata = {}
    frame_entries = []
    skipped_count = 0
    for i, entry in enumerate(entries):
        if i == 0 and "timestamp" not in entry and "raw" in entry:
            metadata = entry
        elif "timestamp" in entry:
            frame_entries.append(entry)
        else:
            skipped_count += 1

    # Build meta dict with optional error
    meta: dict[str, Any] = {}
    if skipped_count > 0:
        error_msg = f"Skipped {skipped_count} entries missing 'timestamp' field"
        if file_path:
            error_msg += f" in {file_path}"
        meta["error"] = error_msg
        logger.info(error_msg)

    chunks: list[dict[str, Any]] = []

    # Build header with entity context if requested
    if include_entity_context and entity_names:
        header_lines = [
            "# Entity Context",
            "",
            f"Frequently used names that may appear: {entity_names}",
            "",
            "---",
            "",
            "# Frame Analyses",
        ]
        meta["header"] = "\n".join(header_lines)
    else:
        meta["header"] = "# Frame Analyses"

    # Extract base timestamp from segment directory (HHMMSS)
    # Expected structure: YYYYMMDD/HHMMSS_LEN/screen.jsonl
    base_hour = base_minute = base_second = 0
    if file_path:
        try:
            from think.utils import segment_parse

            # Get segment start time from parent directory
            file_path = Path(file_path)
            start_time, _ = segment_parse(file_path.parent.name)
            if start_time:
                base_hour = start_time.hour
                base_minute = start_time.minute
                base_second = start_time.second
        except (ValueError, AttributeError):
            pass

    # Check if multiple monitors present
    monitors_present = set(frame.get("monitor", "0") for frame in frame_entries)
    multiple_monitors = len(monitors_present) > 1

    # Sort all frames chronologically
    sorted_frames = sorted(frame_entries, key=lambda f: f.get("timestamp", 0))

    for frame in sorted_frames:
        lines = []

        # Calculate absolute time
        frame_offset = frame.get("timestamp", 0)
        total_seconds = (
            base_hour * 3600 + base_minute * 60 + base_second + int(frame_offset)
        )
        abs_hour = (total_seconds // 3600) % 24
        abs_minute = (total_seconds // 60) % 60
        abs_second = total_seconds % 60

        # Build frame header with timestamp and optional monitor info
        frame_header = f"### {abs_hour:02d}:{abs_minute:02d}:{abs_second:02d}"

        if multiple_monitors:
            monitor_id = frame.get("monitor", "0")
            monitor_position = frame.get("monitor_position")

            if monitor_position:
                frame_header += f" (Monitor {monitor_id} - {monitor_position})"
            else:
                frame_header += f" (Monitor {monitor_id})"

        lines.append(frame_header)
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

        chunks.append({"timestamp": int(frame_offset), "markdown": "\n".join(lines)})

    return chunks, meta


def assemble_markdown(
    frames: list[dict],
    entity_names: str = "",
    video_path: Path | None = None,
    include_entity_context: bool = True,
) -> str:
    """Assemble markdown document from frame analyses.

    This is a convenience wrapper around format_screen() that returns
    a single concatenated markdown string.

    Parameters
    ----------
    frames : list[dict]
        Frame analysis results
    entity_names : str, optional
        Comma-separated entity names for context (default: "")
    video_path : Path, optional
        Path to video file (for extracting base timestamp). If None,
        uses timestamps as-is without base time calculation (default: None)
    include_entity_context : bool, optional
        Whether to include entity context header (default: True)

    Returns
    -------
    str
        Markdown document
    """
    context = {
        "file_path": video_path,
        "entity_names": entity_names,
        "include_entity_context": include_entity_context,
    }
    chunks, meta = format_screen(frames, context)
    parts = []
    if meta.get("header"):
        parts.append(meta["header"])
    parts.extend(chunk["markdown"] for chunk in chunks)
    return "\n".join(parts)


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


def reduce_analysis(jsonl_path: Path) -> int:
    """
    Reduce analysis JSONL to markdown summary.

    Parameters
    ----------
    jsonl_path : Path
        Path to analysis JSONL file

    Returns
    -------
    int
        Exit code (0 success, 1 error)
    """
    start_time = time.time()

    # Derive paths from JSONL
    analysis_path = jsonl_path
    summary_path = jsonl_path.parent / f"{jsonl_path.stem}.md"
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
    markdown = assemble_markdown(frames, entity_names, jsonl_path)

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
        from think.crumbs import (  # Local import to avoid circular dependency
            CrumbBuilder,
        )

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

    # Emit completion event
    journal_path = Path(os.getenv("JOURNAL_PATH", ""))
    duration_ms = int((time.time() - start_time) * 1000)

    try:
        rel_input = analysis_path.relative_to(journal_path)
        rel_output = summary_path.relative_to(journal_path)
    except ValueError:
        rel_input = analysis_path
        rel_output = summary_path

    callosum_send(
        "observe",
        "reduced",
        input=str(rel_input),
        output=str(rel_output),
        duration_ms=duration_ms,
    )

    return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Reduce screencast analysis to markdown summary"
    )
    parser.add_argument(
        "--day",
        type=str,
        required=True,
        help="Day in YYYYMMDD format",
    )
    parser.add_argument(
        "--segment",
        type=str,
        required=True,
        help="Segment key (HHMMSS or HHMMSS_LEN)",
    )
    args = setup_cli(parser)

    # Construct path from semantic arguments
    journal_path = Path(os.getenv("JOURNAL_PATH", ""))
    if not journal_path.is_dir():
        logger.error("JOURNAL_PATH not set or invalid")
        sys.exit(1)

    day_dir = journal_path / args.day
    if not day_dir.exists():
        logger.error(f"Day directory not found: {day_dir}")
        sys.exit(1)

    segment_dir = day_dir / args.segment
    if not segment_dir.exists():
        logger.error(f"Segment directory not found: {segment_dir}")
        sys.exit(1)

    jsonl_path = segment_dir / "screen.jsonl"
    if not jsonl_path.exists():
        logger.error(f"Analysis file not found: {jsonl_path}")
        sys.exit(1)

    exit_code = reduce_analysis(jsonl_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
