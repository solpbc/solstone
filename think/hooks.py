# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Shared utilities for output extraction hooks.

This module provides common functions used by extraction hooks like
occurrence.py and anticipation.py in the muse/ directory.
"""

import json
import os
from pathlib import Path

# Minimum content length for meaningful event extraction
MIN_EXTRACTION_CHARS = 50


def should_skip_extraction(result: str, context: dict) -> str | None:
    """Check if extraction should be skipped and return reason, or None to proceed.

    Args:
        result: The generated output markdown content.
        context: Hook context dict with meta and span.

    Returns:
        Skip reason string if extraction should be skipped, None otherwise.
    """
    meta = context.get("meta", {})

    # Skip if extraction disabled via journal config
    if meta.get("extract") is False:
        return "extraction disabled via journal config"

    # Skip for JSON output (output IS the structured data)
    if meta.get("output") == "json":
        return "JSON output (already structured)"

    # Skip in span mode (multiple sequential segments)
    if context.get("span"):
        return "span mode"

    # Skip for minimal content
    if len(result.strip()) < MIN_EXTRACTION_CHARS:
        return f"minimal content ({len(result.strip())} chars < {MIN_EXTRACTION_CHARS})"

    return None


def write_events_jsonl(
    events: list[dict],
    topic: str,
    occurred: bool,
    source_output: str,
    capture_day: str,
) -> list[Path]:
    """Write events to facet-based JSONL files.

    Groups events by facet and writes each to the appropriate file:
    facets/{facet}/events/{event_day}.jsonl

    Args:
        events: List of event dictionaries from extraction.
        topic: Source generator topic (e.g., "meetings", "schedule").
        occurred: True for occurrences, False for anticipations.
        source_output: Relative path to source output file.
        capture_day: Day the output was captured (YYYYMMDD).

    Returns:
        List of paths to written JSONL files.
    """
    from think.utils import get_journal

    journal = get_journal()

    # Group events by (facet, event_day)
    grouped: dict[tuple[str, str], list[dict]] = {}

    for event in events:
        facet = event.get("facet", "")
        if not facet:
            continue  # Skip events without facet

        # Determine the event day
        if occurred:
            # Occurrences use capture day
            event_day = capture_day
        else:
            # Anticipations use their scheduled date
            event_date = event.get("date", "")
            # Convert YYYY-MM-DD to YYYYMMDD
            event_day = event_date.replace("-", "") if event_date else capture_day

        if not event_day:
            continue

        key = (facet, event_day)
        if key not in grouped:
            grouped[key] = []

        # Enrich event with metadata
        enriched = dict(event)
        enriched["topic"] = topic
        enriched["occurred"] = occurred
        enriched["source"] = source_output

        grouped[key].append(enriched)

    # Write each group to its JSONL file
    written_paths: list[Path] = []

    for (facet, event_day), facet_events in grouped.items():
        events_dir = Path(journal) / "facets" / facet / "events"
        events_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = events_dir / f"{event_day}.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for event in facet_events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

        written_paths.append(jsonl_path)

    return written_paths


def compute_output_source(context: dict) -> str:
    """Compute relative source output path from hook context.

    Args:
        context: Hook context dict with day, segment, name, output_path, meta.

    Returns:
        Relative path like "20240101/agents/meetings.md".
    """
    from think.utils import get_journal, get_output_topic

    day = context.get("day", "")
    output_path = context.get("output_path", "")
    name = context.get("name", "unknown")
    journal = get_journal()

    try:
        return os.path.relpath(output_path, journal)
    except ValueError:
        segment = context.get("segment")
        topic = get_output_topic(name)
        # Check for facet in meta (for multi-facet agents)
        meta = context.get("meta", {})
        facet = meta.get("facet") if meta else None
        if facet:
            filename = f"{topic}_{facet}.md"
        else:
            filename = f"{topic}.md"
        return os.path.join(
            day,
            "agents" if not segment else segment,
            filename,
        )
