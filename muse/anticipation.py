# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Hook for extracting anticipation events from generator output results.

This hook is invoked via "hook": "anticipation" in generator frontmatter.
It extracts structured JSON events for future scheduled items and writes
them to facet-based JSONL files.
"""

import json
import logging
from pathlib import Path

from think.facets import facet_summaries
from think.hooks import (
    compute_output_source,
    should_skip_extraction,
    write_events_jsonl,
)
from think.models import generate
from think.utils import get_output_topic, load_prompt


def process(result: str, context: dict) -> str | None:
    """Extract anticipation events from generator output result.

    This hook extracts structured JSON events for future scheduled items
    from markdown output summaries and writes them to facet-based JSONL files.

    Args:
        result: The generated output markdown content.
        context: Hook context with keys:
            - day: YYYYMMDD string
            - segment: segment key or None
            - name: generator name, e.g., "schedule"
            - output_path: absolute path to output file
            - meta: dict with frontmatter
            - transcript: the clustered transcript markdown
            - multi_segment: True if processing multiple segments

    Returns:
        None - this hook does not modify the output result.
    """
    # Check skip conditions
    skip_reason = should_skip_extraction(result, context)
    if skip_reason:
        logging.info("Skipping anticipation extraction: %s", skip_reason)
        return None

    # Load extraction prompt
    prompt_content = load_prompt("anticipation", base_dir=Path(__file__).parent)

    # Build context with facets (anticipations don't have topic-specific instructions)
    facets_context = facet_summaries(detailed_entities=True)

    # Extract events
    name = context.get("name", "unknown")
    contents = [facets_context, result]

    try:
        response_text = generate(
            contents=contents,
            context=f"agent.{name}.extraction",
            temperature=0.3,
            max_output_tokens=8192 * 6,
            thinking_budget=8192 * 3,
            system_instruction=prompt_content.text,
            json_output=True,
        )
    except Exception as e:
        logging.error("Extraction generation failed: %s", e)
        return None

    try:
        events = json.loads(response_text)
    except json.JSONDecodeError as e:
        logging.error("Invalid JSON from extraction: %s", e)
        return None

    if not isinstance(events, list):
        logging.error("Extraction did not return array")
        return None

    # Write to facet JSONL files (occurred=False for anticipations)
    source_output = compute_output_source(context)
    topic = get_output_topic(name)
    day = context.get("day", "")

    written_paths = write_events_jsonl(
        events=events,
        topic=topic,
        occurred=False,
        source_output=source_output,
        capture_day=day,
    )

    if written_paths:
        print(f"Events written to {len(written_paths)} JSONL file(s):")
        for p in written_paths:
            print(f"  {p}")
    else:
        print("No events with valid facets to write")

    return None  # Don't modify insight result
