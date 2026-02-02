# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Hook for extracting occurrence events from generator output results.

This hook is invoked via "hook": {"post": "occurrence"} in generator frontmatter.
It extracts structured JSON events from markdown summaries and writes
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
from think.muse import get_output_topic
from think.prompts import load_prompt


def post_process(result: str, context: dict) -> str | None:
    """Extract occurrence events from generator output result.

    This hook extracts structured JSON events from markdown output summaries
    and writes them to facet-based JSONL files.

    Args:
        result: The generated output markdown content.
        context: Config dict with keys including day, segment, name,
            output_path, meta, transcript, span, span_mode.

    Returns:
        None - this hook does not modify the output result.
    """
    # Check skip conditions
    skip_reason = should_skip_extraction(result, context)
    if skip_reason:
        logging.info("Skipping occurrence extraction: %s", skip_reason)
        return None

    # Load extraction prompt
    prompt_content = load_prompt("occurrence", base_dir=Path(__file__).parent)

    # Build context with facets + topic-specific instructions
    facets_context = facet_summaries(detailed=True)
    topic_instructions = context.get("meta", {}).get("occurrences")
    if topic_instructions and isinstance(topic_instructions, str):
        extra_instructions = f"{facets_context}\n\n{topic_instructions}"
    else:
        extra_instructions = facets_context

    # Extract events
    name = context.get("name", "unknown")
    contents = [extra_instructions, result]

    try:
        response_text = generate(
            contents=contents,
            context=f"muse.system.{name}",
            temperature=0.3,
            max_output_tokens=16384,
            thinking_budget=0,
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

    # Write to facet JSONL files
    source_output = compute_output_source(context)
    topic = get_output_topic(name)
    day = context.get("day", "")

    written_paths = write_events_jsonl(
        events=events,
        topic=topic,
        occurred=True,
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
