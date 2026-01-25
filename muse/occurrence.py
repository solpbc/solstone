# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Hook for extracting occurrence events from insight results.

This hook is invoked via "hook": "occurrence" in insight frontmatter.
It extracts structured JSON events from markdown summaries and writes
them to facet-based JSONL files.
"""

import json
import logging
from pathlib import Path

from think.facets import facet_summaries
from think.insights import (
    compute_source_insight,
    should_skip_extraction,
    write_events_jsonl,
)
from think.models import generate
from think.utils import get_insight_topic, load_prompt


def process(result: str, context: dict) -> str | None:
    """Extract occurrence events from insight result.

    This hook extracts structured JSON events from markdown insight summaries
    and writes them to facet-based JSONL files.

    Args:
        result: The generated insight markdown content.
        context: Hook context with keys:
            - day: YYYYMMDD string
            - segment: segment key or None
            - insight_key: e.g., "meetings", "flow"
            - output_path: absolute path to output file
            - insight_meta: dict with frontmatter including "occurrences"
            - transcript: the clustered transcript markdown
            - multi_segment: True if processing multiple segments

    Returns:
        None - this hook does not modify the insight result.
    """
    # Check skip conditions
    skip_reason = should_skip_extraction(result, context)
    if skip_reason:
        logging.info("Skipping occurrence extraction: %s", skip_reason)
        return None

    # Load extraction prompt
    prompt_content = load_prompt("occurrence", base_dir=Path(__file__).parent)

    # Build context with facets + topic-specific instructions
    facets_context = facet_summaries(detailed_entities=True)
    topic_instructions = context.get("insight_meta", {}).get("occurrences")
    if topic_instructions and isinstance(topic_instructions, str):
        extra_instructions = f"{facets_context}\n\n{topic_instructions}"
    else:
        extra_instructions = facets_context

    # Extract events
    insight_key = context.get("insight_key", "unknown")
    contents = [extra_instructions, result]

    try:
        response_text = generate(
            contents=contents,
            context=f"insight.{insight_key}.extraction",
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

    # Write to facet JSONL files
    source_insight = compute_source_insight(context)
    topic = get_insight_topic(insight_key)
    day = context.get("day", "")

    written_paths = write_events_jsonl(
        events=events,
        topic=topic,
        occurred=True,
        source_insight=source_insight,
        capture_day=day,
    )

    if written_paths:
        print(f"Events written to {len(written_paths)} JSONL file(s):")
        for p in written_paths:
            print(f"  {p}")
    else:
        print("No events with valid facets to write")

    return None  # Don't modify insight result
