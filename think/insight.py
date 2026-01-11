# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import argparse
import json
import logging
import os
from pathlib import Path

from google import genai
from google.genai import types

from think.cluster import cluster, cluster_period
from think.models import GEMINI_PRO, gemini_generate, resolve_provider
from think.utils import (
    PromptNotFoundError,
    day_log,
    day_path,
    get_insight_topic,
    get_insights,
    get_journal,
    load_prompt,
    setup_cli,
)

COMMON_SYSTEM_INSTRUCTION = "You are an expert productivity analyst tasked with analyzing a full workday transcript containing both audio conversations and screen activity data, organized into recording segments. You will be given the transcripts and then following that you will have a detailed user request for how to process them.  Please follow those instructions carefully. Take time to consider all of the nuance of the interactions from the day, deeply think through how best to prioritize the most important aspects and understandings, formulate the best approach for each step of the analysis."


def _write_events_jsonl(
    events: list[dict],
    topic: str,
    occurred: bool,
    source_insight: str,
    capture_day: str,
) -> list[Path]:
    """Write events to facet-based JSONL files.

    Groups events by facet and writes each to the appropriate file:
    facets/{facet}/events/{event_day}.jsonl

    Args:
        events: List of event dictionaries from extraction.
        topic: Source insight topic (e.g., "meetings", "schedule").
        occurred: True for occurrences, False for anticipations.
        source_insight: Relative path to source insight file.
        capture_day: Day the insight was captured (YYYYMMDD).

    Returns:
        List of paths to written JSONL files.
    """
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
        enriched["source"] = source_insight

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


def _insight_keys() -> list[str]:
    """Return available insight keys."""
    return sorted(get_insights().keys())


def _output_path(
    day_dir: os.PathLike[str], key: str, segment: str | None = None
) -> Path:
    """Return markdown output path for insight ``key`` in ``day_dir``.

    Args:
        day_dir: Day directory path (YYYYMMDD)
        key: Insight key (e.g., "activity" or "chat:sentiment")
        segment: Optional segment key (HHMMSS_LEN)

    Returns:
        Path to markdown file:
        - Daily: YYYYMMDD/insights/{topic}.md (where topic = get_insight_topic(key))
        - Segment: YYYYMMDD/{segment}/{topic}.md
    """
    day = Path(day_dir)
    topic = get_insight_topic(key)

    if segment:
        # Segment insights go directly in segment directory
        return day / segment / f"{topic}.md"
    else:
        # Daily insights go in insights/ subdirectory
        return day / "insights" / f"{topic}.md"


def scan_day(day: str) -> dict[str, list[str]]:
    """Return lists of processed and pending insight markdown files."""
    day_dir = day_path(day)
    processed: list[str] = []
    pending: list[str] = []
    for key in _insight_keys():
        md_path = _output_path(day_dir, key)
        if md_path.exists():
            processed.append(os.path.join("insights", md_path.name))
        else:
            pending.append(os.path.join("insights", md_path.name))
    return {"processed": sorted(processed), "repairable": sorted(pending)}


def count_tokens(markdown: str, prompt: str, api_key: str, model: str) -> None:
    client = genai.Client(api_key=api_key)

    total_tokens = client.models.count_tokens(
        model=model,
        contents=[markdown],
    )
    print(f"Token count: {total_tokens}")


def _get_or_create_cache(
    client: genai.Client, model: str, display_name: str, transcript: str
) -> str | None:
    """Return cache name for ``display_name`` or None if content too small.

    Creates cache with ``transcript`` and :data:`COMMON_SYSTEM_INSTRUCTION` if needed.
    Returns None if content is below estimated 2048 token minimum (~10k chars).

    The cache contains the system instruction + transcript which are identical
    for all topics on the same day, so display_name should be day-based only."""

    MIN_CACHE_CHARS = 10000  # Heuristic: ~4 chars/token → 2048 tokens ≈ 8k-10k chars

    # Check existing caches first
    for c in client.caches.list():
        if c.model == model and c.display_name == display_name:
            return c.name

    # Skip cache creation for small content
    if len(transcript) < MIN_CACHE_CHARS:
        return None

    cache = client.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(
            display_name=display_name,
            system_instruction=COMMON_SYSTEM_INSTRUCTION,
            contents=[transcript],
            ttl="1800s",  # 30 minutes to accommodate multiple topic analyses
        ),
    )
    return cache.name


def send_markdown(
    markdown: str,
    prompt: str,
    api_key: str,
    model: str,
    cache_display_name: str | None = None,
    insight_key: str | None = None,
) -> str:
    # Build context for token logging
    context = f"insight.{insight_key}.markdown" if insight_key else None

    # Try to use cache if display name provided
    client = None
    cache_name = None
    if cache_display_name:
        client = genai.Client(api_key=api_key)
        cache_name = _get_or_create_cache(client, model, cache_display_name, markdown)

    if cache_name:
        # Cache available: content already in cache, just send prompt
        return gemini_generate(
            contents=[prompt],
            model=model,
            temperature=0.3,
            max_output_tokens=8192 * 6,
            thinking_budget=8192 * 3,
            cached_content=cache_name,
            client=client,
            context=context,
        )
    else:
        # No cache: send markdown + prompt with system instruction
        return gemini_generate(
            contents=[markdown, prompt],
            model=model,
            temperature=0.3,
            max_output_tokens=8192 * 6,
            thinking_budget=8192 * 3,
            system_instruction=COMMON_SYSTEM_INSTRUCTION,
            context=context,
        )


# Minimum content length for meaningful event extraction
MIN_EXTRACTION_CHARS = 50


def _should_skip_extraction(result: str) -> bool:
    """Check if result is too minimal for meaningful event extraction.

    When insight generation returns very short content (e.g., "No meetings detected"),
    there's nothing substantive to extract. Skipping extraction in these cases
    avoids unnecessary API calls and prevents hallucination from entity context.

    Args:
        result: The markdown result from send_markdown().

    Returns:
        True if extraction should be skipped, False otherwise.
    """
    return len(result.strip()) < MIN_EXTRACTION_CHARS


def send_extraction(
    markdown: str,
    prompt: str,
    model: str,
    extra_instructions: str | None = None,
    insight_key: str | None = None,
) -> list:
    """Extract structured JSON events from markdown summary.

    Used for both occurrences (past events) and anticipations (future events).

    Parameters
    ----------
    markdown:
        Markdown summary to extract events from.
    prompt:
        System instruction guiding the extraction.
    model:
        Gemini model name.
    extra_instructions:
        Optional additional instructions prepended to ``markdown``.
    insight_key:
        Insight key for token usage context (e.g., "decisions").

    Returns
    -------
    list
        Array of extracted event objects.
    """
    # Build context for token logging
    context = f"insight.{insight_key}.extraction" if insight_key else None

    contents = [markdown]
    if extra_instructions:
        contents.insert(0, extra_instructions)

    response_text = gemini_generate(
        contents=contents,
        model=model,
        temperature=0.3,
        max_output_tokens=8192 * 6,
        thinking_budget=8192 * 3,
        system_instruction=prompt,
        json_output=True,
        context=context,
    )

    try:
        events = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}: {response_text[:100]}")

    if not isinstance(events, list):
        raise ValueError(f"Response is not an array: {response_text[:100]}")

    return events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a day's clustered Markdown to Gemini for analysis."
    )
    parser.add_argument(
        "day",
        help="Day in YYYYMMDD format",
    )
    parser.add_argument(
        "-f",
        "--topic",
        "--prompt",
        dest="topic",
        required=True,
        help="Insight key (e.g., 'activity', 'chat:sentiment') or path to .txt file",
    )
    parser.add_argument(
        "-p",
        "--pro",
        action="store_true",
        help="Use the gemini 2.5 pro model",
    )
    parser.add_argument(
        "-c",
        "--count",
        action="store_true",
        help="Count tokens only and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    parser.add_argument(
        "--segment",
        help="Segment key in HHMMSS_LEN format (processes only this segment within the day)",
    )
    args = setup_cli(parser)

    # Set segment key for token usage logging
    if args.segment:
        os.environ["SEGMENT_KEY"] = args.segment

    # Resolve insight key or path to metadata
    all_insights = get_insights()
    topic_arg = args.topic

    # Check if it's a known insight key first
    if topic_arg in all_insights:
        insight_key = topic_arg
        insight_meta = all_insights[insight_key]
        insight_path = Path(insight_meta["path"])
    elif Path(topic_arg).exists():
        # Fall back to treating it as a file path (backwards compat)
        insight_path = Path(topic_arg)
        # Try to find matching key by path
        insight_key = insight_path.stem
        for key, meta in all_insights.items():
            if meta.get("path") == str(insight_path):
                insight_key = key
                break
        insight_meta = all_insights.get(insight_key, {})
    else:
        parser.error(
            f"Insight not found: {topic_arg}. "
            f"Available: {', '.join(sorted(all_insights.keys()))}"
        )

    extra_occ = insight_meta.get("occurrences")
    skip_occ = extra_occ is False
    do_anticipations = insight_meta.get("anticipations") is True
    success = False

    # Choose clustering function based on mode
    if args.segment:
        markdown, file_count = cluster_period(args.day, args.segment)
    else:
        markdown, file_count = cluster(args.day)
    day_dir = str(day_path(args.day))

    # Skip insight generation when there's nothing to analyze
    if file_count == 0 or len(markdown.strip()) < MIN_EXTRACTION_CHARS:
        logging.info(
            "Insufficient input (files=%d, chars=%d), skipping insight generation",
            file_count,
            len(markdown.strip()),
        )
        day_log(args.day, f"insight {get_insight_topic(topic_arg)} skipped (no input)")
        return

    # Prepend input context note for limited recordings
    if file_count < 3:
        input_note = (
            "**Input Note:** Limited recordings for this day. "
            "Scale analysis to available input.\n\n"
        )
        markdown = input_note + markdown

    try:
        if args.verbose:
            print("Verbose mode enabled")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            parser.error("GOOGLE_API_KEY not found in environment")

        try:
            insight_prompt = load_prompt(
                insight_path.stem, base_dir=insight_path.parent, include_journal=True
            )
        except PromptNotFoundError:
            parser.error(f"Insight file not found: {insight_path}")

        prompt = insight_prompt.text

        _, default_model = resolve_provider(f"insight.{insight_key}")
        model = GEMINI_PRO if args.pro else default_model
        day = args.day
        size_kb = len(markdown.encode("utf-8")) / 1024

        print(
            f"Topic: {insight_key} | Model: {model} | Day: {day} | Files: {file_count} | Size: {size_kb:.1f}KB"
        )

        if args.count:
            count_tokens(markdown, prompt, api_key, model)
            return

        md_path = _output_path(day_dir, insight_key, segment=args.segment)
        # Use cache key scoped to day or segment
        if args.segment:
            cache_display_name = f"{day}_{args.segment}"
        else:
            cache_display_name = f"{day}"

        # Check if markdown file already exists
        md_exists = md_path.exists() and md_path.stat().st_size > 0

        if md_exists and not args.force:
            print(f"Markdown file already exists: {md_path}. Loading existing content.")
            with open(md_path, "r") as f:
                result = f.read()
        elif md_exists and args.force:
            print("Markdown file exists but --force specified. Regenerating.")
            result = send_markdown(
                markdown,
                prompt,
                api_key,
                model,
                cache_display_name=cache_display_name,
                insight_key=insight_key,
            )
        else:
            result = send_markdown(
                markdown,
                prompt,
                api_key,
                model,
                cache_display_name=cache_display_name,
                insight_key=insight_key,
            )

        # Check if we got a valid response
        if result is None:
            print("Error: No text content in response")
            return

        # Only write markdown if it was newly generated
        if not md_exists or args.force:
            os.makedirs(md_path.parent, exist_ok=True)
            with open(md_path, "w") as f:
                f.write(result)
            print(f"Results saved to: {md_path}")

        if skip_occ and not do_anticipations:
            print('"occurrences" set to false; skipping event extraction')
            success = True
            return

        # Skip extraction for minimal content (prevents hallucination from entity context)
        if _should_skip_extraction(result):
            logging.info(
                "Minimal content (%d chars < %d), skipping event extraction",
                len(result.strip()),
                MIN_EXTRACTION_CHARS,
            )
            success = True
            return

        # Determine which prompt to use: anticipations or occurrences
        prompt_name = "anticipation" if do_anticipations else "occurrence"

        # Load the appropriate extraction prompt
        try:
            extraction_prompt_content = load_prompt(
                prompt_name, base_dir=Path(__file__).parent, include_journal=True
            )
        except PromptNotFoundError as exc:
            print(exc)
            return

        extraction_prompt = extraction_prompt_content.text

        try:
            # Load facet summaries and combine with topic-specific instructions
            from think.facets import facet_summaries

            facets_context = facet_summaries(detailed_entities=True)

            # Combine facet summaries with topic-specific instructions
            if extra_occ and not do_anticipations:
                combined_instructions = f"{facets_context}\n\n{extra_occ}"
            else:
                combined_instructions = facets_context

            events = send_extraction(
                result,
                extraction_prompt,
                model,
                extra_instructions=combined_instructions,
                insight_key=insight_key,
            )
        except ValueError as e:
            print(f"Error: {e}")
            return

        # Write to new JSONL format (facets/{facet}/events/{day}.jsonl)
        occurred = not do_anticipations
        insight_topic = get_insight_topic(insight_key)

        # Compute the relative source insight path
        # md_path is absolute, day_dir is the YYYYMMDD directory path
        # source_insight should be like "20240101/insights/meetings.md"
        journal = get_journal()
        try:
            source_insight = os.path.relpath(str(md_path), journal)
        except ValueError:
            # Fallback: construct from day and topic
            source_insight = os.path.join(
                day,
                "insights" if not args.segment else args.segment,
                f"{insight_topic}.md",
            )
        written_paths = _write_events_jsonl(
            events=events,
            topic=insight_topic,
            occurred=occurred,
            source_insight=source_insight,
            capture_day=day,
        )

        if written_paths:
            print(f"Events written to {len(written_paths)} JSONL file(s):")
            for p in written_paths:
                print(f"  {p}")
        else:
            print("No events with valid facets to write")

        success = True

    finally:
        msg = f"insight {insight_key} {'ok' if success else 'failed'}"
        if args.force:
            msg += " --force"
        day_log(args.day, msg)


if __name__ == "__main__":
    main()
