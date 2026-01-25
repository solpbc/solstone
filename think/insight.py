# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types

from think.cluster import cluster, cluster_period, cluster_segments_multi
from think.models import generate
from think.utils import (
    PromptNotFoundError,
    _load_insight_metadata,
    compose_instructions,
    day_log,
    day_path,
    format_day,
    format_segment_times,
    get_insight_topic,
    get_insights,
    load_insight_hook,
    load_prompt,
    segment_parse,
    setup_cli,
)


def _output_path(
    day_dir: os.PathLike[str],
    key: str,
    segment: str | None = None,
    output_format: str | None = None,
) -> Path:
    """Return output path for insight ``key`` in ``day_dir``.

    Args:
        day_dir: Day directory path (YYYYMMDD)
        key: Insight key (e.g., "activity" or "chat:sentiment")
        segment: Optional segment key (HHMMSS_LEN)
        output_format: Output format from insight metadata ("json" or None for markdown)

    Returns:
        Path to output file:
        - Daily: YYYYMMDD/insights/{topic}.{ext}
        - Segment: YYYYMMDD/{segment}/{topic}.{ext}
        Where ext is "json" if output_format=="json", else "md"
    """
    day = Path(day_dir)
    topic = get_insight_topic(key)
    ext = "json" if output_format == "json" else "md"

    if segment:
        # Segment insights go directly in segment directory
        return day / segment / f"{topic}.{ext}"
    else:
        # Daily insights go in insights/ subdirectory
        return day / "insights" / f"{topic}.{ext}"


def scan_day(day: str) -> dict[str, list[str]]:
    """Return lists of processed and pending daily insight output files.

    Only scans daily insights (frequency='daily'). Segment insights are
    stored within segment directories and are not included here.
    """
    from think.utils import get_insights_by_frequency

    day_dir = day_path(day)
    daily_insights = get_insights_by_frequency("daily", include_disabled=True)
    processed: list[str] = []
    pending: list[str] = []
    for key, meta in sorted(daily_insights.items()):
        output_format = meta.get("output")
        output_path = _output_path(day_dir, key, output_format=output_format)
        if output_path.exists():
            processed.append(os.path.join("insights", output_path.name))
        else:
            pending.append(os.path.join("insights", output_path.name))
    return {"processed": sorted(processed), "repairable": sorted(pending)}


def count_tokens(markdown: str, prompt: str, api_key: str, model: str) -> None:
    client = genai.Client(api_key=api_key)

    total_tokens = client.models.count_tokens(
        model=model,
        contents=[markdown],
    )
    print(f"Token count: {total_tokens}")


def _get_or_create_cache(
    client: genai.Client,
    model: str,
    display_name: str,
    transcript: str,
    system_instruction: str,
) -> str | None:
    """Return cache name for ``display_name`` or None if content too small.

    Creates cache with ``transcript`` and provided system instruction if needed.
    Returns None if content is below estimated 2048 token minimum (~10k chars).

    The cache contains the system instruction + transcript which are identical
    for all topics on the same day with the same system prompt, so display_name
    should include both day and system prompt name."""

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
            system_instruction=system_instruction,
            contents=[transcript],
            ttl="1800s",  # 30 minutes to accommodate multiple topic analyses
        ),
    )
    return cache.name


def send_insight(
    transcript: str,
    prompt: str,
    api_key: str,
    cache_display_name: str | None = None,
    insight_key: str | None = None,
    json_output: bool = False,
    system_instruction: str | None = None,
    thinking_budget: int | None = None,
    max_output_tokens: int | None = None,
) -> str:
    """Send clustered transcript to LLM for insight generation.

    Args:
        transcript: Clustered transcript content (markdown format).
        prompt: Insight prompt text.
        api_key: Google API key for caching.
        cache_display_name: Optional cache key for Google content caching.
            Should include system prompt name for proper cache isolation.
        insight_key: Insight identifier for token logging context.
        json_output: If True, request JSON response format.
        system_instruction: System instruction text. If None, loads default
            from journal.md via compose_instructions().
        thinking_budget: Token budget for model thinking. If None, uses default.
        max_output_tokens: Maximum output tokens. If None, uses default.

    Returns:
        Generated insight content (markdown or JSON string).
    """
    # Use provided system_instruction or fall back to default
    if system_instruction is None:
        instructions = compose_instructions(include_datetime=False)
        system_instruction = instructions["system_instruction"]

    # Use defaults if not specified
    if thinking_budget is None:
        thinking_budget = 8192 * 3
    if max_output_tokens is None:
        max_output_tokens = 8192 * 6

    # Build context for provider routing and token logging
    output_type = "json" if json_output else "markdown"
    context = (
        f"insight.{insight_key}.{output_type}" if insight_key else "insight.unknown"
    )

    # Try to use cache if display name provided
    # Note: caching is Google-specific, so we check provider first
    from think.models import resolve_provider

    provider, model = resolve_provider(context)

    client = None
    cache_name = None
    if cache_display_name and provider == "google":
        client = genai.Client(api_key=api_key)
        cache_name = _get_or_create_cache(
            client, model, cache_display_name, transcript, system_instruction
        )

    if cache_name:
        # Cache hit: content already in cache, just send prompt.
        # Google-specific params (cached_content, client) are passed via kwargs.
        return generate(
            contents=[prompt],
            context=context,
            temperature=0.3,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
            model=model,
            cached_content=cache_name,
            client=client,
            json_output=json_output,
        )
    else:
        # No cache: use unified generate()
        return generate(
            contents=[transcript, prompt],
            context=context,
            temperature=0.3,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
            system_instruction=system_instruction,
            json_output=json_output,
        )


# Minimum content length for insight generation
MIN_INPUT_CHARS = 50


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
        help="Insight key (e.g., 'activity', 'chat:sentiment') or path to .md file",
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
    parser.add_argument(
        "--segments",
        help="Comma-separated segment keys (e.g., '090000_300,100000_600'). Requires -o.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (overrides default; required with --segments)",
    )
    args = setup_cli(parser)

    # Validate mutual exclusivity of --segment and --segments
    if args.segment and args.segments:
        parser.error("--segment and --segments are mutually exclusive")

    # Validate -o is required with --segments
    if args.segments and not args.output:
        parser.error("--segments requires -o/--output to specify output file path")

    # Set segment key for token usage logging
    if args.segment:
        os.environ["SEGMENT_KEY"] = args.segment
    elif args.segments:
        # Use first segment for logging context
        first_segment = args.segments.split(",")[0].strip()
        os.environ["SEGMENT_KEY"] = first_segment

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
        found_in_registry = False
        for key, meta in all_insights.items():
            if meta.get("path") == str(insight_path):
                insight_key = key
                found_in_registry = True
                break
        if found_in_registry:
            insight_meta = all_insights[insight_key]
        else:
            # Load metadata directly from file for ad-hoc insights
            insight_meta = _load_insight_metadata(insight_path)
    else:
        parser.error(
            f"Insight not found: {topic_arg}. "
            f"Available: {', '.join(sorted(all_insights.keys()))}"
        )

    # Check if insight is disabled via journal config
    if insight_meta.get("disabled"):
        logging.info("Insight %s is disabled in journal config, skipping", insight_key)
        day_log(args.day, f"insight {get_insight_topic(topic_arg)} skipped (disabled)")
        return

    output_format = insight_meta.get("output")  # "json" or None (markdown)
    success = False

    # Extract instructions config for source filtering and system prompt
    instructions_config = insight_meta.get("instructions")

    # Use compose_instructions to get sources config and system instruction
    instructions = compose_instructions(
        include_datetime=False,
        config_overrides=instructions_config,
    )
    sources = instructions.get("sources")
    system_prompt_name = instructions.get("system_prompt_name", "journal")
    system_instruction = instructions["system_instruction"]

    # Track multi-segment mode for hook context
    multi_segment_mode = bool(args.segments)

    # Choose clustering function based on mode, passing sources config
    if args.segments:
        segment_list = [s.strip() for s in args.segments.split(",")]
        try:
            markdown, file_count = cluster_segments_multi(
                args.day, segment_list, sources=sources
            )
        except ValueError as e:
            parser.error(str(e))
    elif args.segment:
        markdown, file_count = cluster_period(args.day, args.segment, sources=sources)
    else:
        markdown, file_count = cluster(args.day, sources=sources)
    day_dir = str(day_path(args.day))

    # Skip insight generation when there's nothing to analyze
    if file_count == 0 or len(markdown.strip()) < MIN_INPUT_CHARS:
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

        # Build context for template substitution
        prompt_context: dict[str, str] = {
            "day": args.day,
            "date": format_day(args.day),
        }

        # Add segment context
        if args.segment:
            # Single segment mode
            start_str, end_str = format_segment_times(args.segment)
            if start_str and end_str:
                prompt_context["segment"] = args.segment
                prompt_context["segment_start"] = start_str
                prompt_context["segment_end"] = end_str
        elif args.segments:
            # Multi-segment mode: compute earliest start and latest end
            segment_list = [s.strip() for s in args.segments.split(",")]
            all_times = []
            for seg in segment_list:
                start_time, end_time = segment_parse(seg)
                if start_time and end_time:
                    all_times.append((start_time, end_time))

            if all_times:
                earliest_start = min(t[0] for t in all_times)
                latest_end = max(t[1] for t in all_times)
                # Use lstrip('0') for cross-platform compatibility (%-I is Unix-only)
                start_str = (
                    datetime.combine(datetime.today(), earliest_start)
                    .strftime("%I:%M %p")
                    .lstrip("0")
                )
                end_str = (
                    datetime.combine(datetime.today(), latest_end)
                    .strftime("%I:%M %p")
                    .lstrip("0")
                )
                prompt_context["segment_start"] = start_str
                prompt_context["segment_end"] = end_str

        try:
            insight_prompt = load_prompt(
                insight_path.stem, base_dir=insight_path.parent, context=prompt_context
            )
        except PromptNotFoundError:
            parser.error(f"Insight file not found: {insight_path}")

        prompt = insight_prompt.text

        # Resolve provider for display (must match context used in send_insight)
        from think.models import resolve_provider

        display_output_type = "json" if output_format == "json" else "markdown"
        _, model = resolve_provider(f"insight.{insight_key}.{display_output_type}")
        day = args.day
        size_kb = len(markdown.encode("utf-8")) / 1024

        print(
            f"Topic: {insight_key} | Model: {model} | Day: {day} | Files: {file_count} | Size: {size_kb:.1f}KB"
        )

        if args.count:
            count_tokens(markdown, prompt, api_key, model)
            return

        is_json_output = output_format == "json"

        # Determine output path: -o overrides default for any mode
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = _output_path(
                day_dir, insight_key, segment=args.segment, output_format=output_format
            )

        # Determine cache settings: skip for multi-segment, otherwise scope to day/segment
        # Include system prompt name in cache key for proper isolation
        if multi_segment_mode:
            cache_display_name = None
        elif args.segment:
            cache_display_name = f"{system_prompt_name}_{day}_{args.segment}"
        else:
            cache_display_name = f"{system_prompt_name}_{day}"

        # Check if output file already exists
        output_exists = output_path.exists() and output_path.stat().st_size > 0

        # Extract optional generation parameters from insight metadata
        meta_thinking_budget = insight_meta.get("thinking_budget")
        meta_max_output_tokens = insight_meta.get("max_output_tokens")

        if output_exists and not args.force:
            print(
                f"Output file already exists: {output_path}. Loading existing content."
            )
            with open(output_path, "r") as f:
                result = f.read()
        elif output_exists and args.force:
            print("Output file exists but --force specified. Regenerating.")
            result = send_insight(
                markdown,
                prompt,
                api_key,
                cache_display_name=cache_display_name,
                insight_key=insight_key,
                json_output=is_json_output,
                system_instruction=system_instruction,
                thinking_budget=meta_thinking_budget,
                max_output_tokens=meta_max_output_tokens,
            )
        else:
            result = send_insight(
                markdown,
                prompt,
                api_key,
                cache_display_name=cache_display_name,
                insight_key=insight_key,
                json_output=is_json_output,
                system_instruction=system_instruction,
                thinking_budget=meta_thinking_budget,
                max_output_tokens=meta_max_output_tokens,
            )

        # Check if we got a valid response
        if result is None:
            print("Error: No text content in response")
            return

        # Run post-processing hook if present (only for newly generated results)
        if (not output_exists or args.force) and insight_meta.get("hook_path"):
            hook_path = insight_meta["hook_path"]
            try:
                hook_process = load_insight_hook(hook_path)
                hook_context = {
                    "day": args.day,
                    "segment": args.segment,
                    "multi_segment": multi_segment_mode,
                    "insight_key": insight_key,
                    "output_path": str(output_path),
                    "insight_meta": dict(insight_meta),
                    "transcript": markdown,
                }
                hook_result = hook_process(result, hook_context)
                if hook_result is not None:
                    result = hook_result
                    logging.info("Hook %s transformed result", hook_path)
                else:
                    logging.info(
                        "Hook %s returned None, using original result", hook_path
                    )
            except Exception as exc:
                logging.error("Hook %s failed: %s", hook_path, exc)
                # Continue with original result on hook failure

        # Only write output if it was newly generated
        if not output_exists or args.force:
            os.makedirs(output_path.parent, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(result)
            print(f"Results saved to: {output_path}")

        success = True

    finally:
        msg = f"insight {insight_key} {'ok' if success else 'failed'}"
        if args.force:
            msg += " --force"
        day_log(args.day, msg)


if __name__ == "__main__":
    main()
