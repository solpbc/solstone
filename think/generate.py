# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Generator pipeline for transcript analysis.

Spawned by cortex when a request has 'output' field (no 'tools').
Reads NDJSON config from stdin, emits JSONL events to stdout.
"""

import json
import logging
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types

from think.agents import GenerateResult, JSONEventWriter
from think.cluster import cluster, cluster_period, cluster_segments_multi
from think.utils import (
    compose_instructions,
    day_log,
    day_path,
    format_day,
    format_segment_times,
    get_generator_agents,
    get_output_path,
    load_output_hook,
    load_prompt,
    segment_parse,
)


def scan_day(day: str) -> dict[str, list[str]]:
    """Return lists of processed and pending daily generator output files.

    Only scans daily generators (schedule='daily'). Segment generators are
    stored within segment directories and are not included here.
    """
    from think.utils import get_generator_agents_by_schedule

    day_dir = day_path(day)
    daily_generators = get_generator_agents_by_schedule("daily", include_disabled=True)
    processed: list[str] = []
    pending: list[str] = []
    for key, meta in sorted(daily_generators.items()):
        output_format = meta.get("output")
        output_path = get_output_path(day_dir, key, output_format=output_format)
        if output_path.exists():
            processed.append(os.path.join("agents", output_path.name))
        else:
            pending.append(os.path.join("agents", output_path.name))
    return {"processed": sorted(processed), "repairable": sorted(pending)}


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


def generate_agent_output(
    transcript: str,
    prompt: str,
    api_key: str,
    cache_display_name: str | None = None,
    name: str | None = None,
    json_output: bool = False,
    system_instruction: str | None = None,
    thinking_budget: int | None = None,
    max_output_tokens: int | None = None,
    return_result: bool = False,
) -> str | GenerateResult:
    """Send clustered transcript to LLM for agent output generation.

    Args:
        transcript: Clustered transcript content (markdown format).
        prompt: Agent prompt text.
        api_key: Google API key for caching.
        cache_display_name: Optional cache key for Google content caching.
            Should include system prompt name for proper cache isolation.
        name: Agent name for token logging context.
        json_output: If True, request JSON response format.
        system_instruction: System instruction text. If None, loads default
            from journal.md via compose_instructions().
        thinking_budget: Token budget for model thinking. If None, uses default.
        max_output_tokens: Maximum output tokens. If None, uses default.
        return_result: If True, return full GenerateResult with usage data.

    Returns:
        Generated agent output content (markdown or JSON string), or
        GenerateResult dict if return_result=True.
    """
    from think.models import generate_with_result, resolve_provider

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
    context = f"agent.{name}.{output_type}" if name else "agent.unknown"

    # Try to use cache if display name provided
    # Note: caching is Google-specific, so we check provider first
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
        result = generate_with_result(
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
        result = generate_with_result(
            contents=[transcript, prompt],
            context=context,
            temperature=0.3,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
            system_instruction=system_instruction,
            json_output=json_output,
        )

    if return_result:
        return result
    return result["text"]


# Minimum content length for insight generation
MIN_INPUT_CHARS = 50


def _run_generator(config: dict, emit_event: Callable[[dict], None]) -> None:
    """Execute generator pipeline with config from cortex.

    Args:
        config: Merged config from cortex containing:
            - name: Generator key (e.g., 'activity', 'chat:sentiment')
            - day: Day in YYYYMMDD format
            - segment: Optional single segment key
            - segments: Optional list of segment keys
            - output: Output format ('md' or 'json')
            - output_path: Optional custom output path
            - force: Whether to regenerate existing output
            - provider: AI provider
            - model: Model name
        emit_event: Callback to emit JSONL events
    """
    name = config.get("name", "default")
    day = config.get("day")
    segment = config.get("segment")
    segments = config.get("segments")  # List of segment keys
    output_format = config.get("output", "md")
    output_path_override = config.get("output_path")
    force = config.get("force", False)
    provider = config.get("provider", "google")
    model = config.get("model")

    if not day:
        raise ValueError("Missing 'day' field in generator config")

    # Emit start event
    emit_event(
        {
            "event": "start",
            "ts": int(time.time() * 1000),
            "prompt": "",  # Generators don't have user prompts
            "name": name,
            "model": model or "unknown",
            "provider": provider,
        }
    )

    # Set segment key for token usage logging
    if segment:
        os.environ["SEGMENT_KEY"] = segment
    elif segments:
        os.environ["SEGMENT_KEY"] = segments[0]

    # Load generator metadata
    all_generators = get_generator_agents()
    if name in all_generators:
        meta = all_generators[name]
        agent_path = Path(meta["path"])
    else:
        raise ValueError(f"Generator not found: {name}")

    # Check if generator is disabled
    if meta.get("disabled"):
        logging.info("Generator %s is disabled, skipping", name)
        emit_event(
            {
                "event": "finish",
                "ts": int(time.time() * 1000),
                "result": "",
                "skipped": "disabled",
            }
        )
        return

    # Extract instructions config for source filtering and system prompt
    instructions_config = meta.get("instructions")
    instructions = compose_instructions(
        include_datetime=False,
        config_overrides=instructions_config,
    )
    sources = instructions.get("sources")
    system_prompt_name = instructions.get("system_prompt_name", "journal")
    system_instruction = instructions["system_instruction"]

    # Track multi-segment mode
    multi_segment_mode = bool(segments)

    # Build transcript via clustering
    if segments:
        markdown, file_count = cluster_segments_multi(day, segments, sources=sources)
    elif segment:
        markdown, file_count = cluster_period(day, segment, sources=sources)
    else:
        markdown, file_count = cluster(day, sources=sources)

    day_dir = str(day_path(day))

    # Skip generation when there's nothing to analyze
    if file_count == 0 or len(markdown.strip()) < MIN_INPUT_CHARS:
        logging.info(
            "Insufficient input (files=%d, chars=%d), skipping",
            file_count,
            len(markdown.strip()),
        )
        emit_event(
            {
                "event": "finish",
                "ts": int(time.time() * 1000),
                "result": "",
                "skipped": "no_input",
            }
        )
        day_log(day, f"generate {name} skipped (no input)")
        return

    # Prepend input context note for limited recordings
    if file_count < 3:
        input_note = (
            "**Input Note:** Limited recordings for this day. "
            "Scale analysis to available input.\n\n"
        )
        markdown = input_note + markdown

    # Build context for template substitution
    prompt_context: dict[str, str] = {
        "day": day,
        "date": format_day(day),
    }

    # Add segment context
    if segment:
        start_str, end_str = format_segment_times(segment)
        if start_str and end_str:
            prompt_context["segment"] = segment
            prompt_context["segment_start"] = start_str
            prompt_context["segment_end"] = end_str
    elif segments:
        all_times = []
        for seg in segments:
            start_time, end_time = segment_parse(seg)
            if start_time and end_time:
                all_times.append((start_time, end_time))

        if all_times:
            earliest_start = min(t[0] for t in all_times)
            latest_end = max(t[1] for t in all_times)
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

    # Load prompt
    agent_prompt = load_prompt(
        agent_path.stem, base_dir=agent_path.parent, context=prompt_context
    )
    prompt = agent_prompt.text

    # Determine output path
    is_json_output = output_format == "json"
    if output_path_override:
        output_path = Path(output_path_override)
    else:
        output_path = get_output_path(
            day_dir, name, segment=segment, output_format=output_format
        )

    # Check if output exists (force check happens in cortex, but we handle it here too)
    output_exists = output_path.exists() and output_path.stat().st_size > 0

    # Determine cache settings
    if multi_segment_mode:
        cache_display_name = None
    elif segment:
        cache_display_name = f"{system_prompt_name}_{day}_{segment}"
    else:
        cache_display_name = f"{system_prompt_name}_{day}"

    # Extract generation parameters from metadata
    meta_thinking_budget = meta.get("thinking_budget")
    meta_max_output_tokens = meta.get("max_output_tokens")

    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY", "")

    usage_data = None

    if output_exists and not force:
        # Load existing content (no LLM call)
        logging.info("Output exists, loading: %s", output_path)
        with open(output_path, "r") as f:
            result = f.read()
    else:
        # Generate new content
        if output_exists and force:
            logging.info("Force regenerating: %s", output_path)

        gen_result = generate_agent_output(
            markdown,
            prompt,
            api_key,
            cache_display_name=cache_display_name,
            name=name,
            json_output=is_json_output,
            system_instruction=system_instruction,
            thinking_budget=meta_thinking_budget,
            max_output_tokens=meta_max_output_tokens,
            return_result=True,
        )
        result = gen_result["text"]
        usage_data = gen_result.get("usage")

        # Run post-processing hook if present
        if meta.get("hook_path"):
            hook_path = meta["hook_path"]
            try:
                hook_process = load_output_hook(hook_path)
                hook_context = {
                    "day": day,
                    "segment": segment,
                    "multi_segment": multi_segment_mode,
                    "name": name,
                    "output_path": str(output_path),
                    "meta": dict(meta),
                    "transcript": markdown,
                }
                hook_result = hook_process(result, hook_context)
                if hook_result is not None:
                    result = hook_result
                    logging.info("Hook %s transformed result", hook_path)
            except Exception as exc:
                logging.error("Hook %s failed: %s", hook_path, exc)

    # Emit finish event with result (cortex handles file writing)
    finish_event = {
        "event": "finish",
        "ts": int(time.time() * 1000),
        "result": result,
    }
    if usage_data:
        finish_event["usage"] = usage_data

    emit_event(finish_event)

    # Log completion
    msg = f"generate {name} ok"
    if force:
        msg += " --force"
    day_log(day, msg)


def main() -> None:
    """NDJSON-based CLI for generator pipeline.

    Reads config from stdin, emits JSONL events to stdout.
    Spawned by cortex when request has 'output' field (no 'tools').
    """
    import traceback

    # Configure basic logging (no argparse needed for NDJSON mode)
    logging.basicConfig(level=logging.INFO)

    # Always write to stdout only
    event_writer = JSONEventWriter(None)

    def emit_event(data: dict) -> None:
        if "ts" not in data:
            data["ts"] = int(time.time() * 1000)
        event_writer.emit(data)

    try:
        # NDJSON input mode from stdin
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                config = json.loads(line)

                _run_generator(config, emit_event)

            except json.JSONDecodeError as e:
                emit_event(
                    {
                        "event": "error",
                        "error": f"Invalid JSON: {str(e)}",
                        "ts": int(time.time() * 1000),
                    }
                )
            except Exception as e:
                emit_event(
                    {
                        "event": "error",
                        "error": str(e),
                        "trace": traceback.format_exc(),
                        "ts": int(time.time() * 1000),
                    }
                )

    except Exception as exc:
        emit_event(
            {
                "event": "error",
                "error": str(exc),
                "trace": traceback.format_exc(),
            }
        )
        raise
    finally:
        event_writer.close()


if __name__ == "__main__":
    main()
