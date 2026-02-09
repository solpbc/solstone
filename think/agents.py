# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unified agent CLI for solstone.

Spawned by cortex for all agent types:
- Tool-using agents (with configured tools)
- Generators (transcript analysis, no tools)

Both paths share unified config preparation and execution flow.
Reads NDJSON config from stdin, emits JSONL events to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from think.cluster import cluster, cluster_period, cluster_span
from think.muse import (
    get_agent_filter,
    get_muse_configs,
    get_output_path,
    load_post_hook,
    load_pre_hook,
    load_prompt,
    source_is_enabled,
    source_is_required,
)
from think.providers.shared import Event
from think.utils import (
    day_log,
    day_path,
    format_day,
    format_segment_times,
    now_ms,
    segment_parse,
    setup_cli,
)

LOG = logging.getLogger("think.agents")

# Minimum content length for transcript-based generation
MIN_INPUT_CHARS = 50


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for agent CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, stream=sys.stdout)
    return LOG


class JSONEventWriter:
    """Write JSONL events to stdout and optionally to a file."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.file = None
        if path:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                self.file = open(path, "a", encoding="utf-8")
            except Exception:
                pass  # Fail silently if can't open file

    def emit(self, data: Event) -> None:
        line = json.dumps(data, ensure_ascii=False)
        print(line)
        sys.stdout.flush()  # Ensure immediate output for cortex
        if self.file:
            try:
                self.file.write(line + "\n")
                self.file.flush()
            except Exception:
                pass  # Fail silently on write errors

    def close(self) -> None:
        if self.file:
            try:
                self.file.close()
            except Exception:
                pass


# =============================================================================
# Unified Config Preparation
# =============================================================================


def _build_prompt_context(
    day: str | None, segment: str | None, span: list[str] | None
) -> dict[str, str]:
    """Build context dict for prompt template substitution.

    Args:
        day: Day in YYYYMMDD format
        segment: Segment key (HHMMSS_LEN)
        span: List of segment keys

    Returns:
        Dict with template variables:
        - day: Friendly format (e.g., "Sunday, February 2, 2025")
        - day_YYYYMMDD: Raw day string (e.g., "20250202")
        - segment_start, segment_end: Time strings if segment/span provided
    """
    context: dict[str, str] = {}
    if not day:
        return context

    context["day"] = format_day(day)
    context["day_YYYYMMDD"] = day

    if segment:
        start_str, end_str = format_segment_times(segment)
        if start_str and end_str:
            context["segment"] = segment
            context["segment_start"] = start_str
            context["segment_end"] = end_str
    elif span:
        all_times = []
        for seg in span:
            start_time, end_time = segment_parse(seg)
            if start_time and end_time:
                all_times.append((start_time, end_time))

        if all_times:
            earliest_start = min(t[0] for t in all_times)
            latest_end = max(t[1] for t in all_times)
            context["segment_start"] = (
                datetime.combine(datetime.today(), earliest_start)
                .strftime("%I:%M %p")
                .lstrip("0")
            )
            context["segment_end"] = (
                datetime.combine(datetime.today(), latest_end)
                .strftime("%I:%M %p")
                .lstrip("0")
            )

    return context


def _load_transcript(
    day: str,
    segment: str | None,
    span: list[str] | None,
    sources: dict,
) -> tuple[str, dict[str, int]]:
    """Load and cluster transcript for day/segment/span.

    Args:
        day: Day in YYYYMMDD format
        segment: Optional segment key
        span: Optional list of segment keys
        sources: Source config dict from instructions

    Returns:
        Tuple of (transcript text, source_counts dict)
    """
    # Set segment key for token usage logging
    if segment:
        os.environ["SEGMENT_KEY"] = segment
    elif span:
        os.environ["SEGMENT_KEY"] = span[0]

    # Convert sources config for clustering
    cluster_sources: dict = {}
    for k, v in sources.items():
        if k == "agents":
            agent_filter = get_agent_filter(v)
            if agent_filter is None:
                cluster_sources[k] = source_is_enabled(v)
            elif not agent_filter:
                cluster_sources[k] = False
            else:
                cluster_sources[k] = agent_filter
        else:
            cluster_sources[k] = source_is_enabled(v)

    # Build transcript via clustering
    if span:
        return cluster_span(day, span, sources=cluster_sources)
    elif segment:
        return cluster_period(day, segment, sources=cluster_sources)
    else:
        return cluster(day, sources=cluster_sources)


def prepare_config(request: dict) -> dict:
    """Prepare complete agent config from request.

    Single unified preparation path for all agent types. Takes raw request
    from cortex and returns fully prepared config ready for execution.

    Config fields produced:
    - name: Agent name
    - provider, model: Resolved from context/request
    - system_instruction: System prompt
    - user_instruction: Agent instruction from .md file
    - extra_context: Facets and context from instructions.now/day settings
    - prompt: User's runtime query/request
    - transcript: Clustered transcript (if day provided)
    - output_path: Where to write output (if output format set)
    - skip_reason: Why to skip (if applicable)

    Context is controlled by explicit frontmatter settings:
    - instructions.now: Include current datetime in extra_context
    - instructions.day: Include analysis day context (requires day parameter)
    - Day-based calls also load clustered transcript

    Args:
        request: Raw request dict from cortex

    Returns:
        Fully prepared config dict
    """
    from think.models import resolve_model_for_provider, resolve_provider
    from think.muse import get_agent, key_to_context

    name = request.get("name", "default")
    facet = request.get("facet")
    day = request.get("day")
    segment = request.get("segment")
    span = request.get("span")
    output_format = request.get("output")
    output_path_override = request.get("output_path")
    user_prompt = request.get("prompt", "")

    # Load complete agent config, passing day for instructions.day context
    config = get_agent(name, facet=facet, analysis_day=day)

    # Config now contains all frontmatter fields plus:
    # - path: Path to the .md file
    # - system_instruction, user_instruction, extra_context
    # - sources: Source config for transcript loading
    # - All frontmatter: tools, hook, disabled, thinking_budget, max_output_tokens, etc.

    # Convert path string to Path object for convenience
    agent_path = Path(config["path"]) if config.get("path") else None
    sources = config.get("sources", {})

    # Merge request values (request overrides agent defaults)
    config.update({k: v for k, v in request.items() if v is not None})

    # Track additional state
    config["span_mode"] = bool(span)
    config["source_counts"] = {}

    # Resolve provider and model from context
    context = key_to_context(name)
    default_provider, default_model = resolve_provider(context)

    provider = config.get("provider") or default_provider
    model = config.get("model")
    if not model:
        if provider != default_provider:
            model = resolve_model_for_provider(context, provider)
        else:
            model = default_model

    config["provider"] = provider
    config["model"] = model

    # Check if disabled
    if config.get("disabled"):
        config["skip_reason"] = "disabled"
        return config

    # Day-based processing: load transcript and apply template substitution
    if day:
        # Merge extra_context into system_instruction for day-based analysis
        if "system_instruction" not in request:
            system_instruction = config.get("system_instruction", "")
            extra_ctx = config.get("extra_context")
            if extra_ctx:
                system_instruction = f"{system_instruction}\n\n{extra_ctx}"
            config["system_instruction"] = system_instruction

        # Load transcript
        transcript, source_counts = _load_transcript(day, segment, span, sources)
        config["transcript"] = transcript
        config["source_counts"] = source_counts
        total_count = sum(source_counts.values())

        # Check required sources
        for source_type, mode in sources.items():
            if source_is_required(mode) and source_counts.get(source_type, 0) == 0:
                config["skip_reason"] = f"missing_required_{source_type}"
                return config

        # Skip if no content
        if total_count == 0 or len(transcript.strip()) < MIN_INPUT_CHARS:
            config["skip_reason"] = "no_input"
            return config

        # Note for limited recordings
        if total_count < 3:
            config["transcript"] = (
                "**Input Note:** Limited recordings for this day. "
                "Scale analysis to available input.\n\n" + transcript
            )

        # Reload agent instruction with template substitution for day/segment context
        if agent_path and agent_path.exists():
            prompt_context = _build_prompt_context(day, segment, span)
            agent_prompt_obj = load_prompt(
                agent_path.stem, base_dir=agent_path.parent, context=prompt_context
            )
            config["user_instruction"] = agent_prompt_obj.text

    # Set prompt (user's runtime query)
    # For tool agents: prompt is the user's question
    # For generators: prompt is typically empty (instruction is in user_instruction)
    config["prompt"] = user_prompt

    # Determine output path
    if output_format:
        if output_path_override:
            config["output_path"] = Path(output_path_override)
        elif day:
            day_dir = str(day_path(day))
            config["output_path"] = get_output_path(
                day_dir, name, segment=segment, output_format=output_format, facet=facet
            )

    return config


def validate_config(config: dict) -> str | None:
    """Validate prepared config.

    Args:
        config: Prepared config dict

    Returns:
        Error message string if invalid, None if valid
    """
    is_cogitate = config["type"] == "cogitate"
    has_prompt = bool(config.get("prompt"))
    has_user_instruction = bool(config.get("user_instruction"))
    has_day = bool(config.get("day"))

    # Cogitate agents need a prompt (user's question)
    if is_cogitate and not has_prompt:
        return "Missing 'prompt' field for cogitate agent"

    # Generate prompts need either day (transcript) or user_instruction
    if not is_cogitate and not has_day and not has_user_instruction and not has_prompt:
        return "Invalid config: must have 'type', 'day', or 'prompt'"

    # Segment/span requires day
    if (config.get("segment") or config.get("span")) and not has_day:
        return "Invalid config: 'segment' or 'span' requires 'day'"

    return None


# =============================================================================
# Hook Execution
# =============================================================================


def _run_pre_hooks(config: dict) -> dict:
    """Run pre-processing hooks, return dict of modifications.

    Args:
        config: Full config dict (hooks receive this directly)

    Returns:
        Dict of field modifications to apply to config
    """
    pre_hook = load_pre_hook(config)
    if not pre_hook:
        return {}

    try:
        modifications = pre_hook(config)
        if modifications:
            LOG.info("Pre-hook returned modifications: %s", list(modifications.keys()))
            return modifications
    except Exception as exc:
        LOG.error("Pre-hook failed: %s", exc)

    return {}


def _run_post_hooks(result: str, config: dict) -> str:
    """Run post-processing hooks, return transformed result.

    Args:
        result: LLM output text
        config: Full config dict (hooks receive this directly)

    Returns:
        Transformed result (or original if no hook)
    """
    post_hook = load_post_hook(config)
    if not post_hook:
        return result

    try:
        hook_result = post_hook(result, config)
        if hook_result is not None:
            LOG.info("Post-hook transformed result")
            return hook_result
    except Exception as exc:
        LOG.error("Post-hook failed: %s", exc)

    return result


# =============================================================================
# Unified Agent Execution
# =============================================================================


def _write_output(output_path: Path, result: str) -> None:
    """Write result to output file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)
    LOG.info("Wrote output to %s", output_path)


def _build_dry_run_event(config: dict, before_values: dict) -> dict:
    """Build a dry-run event with all context."""
    agent_type = config["type"]

    event: dict[str, Any] = {
        "event": "dry_run",
        "ts": now_ms(),
        "type": agent_type,
        "name": config.get("name", "default"),
        "provider": config.get("provider", ""),
        "model": config.get("model") or "unknown",
        "system_instruction": config.get("system_instruction", ""),
        "user_instruction": config.get("user_instruction", ""),
        "prompt": config.get("prompt", ""),
    }

    if agent_type == "cogitate":
        event["extra_context"] = config.get("extra_context", "")

    # Day-based fields
    if config.get("day"):
        event["day"] = config["day"]
        event["segment"] = config.get("segment")
        transcript = config.get("transcript", "")
        if transcript:
            event["transcript"] = transcript
            event["transcript_chars"] = len(transcript)
            event["transcript_files"] = sum(config.get("source_counts", {}).values())
        output_path = config.get("output_path")
        if output_path:
            event["output_path"] = str(output_path)

    # Show before values for comparison
    for key, before_val in before_values.items():
        current_val = config.get(key, "")
        if current_val != before_val:
            if key == "transcript":
                event["transcript_before_chars"] = len(before_val)
            else:
                event[f"{key}_before"] = before_val

    return event


async def _execute_with_tools(
    config: dict,
    emit_event: Callable[[dict], None],
) -> None:
    """Execute tool-using agent via provider's run_cogitate.

    Args:
        config: Prepared config dict
        emit_event: Event emission callback
    """
    from .providers import PROVIDER_REGISTRY, get_provider_module

    provider = config.get("provider", "google")
    output_path = config.get("output_path")

    if provider not in PROVIDER_REGISTRY:
        valid = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
        raise ValueError(f"Unknown provider: {provider!r}. Valid providers: {valid}")

    provider_mod = get_provider_module(provider)

    # Wrapper to intercept finish event for post-processing
    def agent_emit_event(data: Event) -> None:
        if data.get("event") == "finish":
            result = data.get("result", "")
            result = _run_post_hooks(result, config)
            if result != data.get("result", ""):
                data = {**data, "result": result}
            if output_path and result:
                _write_output(output_path, result)
            if config.get("handoff"):
                data = {**data, "handoff": config["handoff"]}

        # Filter out start events from providers (we already emitted ours)
        if data.get("event") == "start":
            return

        emit_event(data)

    await provider_mod.run_cogitate(config=config, on_event=agent_emit_event)


async def _execute_generate(
    config: dict,
    emit_event: Callable[[dict], None],
) -> None:
    """Execute single-shot generation (no tools).

    Args:
        config: Prepared config dict
        emit_event: Event emission callback
    """
    from think.models import generate_with_result
    from think.muse import key_to_context

    name = config.get("name", "default")
    transcript = config.get("transcript", "")
    user_instruction = config.get("user_instruction", "")
    prompt = config.get("prompt", "")
    system_instruction = config.get("system_instruction", "")
    output_path = config.get("output_path")
    output_format = config.get("output")

    # Get generation parameters from config (set in frontmatter)
    thinking_budget = config.get("thinking_budget") or 8192 * 3
    max_output_tokens = config.get("max_output_tokens") or 8192 * 6
    is_json_output = output_format == "json"

    # Build contents: transcript + instruction + prompt
    contents = []
    if transcript:
        contents.append(transcript)
    if user_instruction:
        contents.append(user_instruction)
    if prompt:
        contents.append(prompt)

    # Fallback if no contents
    if not contents:
        contents = ["No input provided."]

    context = key_to_context(name)
    gen_result = generate_with_result(
        contents=contents,
        context=context,
        temperature=0.3,
        max_output_tokens=max_output_tokens,
        thinking_budget=thinking_budget,
        system_instruction=system_instruction,
        json_output=is_json_output,
    )

    result = gen_result["text"]
    usage_data = gen_result.get("usage")

    # Run post-hooks
    result = _run_post_hooks(result, config)

    # Write output
    if output_path and result:
        _write_output(output_path, result)

    # Emit finish event
    finish_event: dict[str, Any] = {
        "event": "finish",
        "ts": now_ms(),
        "result": result,
    }
    if usage_data:
        finish_event["usage"] = usage_data
    if config.get("handoff"):
        finish_event["handoff"] = config["handoff"]
    emit_event(finish_event)


async def _run_agent(
    config: dict,
    emit_event: Callable[[dict], None],
    dry_run: bool = False,
) -> None:
    """Execute agent based on config.

    Unified execution path for all agent types. Handles:
    - Skip conditions (disabled, no input, etc.)
    - Output existence checking (skip if exists unless force)
    - Pre/post hooks
    - Dry-run mode
    - Routing to tool or generate execution

    Args:
        config: Fully prepared config dict
        emit_event: Callback to emit JSONL events
        dry_run: If True, emit dry_run event instead of calling LLM
    """
    name = config.get("name", "default")
    provider = config.get("provider", "google")
    model = config.get("model")
    is_cogitate = config["type"] == "cogitate"
    force = config.get("force", False)
    output_path = config.get("output_path")

    # Emit start event
    start_event: dict[str, Any] = {
        "event": "start",
        "ts": now_ms(),
        "prompt": config.get("prompt", ""),
        "name": name,
        "model": model or "unknown",
        "provider": provider,
    }
    if config.get("session_id"):
        start_event["session_id"] = config["session_id"]
    if config.get("chat_id"):
        start_event["chat_id"] = config["chat_id"]
    emit_event(start_event)

    # Handle skip conditions
    skip_reason = config.get("skip_reason")
    if skip_reason:
        LOG.info("Config %s skipped: %s", name, skip_reason)
        emit_event(
            {
                "event": "finish",
                "ts": now_ms(),
                "result": "",
                "skipped": skip_reason,
            }
        )
        if config.get("day"):
            day_log(config["day"], f"agent {name} skipped ({skip_reason})")
        return

    # Check if output already exists (applies to both tool agents and generators)
    if output_path and not force and not dry_run:
        if output_path.exists() and output_path.stat().st_size > 0:
            LOG.info("Output exists, loading: %s", output_path)
            with open(output_path, "r") as f:
                result = f.read()
            emit_event(
                {
                    "event": "finish",
                    "ts": now_ms(),
                    "result": result,
                }
            )
            return

    # Capture state before pre-hooks
    before_values = {
        "prompt": config.get("prompt", ""),
        "system_instruction": config.get("system_instruction", ""),
        "user_instruction": config.get("user_instruction", ""),
        "transcript": config.get("transcript", ""),
    }
    if is_cogitate:
        before_values["extra_context"] = config.get("extra_context", "")

    # Run pre-hooks
    modifications = _run_pre_hooks(config)
    for key, value in modifications.items():
        config[key] = value

    # Dry-run mode
    if dry_run:
        emit_event(_build_dry_run_event(config, before_values))
        return

    # Execute based on agent type
    if is_cogitate:
        await _execute_with_tools(config, emit_event)
    else:
        await _execute_generate(config, emit_event)

    # Log completion
    if config.get("day"):
        day_log(config["day"], f"agent {name} ok")


# =============================================================================
# Utility Functions
# =============================================================================


def scan_day(day: str) -> dict[str, list[str]]:
    """Return lists of processed and pending daily generator output files.

    Only scans daily generators (schedule='daily'). Segment generators are
    stored within segment directories and are not included here.
    """
    day_dir = day_path(day)
    daily_generators = get_muse_configs(
        type="generate", schedule="daily", include_disabled=True
    )
    processed: list[str] = []
    pending: list[str] = []
    for key, meta in sorted(daily_generators.items()):
        output_format = meta.get("output")
        output_file = get_output_path(day_dir, key, output_format=output_format)
        if output_file.exists():
            processed.append(os.path.join("agents", output_file.name))
        else:
            pending.append(os.path.join("agents", output_file.name))
    return {"processed": sorted(processed), "repairable": sorted(pending)}


# =============================================================================
# Main Entry Point
# =============================================================================


async def main_async() -> None:
    """NDJSON-based CLI for agents."""
    parser = argparse.ArgumentParser(
        description="solstone Agent CLI - Accepts NDJSON input via stdin"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be sent to the provider without calling the LLM",
    )

    args = setup_cli(parser)
    dry_run = args.dry_run

    app_logger = setup_logging(args.verbose)
    event_writer = JSONEventWriter(None)

    def emit_event(data: Event) -> None:
        if "ts" not in data:
            data["ts"] = now_ms()
        event_writer.emit(data)

    try:
        app_logger.info("Processing NDJSON input from stdin")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                config = prepare_config(request)

                error = validate_config(config)
                if error:
                    emit_event({"event": "error", "error": error, "ts": now_ms()})
                    continue

                await _run_agent(config, emit_event, dry_run=dry_run)

            except json.JSONDecodeError as e:
                emit_event(
                    {
                        "event": "error",
                        "error": f"Invalid JSON: {str(e)}",
                        "ts": now_ms(),
                    }
                )
            except Exception as e:
                emit_event(
                    {
                        "event": "error",
                        "error": str(e),
                        "trace": traceback.format_exc(),
                        "ts": now_ms(),
                    }
                )

    except Exception as exc:
        err = {
            "event": "error",
            "error": str(exc),
            "trace": traceback.format_exc(),
        }
        if not getattr(exc, "_evented", False):
            emit_event(err)
        raise
    finally:
        event_writer.close()


def main() -> None:
    """Entry point wrapper."""
    asyncio.run(main_async())


__all__ = [
    "prepare_config",
    "validate_config",
    "scan_day",
]
