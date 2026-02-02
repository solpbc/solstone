# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unified agent and generator CLI for solstone.

Spawned by cortex for both:
- Agents (with or without tools, conversational or tool-using)
- Transcript generators (configs with 'output' field, no 'tools')

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
    compose_instructions,
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


def format_tool_summary(tool_calls: list) -> str:
    """Format tool calls into a readable summary string.

    Args:
        tool_calls: List of dicts with 'name' and 'args' keys

    Returns:
        Formatted string like "Tools used: fetch_data(url='...'), process(id=123)"
        or empty string if no tool calls
    """
    if not tool_calls:
        return ""

    tool_summaries = []
    for tool in tool_calls:
        name = tool.get("name", "unknown")
        args = tool.get("args", {})
        # Format args as compact string
        args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in args.items())
        tool_summaries.append(f"{name}({args_str})")

    return "\n\nTools used: " + ", ".join(tool_summaries)


def parse_agent_events_to_turns(conversation_id: str) -> list:
    """Parse agent event log into conversation turns.

    Converts agent event logs into simple conversation turns with role and content.
    Automatically combines assistant text with tool usage summaries.

    Args:
        conversation_id: Agent ID whose conversation to load

    Returns:
        List of dicts like [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        Returns empty list if conversation not found

    Note:
        Incomplete turns (missing finish event) are skipped
    """
    from think.cortex_client import read_agent_events

    try:
        events = read_agent_events(conversation_id)
    except FileNotFoundError:
        LOG.warning(f"Cannot continue from {conversation_id}: log not found")
        return []

    turns = []
    current_tool_calls: list = []

    for event in events:
        event_type = event.get("event")

        if event_type == "start":
            # User's initial prompt
            prompt = event.get("prompt", "")
            if prompt:
                turns.append({"role": "user", "content": prompt})

        elif event_type == "tool_start":
            # Track tool calls for current assistant turn
            tool_name = event.get("tool", "")
            tool_args = event.get("args", {})
            current_tool_calls.append({"name": tool_name, "args": tool_args})

        elif event_type == "finish":
            # Assistant's response with optional tool summary
            result_text = event.get("result", "").strip()

            # Build content combining response and tool usage
            content_parts = []
            if result_text:
                content_parts.append(result_text)

            if current_tool_calls:
                tool_summary = format_tool_summary(current_tool_calls)
                content_parts.append(tool_summary)

            if content_parts:
                turns.append({"role": "assistant", "content": "\n".join(content_parts)})

            # Reset tool tracking for next turn
            current_tool_calls = []

    return turns


# =============================================================================
# Config Hydration, Enrichment, and Validation
# =============================================================================


def hydrate_config(request: dict) -> dict:
    """Load agent config and merge with request.

    Takes the raw request from cortex and returns a fully hydrated config with:
    - Base agent config loaded from muse/*.md
    - Request values merged (request overrides defaults)
    - Provider and model resolved from context
    - Tools expanded from pack names to tool list

    Args:
        request: Raw request dict from cortex with at least 'name' field

    Returns:
        Fully hydrated config dict ready for routing
    """
    from think.models import resolve_model_for_provider, resolve_provider
    from think.muse import get_agent, key_to_context

    name = request.get("name", "default")
    facet = request.get("facet")

    # Load base config from agent definition
    config = get_agent(name, facet=facet)

    # Merge request into config (request values override agent defaults)
    # Only override with non-None values to preserve agent defaults
    config.update({k: v for k, v in request.items() if v is not None})

    # Resolve provider and model from context
    context = key_to_context(name)
    default_provider, default_model = resolve_provider(context)

    # Provider can be overridden by request or agent config
    provider = config.get("provider") or default_provider

    # Model: use explicit model from request/config, or resolve from provider
    model = config.get("model")
    if not model:
        if provider != default_provider:
            model = resolve_model_for_provider(context, provider)
        else:
            model = default_model

    config["provider"] = provider
    config["model"] = model

    # Expand tools if it's a string (tool pack name)
    tools_config = config.get("tools")
    if isinstance(tools_config, str):
        config["tools"] = expand_tools(tools_config)

    return config


def expand_tools(tools_config: str) -> list[str]:
    """Expand tool pack names to a list of tool names.

    Args:
        tools_config: Comma-separated tool pack names (e.g., "default,entities")

    Returns:
        List of unique tool names from all packs
    """
    from think.mcp import get_tools

    pack_names = [p.strip() for p in tools_config.split(",") if p.strip()]
    if not pack_names:
        pack_names = ["default"]

    expanded: list[str] = []
    for pack in pack_names:
        try:
            for tool in get_tools(pack):
                if tool not in expanded:
                    expanded.append(tool)
        except KeyError:
            LOG.warning(f"Invalid tool pack '{pack}', using default")
            for tool in get_tools("default"):
                if tool not in expanded:
                    expanded.append(tool)

    return expanded


def validate_config(config: dict) -> str | None:
    """Validate agent config.

    Args:
        config: Hydrated config dict

    Returns:
        Error message string if invalid, None if valid
    """
    has_tools = bool(config.get("tools"))
    has_prompt = bool(config.get("prompt"))
    has_day = bool(config.get("day"))

    # Tools path requires prompt
    if has_tools and not has_prompt:
        return "Missing 'prompt' field for tool agent"

    # Generate path requires at least prompt or day
    if not has_tools and not has_prompt and not has_day:
        return "Invalid config: must have 'tools', 'prompt', or 'day' field"

    # Segment/span requires day
    if (config.get("segment") or config.get("span")) and not has_day:
        return "Invalid config: 'segment' or 'span' requires 'day' field"

    # Validate continue_from if present
    continue_from = config.get("continue_from")
    if continue_from:
        from think.cortex_client import get_agent_log_status

        status = get_agent_log_status(continue_from)
        if status == "running":
            return f"Cannot continue from {continue_from}: agent is still running"
        if status == "not_found":
            return f"Cannot continue from {continue_from}: agent not found"

    return None


def enrich_config(config: dict) -> None:
    """Enrich config with transcript, system instruction, output path, etc.

    Mutates config in place, adding:
    - transcript: Clustered transcript content (if day specified)
    - system_instruction: System prompt (if not already set)
    - output_path: Where to write output (if output format specified)
    - source_counts: Dict of source type -> count
    - skip_reason: Why to skip execution (if applicable)
    - span_mode: Whether in span mode
    - meta: Agent metadata from muse config

    Args:
        config: Hydrated config dict to enrich
    """
    name = config.get("name", "default")
    day = config.get("day")
    segment = config.get("segment")
    span = config.get("span")  # List of sequential segment keys
    facet = config.get("facet")
    output_format = config.get("output")
    output_path_override = config.get("output_path")
    user_prompt = config.get("prompt", "")

    # Load config metadata (for hooks, system prompt, etc.)
    all_configs = get_muse_configs(has_tools=False)
    if name not in all_configs:
        all_configs = get_muse_configs()  # Include tool configs
    if name in all_configs:
        meta = all_configs[name]
        agent_path = Path(meta["path"])
    else:
        meta = {}
        agent_path = None

    config["meta"] = meta
    config["agent_path"] = agent_path
    config["span_mode"] = bool(span)
    config["source_counts"] = {}

    # Check if config is disabled
    if meta.get("disabled"):
        config["skip_reason"] = "disabled"
        return

    # Extract instructions config for source filtering and system prompt
    instructions_config = meta.get("instructions")
    instructions = compose_instructions(
        facet=facet,
        include_datetime=not day,
        config_overrides=instructions_config,
    )
    sources = instructions.get("sources", {})
    config["system_prompt_name"] = instructions.get("system_prompt_name", "journal")

    # Set system_instruction if not already provided
    if not config.get("system_instruction"):
        system_instruction = instructions["system_instruction"]
        # Append extra_context (facets, etc.) to system instruction if present
        extra_context = instructions.get("extra_context")
        if extra_context:
            system_instruction = f"{system_instruction}\n\n{extra_context}"
        config["system_instruction"] = system_instruction

    # Transcript loading (only if day is provided)
    if day:
        # Set segment key for token usage logging
        if segment:
            os.environ["SEGMENT_KEY"] = segment
        elif span:
            os.environ["SEGMENT_KEY"] = span[0]

        # Convert sources for clustering
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
            transcript, source_counts = cluster_span(day, span, sources=cluster_sources)
        elif segment:
            transcript, source_counts = cluster_period(
                day, segment, sources=cluster_sources
            )
        else:
            transcript, source_counts = cluster(day, sources=cluster_sources)

        config["transcript"] = transcript
        config["source_counts"] = source_counts
        total_count = sum(source_counts.values())

        # Check required sources have content
        for source_type, mode in sources.items():
            if source_is_required(mode) and source_counts.get(source_type, 0) == 0:
                config["skip_reason"] = f"missing_required_{source_type}"
                return

        # Skip when there's nothing to analyze
        if total_count == 0 or len(transcript.strip()) < MIN_INPUT_CHARS:
            config["skip_reason"] = "no_input"
            return

        # Prepend input context note for limited recordings
        if total_count < 3:
            input_note = (
                "**Input Note:** Limited recordings for this day. "
                "Scale analysis to available input.\n\n"
            )
            config["transcript"] = input_note + transcript

    # Build context for template substitution
    prompt_context: dict[str, str] = {}
    if day:
        prompt_context["day"] = day
        prompt_context["date"] = format_day(day)

        if segment:
            start_str, end_str = format_segment_times(segment)
            if start_str and end_str:
                prompt_context["segment"] = segment
                prompt_context["segment_start"] = start_str
                prompt_context["segment_end"] = end_str
        elif span:
            all_times = []
            for seg in span:
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

    # Load prompt from agent file if available, otherwise use user prompt
    if agent_path and agent_path.exists():
        agent_prompt_obj = load_prompt(
            agent_path.stem, base_dir=agent_path.parent, context=prompt_context
        )
        prompt = agent_prompt_obj.text
        # Append user prompt if both exist
        if user_prompt and prompt != user_prompt:
            prompt = f"{prompt}\n\n{user_prompt}"
        config["prompt"] = prompt

    # Determine output path
    if output_format:
        if output_path_override:
            config["output_path"] = Path(output_path_override)
        elif day:
            day_dir = str(day_path(day))
            config["output_path"] = get_output_path(
                day_dir, name, segment=segment, output_format=output_format, facet=facet
            )


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
    meta = config.get("meta", {})
    pre_hook = load_pre_hook(meta)
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
    meta = config.get("meta", {})
    post_hook = load_post_hook(meta)
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
    has_tools = bool(config.get("tools"))
    run_type = "agent" if has_tools else "generate"

    event: dict[str, Any] = {
        "event": "dry_run",
        "ts": now_ms(),
        "type": run_type,
        "name": config.get("name", "default"),
        "provider": config.get("provider", ""),
        "model": config.get("model") or "unknown",
        "system_instruction": config.get("system_instruction", ""),
        "prompt": config.get("prompt", ""),
    }

    if has_tools:
        event["user_instruction"] = config.get("user_instruction", "")
        event["extra_context"] = config.get("extra_context", "")
        event["tools"] = config.get("tools", [])
    else:
        event["system_instruction_source"] = config.get("system_prompt_name", "journal")
        agent_path = config.get("agent_path")
        event["prompt_source"] = str(agent_path) if agent_path else "request"

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


async def _run_agent(
    config: dict,
    emit_event: Callable[[dict], None],
    dry_run: bool = False,
) -> None:
    """Execute agent or generator based on config.

    Unified execution path for both tool-using agents and transcript generators.
    The only branch is at the LLM call - everything else is shared.

    Args:
        config: Fully hydrated and enriched config dict
        emit_event: Callback to emit JSONL events
        dry_run: If True, emit dry_run event instead of calling LLM
    """
    name = config.get("name", "default")
    provider = config.get("provider", "google")
    model = config.get("model")
    has_tools = bool(config.get("tools"))
    force = config.get("force", False)

    # Emit start event
    start_event: dict[str, Any] = {
        "event": "start",
        "ts": now_ms(),
        "prompt": config.get("prompt", ""),
        "name": name,
        "model": model or "unknown",
        "provider": provider,
    }
    if config.get("continue_from"):
        start_event["continue_from"] = config["continue_from"]
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

    # Check if output already exists (generators only, not tool agents)
    output_path = config.get("output_path")
    output_format = config.get("output")
    if not has_tools and output_path and not force and not dry_run:
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
        "transcript": config.get("transcript", ""),
    }
    if has_tools:
        before_values["user_instruction"] = config.get("user_instruction", "")
        before_values["extra_context"] = config.get("extra_context", "")

    # Run pre-hooks
    modifications = _run_pre_hooks(config)
    for key, value in modifications.items():
        config[key] = value

    # Dry-run mode
    if dry_run:
        emit_event(_build_dry_run_event(config, before_values))
        return

    # Execute LLM call - this is the only real branch
    if has_tools:
        # Tool-using agent path
        from .providers import PROVIDER_REGISTRY, get_provider_module

        if provider not in PROVIDER_REGISTRY:
            valid = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
            raise ValueError(
                f"Unknown provider: {provider!r}. Valid providers: {valid}"
            )

        provider_mod = get_provider_module(provider)

        # Create wrapper to intercept finish event
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

        await provider_mod.run_tools(config=config, on_event=agent_emit_event)

    else:
        # Generator path - single-shot generation
        from think.models import generate_with_result
        from think.muse import key_to_context

        transcript = config.get("transcript", "")
        prompt = config.get("prompt", "")
        system_instruction = config.get("system_instruction", "")
        meta = config.get("meta", {})

        # Get generation parameters
        thinking_budget = meta.get("thinking_budget") or 8192 * 3
        max_output_tokens = meta.get("max_output_tokens") or 8192 * 6
        is_json_output = output_format == "json"

        context = key_to_context(name)
        gen_result = generate_with_result(
            contents=[transcript, prompt] if transcript else [prompt],
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
        has_tools=False, has_output=True, schedule="daily", include_disabled=True
    )
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


# =============================================================================
# Main Entry Point
# =============================================================================


async def main_async() -> None:
    """NDJSON-based CLI for agents and generators."""
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
                config = hydrate_config(request)

                error = validate_config(config)
                if error:
                    emit_event({"event": "error", "error": error, "ts": now_ms()})
                    continue

                enrich_config(config)
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
    "format_tool_summary",
    "parse_agent_events_to_turns",
    "hydrate_config",
    "expand_tools",
    "validate_config",
    "enrich_config",
    "scan_day",
]
