# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unified agent and generator CLI for solstone.

Spawned by cortex for both:
- Tool-using agents (configs with 'tools' field)
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
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypedDict, Union

from google import genai
from google.genai import types
from typing_extensions import Required

from think.cluster import cluster, cluster_period, cluster_span
from think.utils import (
    compose_instructions,
    day_log,
    day_path,
    format_day,
    format_segment_times,
    get_muse_configs,
    get_output_path,
    load_output_hook,
    load_prompt,
    segment_parse,
    setup_cli,
)

LOG = logging.getLogger("think.agents")


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for agent CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, stream=sys.stdout)
    return LOG


class ToolStartEvent(TypedDict, total=False):
    """Event emitted when a tool starts."""

    event: Literal["tool_start"]
    ts: int
    tool: str
    args: Optional[dict[str, Any]]
    call_id: Optional[str]  # Unique ID to pair with tool_end event


class ToolEndEvent(TypedDict, total=False):
    """Event emitted when a tool finishes."""

    event: Literal["tool_end"]
    ts: int
    tool: str
    args: Optional[dict[str, Any]]
    result: Any
    call_id: Optional[str]  # Matches the call_id from tool_start


class StartEvent(TypedDict):
    """Event emitted when an agent run begins."""

    event: Literal["start"]
    ts: int
    prompt: str
    name: str
    model: str
    provider: str


class FinishEvent(TypedDict):
    """Event emitted when an agent run finishes successfully."""

    event: Literal["finish"]
    ts: int
    result: str


class ErrorEvent(TypedDict, total=False):
    """Event emitted when an error occurs."""

    event: Literal["error"]
    ts: int
    error: str
    trace: Optional[str]


class AgentUpdatedEvent(TypedDict):
    """Event emitted when the agent context changes."""

    event: Literal["agent_updated"]
    ts: int
    agent: str


class ThinkingEvent(TypedDict, total=False):
    """Event emitted when thinking/reasoning summaries are available.

    For Anthropic models, may include a signature for verification when
    passing thinking blocks back during tool use continuations.
    For redacted thinking, summary will contain "[redacted]" and
    redacted_data will contain the encrypted content.
    """

    event: Required[Literal["thinking"]]
    ts: Required[int]
    summary: Required[str]
    model: Optional[str]
    signature: Optional[str]  # Anthropic thinking block signature
    redacted_data: Optional[str]  # Encrypted data for redacted thinking


Event = Union[
    ToolStartEvent,
    ToolEndEvent,
    StartEvent,
    FinishEvent,
    ErrorEvent,
    ThinkingEvent,
    AgentUpdatedEvent,
]


class GenerateResult(TypedDict, total=False):
    """Result from provider run_generate/run_agenerate functions.

    Structured result that allows the wrapper to handle cross-cutting concerns
    like token logging and JSON validation centrally.

    The thinking field contains dicts with: summary (str), signature (optional str),
    redacted_data (optional str for Anthropic redacted thinking).
    """

    text: Required[str]  # Response text
    usage: Optional[dict]  # Normalized usage dict (input_tokens, output_tokens, etc.)
    finish_reason: Optional[str]  # Normalized: "stop", "max_tokens", "safety", etc.
    thinking: Optional[list]  # List of thinking block dicts


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


class JSONEventCallback:
    """Emit JSON events via a callback."""

    def __init__(self, callback: Optional[Callable[[Event], None]] = None) -> None:
        self.callback = callback

    def emit(self, data: Event) -> None:
        if "ts" not in data:
            data = {**data, "ts": int(time.time() * 1000)}
        if self.callback:
            self.callback(data)

    def close(self) -> None:
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
    import logging

    from think.cortex_client import read_agent_events

    logger = logging.getLogger(__name__)

    try:
        events = read_agent_events(conversation_id)
    except FileNotFoundError:
        logger.warning(f"Cannot continue from {conversation_id}: log not found")
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


__all__ = [
    "ToolStartEvent",
    "ToolEndEvent",
    "StartEvent",
    "FinishEvent",
    "ErrorEvent",
    "AgentUpdatedEvent",
    "ThinkingEvent",
    "GenerateResult",
    "Event",
    "JSONEventWriter",
    "JSONEventCallback",
    "format_tool_summary",
    "parse_agent_events_to_turns",
    "scan_day",
    "generate_agent_output",
]


# =============================================================================
# Generator Functions (for transcript analysis without tools)
# =============================================================================

# Minimum content length for generator output
MIN_INPUT_CHARS = 50


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
    should include both day and system prompt name.
    """
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


def _run_generator(config: dict, emit_event: Callable[[dict], None]) -> None:
    """Execute generator pipeline with config from cortex.

    Args:
        config: Merged config from cortex containing:
            - name: Generator key (e.g., 'activity', 'chat:sentiment')
            - day: Day in YYYYMMDD format
            - segment: Optional single segment key
            - span: Optional list of sequential segment keys
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
    span = config.get("span")  # List of sequential segment keys
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
    elif span:
        os.environ["SEGMENT_KEY"] = span[0]

    # Load generator metadata
    all_generators = get_muse_configs(has_tools=False, has_output=True)
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

    # Track span mode (multiple sequential segments)
    span_mode = bool(span)

    # Build transcript via clustering
    if span:
        markdown, file_count = cluster_span(day, span, sources=sources)
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
    if span_mode:
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
                    "span": span_mode,
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


# =============================================================================
# Main Entry Point
# =============================================================================


async def main_async() -> None:
    """NDJSON-based CLI for agents and generators.

    Routes based on config:
    - 'tools' field present -> tool-using agent (via provider)
    - 'output' field present (no 'tools') -> generator (transcript analysis)
    """
    parser = argparse.ArgumentParser(
        description="solstone Agent CLI - Accepts NDJSON input via stdin"
    )

    args = setup_cli(parser)

    app_logger = setup_logging(args.verbose)

    # Always write to stdout only
    event_writer = JSONEventWriter(None)

    def emit_event(data: Event) -> None:
        if "ts" not in data:
            data["ts"] = int(time.time() * 1000)
        event_writer.emit(data)

    try:
        # NDJSON input mode from stdin only
        app_logger.info("Processing NDJSON input from stdin")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                # Parse NDJSON line - this is the complete merged config from Cortex
                config = json.loads(line)

                # Route based on config type
                has_tools = bool(config.get("tools"))
                has_output = bool(config.get("output"))

                if has_output and not has_tools:
                    # Generator: transcript analysis without tools
                    app_logger.debug(f"Processing generator: {config.get('name')}")
                    _run_generator(config, emit_event)

                elif has_tools:
                    # Tool-using agent: validate prompt exists
                    prompt = config.get("prompt")
                    if not prompt:
                        emit_event(
                            {
                                "event": "error",
                                "error": "Missing 'prompt' field for tool agent",
                                "ts": int(time.time() * 1000),
                            }
                        )
                        continue

                    # Extract provider to route to correct module
                    from .providers import PROVIDER_REGISTRY, get_provider_module

                    provider = config.get("provider", "google")

                    # Set OpenAI key if needed
                    if provider == "openai":
                        api_key = os.getenv("OPENAI_API_KEY", "")
                        if api_key:
                            from agents import set_default_openai_key

                            set_default_openai_key(api_key)

                    app_logger.debug(f"Processing agent: provider={provider}")

                    # Route to appropriate provider module
                    if provider in PROVIDER_REGISTRY:
                        provider_mod = get_provider_module(provider)
                    else:
                        # Explicit error for unknown providers
                        valid = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
                        raise ValueError(
                            f"Unknown provider: {provider!r}. Valid providers: {valid}"
                        )

                    # Pass complete config to provider
                    await provider_mod.run_agent(
                        config=config,
                        on_event=emit_event,
                    )

                else:
                    # Neither tools nor output - invalid config
                    emit_event(
                        {
                            "event": "error",
                            "error": "Invalid config: must have 'tools' or 'output' field",
                            "ts": int(time.time() * 1000),
                        }
                    )

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

    except Exception as exc:  # pragma: no cover - unexpected
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
