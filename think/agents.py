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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

from google import genai
from google.genai import types

from think.cluster import cluster, cluster_period, cluster_span
from think.providers.shared import Event, GenerateResult
from think.utils import (
    compose_instructions,
    day_log,
    day_path,
    format_day,
    format_segment_times,
    get_muse_configs,
    get_output_path,
    load_prompt,
    now_ms,
    segment_parse,
    setup_cli,
    source_is_enabled,
    source_is_required,
)

LOG = logging.getLogger("think.agents")


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


# =============================================================================
# Hook Framework (unified for agents and generators)
# =============================================================================


class HookContext(TypedDict, total=False):
    """Context passed to hook functions.

    Provides unified context for both tool-using agents and generators.
    Not all fields are present for all modalities.
    """

    # Identity
    name: str  # Agent/generator name
    agent_id: str  # Unique agent ID
    provider: str  # google/anthropic/openai
    model: str  # Model used

    # Temporal (generators)
    day: str  # YYYYMMDD
    segment: str  # Segment key
    span: bool  # True if span mode

    # Content
    prompt: str  # Original prompt (agents) or empty (generators)
    transcript: str  # Clustered transcript (generators only)

    # Output
    output_path: str  # Where result will be written
    output_format: str  # 'md' or 'json'

    # Full config
    meta: dict  # Full frontmatter/config


class PreHookContext(TypedDict, total=False):
    """Context passed to pre-processing hook functions.

    Pre-hooks receive all inputs before the LLM call and can modify them.
    Returns a dict of modified fields to merge back.
    """

    # Identity
    name: str  # Agent/generator name
    agent_id: str  # Unique agent ID
    provider: str  # google/anthropic/openai
    model: str  # Model used

    # Temporal (generators)
    day: str  # YYYYMMDD
    segment: str  # Segment key
    span: bool  # True if span mode

    # Modifiable inputs
    prompt: str  # User prompt (can modify)
    system_instruction: str  # System prompt (can modify)
    user_instruction: str  # User instruction (agents, can modify)
    extra_context: str  # Extra context (agents, can modify)
    transcript: str  # Clustered transcript (generators, can modify)

    # Output settings
    output_path: str  # Where result will be written
    output_format: str  # 'md' or 'json'

    # Full config (read-only reference)
    meta: dict  # Full frontmatter/config


# MUSE_DIR for hook resolution
_MUSE_DIR = Path(__file__).parent.parent / "muse"


def _resolve_hook_path(hook_name: str) -> Path:
    """Resolve hook name to file path.

    Resolution:
    - Named: "name" -> muse/{name}.py
    - App-qualified: "app:name" -> apps/{app}/muse/{name}.py
    - Explicit path: "path/to/hook.py" -> direct path
    """
    if "/" in hook_name or hook_name.endswith(".py"):
        return Path(hook_name)
    elif ":" in hook_name:
        app, name = hook_name.split(":", 1)
        return Path(__file__).parent.parent / "apps" / app / "muse" / f"{name}.py"
    else:
        return _MUSE_DIR / f"{hook_name}.py"


def _load_hook_function(config: dict, key: str, func_name: str) -> Callable | None:
    """Load a hook function from config.

    Args:
        config: Agent/generator config dict
        key: Hook key in config ("pre" or "post")
        func_name: Function name to load ("pre_process" or "post_process")

    Returns:
        The hook function, or None if no hook configured.

    Raises:
        ValueError: If hook file doesn't define the required function.
        ImportError: If hook file cannot be loaded.
    """
    import importlib.util

    hook_config = config.get("hook")
    if not hook_config or not isinstance(hook_config, dict):
        return None

    hook_name = hook_config.get(key)
    if not hook_name:
        return None

    hook_path = _resolve_hook_path(hook_name)

    if not hook_path.exists():
        raise ImportError(f"Hook file not found: {hook_path}")

    spec = importlib.util.spec_from_file_location(
        f"{key}_hook_{hook_path.stem}", hook_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load hook from {hook_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, func_name):
        raise ValueError(f"Hook {hook_path} must define a '{func_name}' function")

    process_func = getattr(module, func_name)
    if not callable(process_func):
        raise ValueError(f"Hook {hook_path} '{func_name}' must be callable")

    return process_func


def load_post_hook(config: dict) -> Callable[[str, HookContext], str | None] | None:
    """Load post-processing hook from config if defined.

    Hook config format: {"hook": {"post": "name"}}
    """
    return _load_hook_function(config, "post", "post_process")


def load_pre_hook(config: dict) -> Callable[[PreHookContext], dict | None] | None:
    """Load pre-processing hook from config if defined.

    Hook config format: {"hook": {"pre": "name"}}
    """
    return _load_hook_function(config, "pre", "pre_process")


def _build_base_context(config: dict) -> dict:
    """Build common context fields shared by pre and post hooks."""
    context = {
        "name": config.get("name", ""),
        "agent_id": config.get("agent_id", ""),
        "provider": config.get("provider", ""),
        "model": config.get("model", ""),
        "prompt": config.get("prompt", ""),
        "output_format": config.get("output", "md"),
        "meta": config,
    }

    # Add generator-specific fields if present
    if "day" in config:
        context["day"] = config["day"]
    if "segment" in config:
        context["segment"] = config["segment"]

    return context


def build_pre_hook_context(config: dict, **extras: Any) -> PreHookContext:
    """Build PreHookContext from config and extra values."""
    context: PreHookContext = _build_base_context(config)

    # Add pre-hook specific fields
    context["system_instruction"] = config.get("system_instruction", "")
    context["user_instruction"] = config.get("user_instruction", "")
    context["extra_context"] = config.get("extra_context", "")

    # Merge extras (transcript, output_path, span, etc.)
    context.update(extras)

    return context


def build_hook_context(config: dict, **extras: Any) -> HookContext:
    """Build HookContext from config and extra values."""
    context: HookContext = _build_base_context(config)

    # Merge extras (transcript, output_path, span, etc.)
    context.update(extras)

    return context


def run_pre_hook(
    context: PreHookContext,
    hook_fn: Callable[[PreHookContext], dict | None],
) -> dict | None:
    """Execute pre-processing hook and return modifications dict.

    Hook errors are logged and return None (graceful degradation).
    """
    try:
        modifications = hook_fn(context)
        if modifications is not None:
            logging.info(
                "Pre-hook returned modifications: %s", list(modifications.keys())
            )
            return modifications
    except Exception as exc:
        logging.error("Pre-hook failed: %s", exc)

    return None


def run_post_hook(
    result: str,
    context: HookContext,
    hook_fn: Callable[[str, HookContext], str | None],
) -> str:
    """Execute post-processing hook and return (potentially transformed) result.

    Args:
        result: The LLM-generated output text
        context: Hook context with metadata
        hook_fn: The post_process function to call

    Returns:
        Transformed result if hook returns string, original result otherwise.
    """
    try:
        hook_result = hook_fn(result, context)
        if hook_result is not None:
            logging.info("Hook transformed result")
            return hook_result
    except Exception as exc:
        logging.error("Hook failed: %s", exc)

    return result


__all__ = [
    # Re-exported from think.providers.shared
    "Event",
    "GenerateResult",
    # Local definitions
    "HookContext",
    "PreHookContext",
    "InputContext",
    "JSONEventWriter",
    "format_tool_summary",
    "parse_agent_events_to_turns",
    "load_post_hook",
    "load_pre_hook",
    "build_hook_context",
    "build_pre_hook_context",
    "run_post_hook",
    "run_pre_hook",
    "assemble_inputs",
    "scan_day",
    "generate_agent_output",
    "hydrate_config",
    "expand_tools",
    "validate_config",
]


# =============================================================================
# Config Hydration and Validation (moved from cortex.py)
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
    from think.utils import get_agent, key_to_context

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


# =============================================================================
# Unified Input Assembly (shared by generate and tools paths)
# =============================================================================

# Minimum content length for transcript-based generation
MIN_INPUT_CHARS = 50


@dataclass
class InputContext:
    """Assembled inputs for generation or tool execution.

    Contains all resolved inputs ready for LLM call, including transcript,
    prompts, and output path. Used by generate path always, and by tools path
    when day is specified for transcript loading.
    """

    # Transcript (from day/segment/span clustering)
    transcript: str
    source_counts: dict[str, int]

    # Prompts
    prompt: str  # Final prompt (with template substitution)
    system_instruction: str
    system_prompt_name: str  # For cache key construction

    # Output
    output_path: Optional[Path]
    output_format: Optional[str]  # 'md' or 'json'

    # Metadata for hooks and logging
    meta: dict  # Agent config metadata
    agent_path: Optional[Path]  # Path to agent .md file

    # Skip reason (if should skip execution)
    skip_reason: Optional[str]

    # Day/segment context
    day: Optional[str]
    segment: Optional[str]
    span_mode: bool


def assemble_inputs(config: dict) -> InputContext:
    """Assemble all inputs for generation or tool execution.

    Handles:
    - Loading agent config metadata
    - Transcript loading from journal (if day specified)
    - Source filtering and required source validation
    - Minimum content checks
    - Prompt template substitution
    - System instruction composition
    - Output path resolution

    Args:
        config: Hydrated config dict from cortex

    Returns:
        InputContext with all resolved inputs, or skip_reason if should skip
    """
    name = config.get("name", "default")
    day = config.get("day")
    segment = config.get("segment")
    span = config.get("span")  # List of sequential segment keys
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

    # Check if config is disabled
    if meta.get("disabled"):
        return InputContext(
            transcript="",
            source_counts={},
            prompt=user_prompt,
            system_instruction="",
            system_prompt_name="journal",
            output_path=None,
            output_format=output_format,
            meta=meta,
            agent_path=agent_path,
            skip_reason="disabled",
            day=day,
            segment=segment,
            span_mode=bool(span),
        )

    # Extract instructions config for source filtering and system prompt
    instructions_config = meta.get("instructions")
    instructions = compose_instructions(
        include_datetime=not day,
        config_overrides=instructions_config,
    )
    sources = instructions.get("sources", {})
    system_prompt_name = instructions.get("system_prompt_name", "journal")
    system_instruction = instructions["system_instruction"]

    # Append extra_context (facets, etc.) to system instruction if present
    extra_context = instructions.get("extra_context")
    if extra_context:
        system_instruction = f"{system_instruction}\n\n{extra_context}"

    # Track span mode
    span_mode = bool(span)

    # Initialize transcript variables
    transcript = ""
    source_counts: dict[str, int] = {}

    # Transcript loading (only if day is provided)
    if day:
        # Set segment key for token usage logging
        if segment:
            os.environ["SEGMENT_KEY"] = segment
        elif span:
            os.environ["SEGMENT_KEY"] = span[0]

        # Convert sources for clustering
        cluster_sources = {k: source_is_enabled(v) for k, v in sources.items()}

        # Build transcript via clustering
        if span:
            transcript, source_counts = cluster_span(day, span, sources=cluster_sources)
        elif segment:
            transcript, source_counts = cluster_period(
                day, segment, sources=cluster_sources
            )
        else:
            transcript, source_counts = cluster(day, sources=cluster_sources)

        total_count = sum(source_counts.values())

        # Check required sources have content
        for source_type, mode in sources.items():
            if source_is_required(mode) and source_counts.get(source_type, 0) == 0:
                return InputContext(
                    transcript=transcript,
                    source_counts=source_counts,
                    prompt=user_prompt,
                    system_instruction=system_instruction,
                    system_prompt_name=system_prompt_name,
                    output_path=None,
                    output_format=output_format,
                    meta=meta,
                    agent_path=agent_path,
                    skip_reason=f"missing_required_{source_type}",
                    day=day,
                    segment=segment,
                    span_mode=span_mode,
                )

        # Skip when there's nothing to analyze
        if total_count == 0 or len(transcript.strip()) < MIN_INPUT_CHARS:
            return InputContext(
                transcript=transcript,
                source_counts=source_counts,
                prompt=user_prompt,
                system_instruction=system_instruction,
                system_prompt_name=system_prompt_name,
                output_path=None,
                output_format=output_format,
                meta=meta,
                agent_path=agent_path,
                skip_reason="no_input",
                day=day,
                segment=segment,
                span_mode=span_mode,
            )

        # Prepend input context note for limited recordings
        if total_count < 3:
            input_note = (
                "**Input Note:** Limited recordings for this day. "
                "Scale analysis to available input.\n\n"
            )
            transcript = input_note + transcript

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
    else:
        prompt = user_prompt

    # Append user prompt if both agent prompt and user prompt exist
    if agent_path and user_prompt and prompt != user_prompt:
        prompt = f"{prompt}\n\n{user_prompt}"

    # Determine output path
    output_path: Optional[Path] = None
    if output_format:
        if output_path_override:
            output_path = Path(output_path_override)
        elif day:
            day_dir = str(day_path(day))
            output_path = get_output_path(
                day_dir, name, segment=segment, output_format=output_format
            )

    return InputContext(
        transcript=transcript,
        source_counts=source_counts,
        prompt=prompt,
        system_instruction=system_instruction,
        system_prompt_name=system_prompt_name,
        output_path=output_path,
        output_format=output_format,
        meta=meta,
        agent_path=agent_path,
        skip_reason=None,
        day=day,
        segment=segment,
        span_mode=span_mode,
    )


# =============================================================================
# Unified Execution Helpers (shared by generate and tools paths)
# =============================================================================


def _emit_start_event(
    emit_event: Callable[[dict], None],
    name: str,
    model: str,
    provider: str,
    prompt: str,
    continue_from: Optional[str] = None,
) -> None:
    """Emit a unified start event for both generate and tools paths."""
    start_event: dict[str, Any] = {
        "event": "start",
        "ts": now_ms(),
        "prompt": prompt,
        "name": name,
        "model": model or "unknown",
        "provider": provider,
    }
    if continue_from:
        start_event["continue_from"] = continue_from
    emit_event(start_event)


def _handle_skip(
    inputs: InputContext,
    name: str,
    path_type: str,
    emit_event: Callable[[dict], None],
) -> bool:
    """Handle skip conditions from input assembly.

    Args:
        inputs: InputContext with skip_reason set
        name: Agent/generator name
        path_type: "generate" or "agent" for logging
        emit_event: Event emitter callback

    Returns:
        True if skipped and caller should return/continue, False otherwise
    """
    if not inputs.skip_reason:
        return False

    logging.info("Config %s skipped: %s", name, inputs.skip_reason)
    emit_event(
        {
            "event": "finish",
            "ts": now_ms(),
            "result": "",
            "skipped": inputs.skip_reason,
        }
    )
    if inputs.day:
        day_log(inputs.day, f"{path_type} {name} skipped ({inputs.skip_reason})")
    return True


def _execute_pre_hooks(
    meta: dict,
    modifiable: dict[str, str],
    output_path: Optional[Path] = None,
    day: Optional[str] = None,
    segment: Optional[str] = None,
    span_mode: bool = False,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Execute pre-processing hooks and return modified values.

    Args:
        meta: Agent metadata containing hook config
        modifiable: Dict of modifiable field values (prompt, system_instruction,
            transcript, etc.) - these are passed to the hook context
        output_path: Output path for context
        day: Day string for context
        segment: Segment string for context
        span_mode: Whether in span mode

    Returns:
        Tuple of (modified_values, hook_info for dry-run)
        modified_values contains only fields that were modified
        hook_info contains name and list of modifications for dry-run display
    """
    hook_info: dict[str, Any] = {}
    modified: dict[str, str] = {}

    pre_hook = load_pre_hook(meta)
    if not pre_hook:
        return modified, hook_info

    # Get hook name for logging
    hook_config = meta.get("hook", {})
    hook_name = hook_config.get("pre") if isinstance(hook_config, dict) else None
    hook_info["name"] = hook_name

    # Build context with all modifiable fields
    # Note: modifiable may contain transcript, so we don't pass it separately
    pre_context = build_pre_hook_context(
        meta,
        output_path=str(output_path) if output_path else "",
        day=day,
        segment=segment,
        span=span_mode,
        **modifiable,
    )

    modifications = run_pre_hook(pre_context, pre_hook)
    if modifications:
        # Only include fields that were actually modified
        for key in modifiable:
            if key in modifications:
                modified[key] = modifications[key]
        hook_info["modifications"] = list(modifications.keys())

    return modified, hook_info


def _build_dry_run_event(
    run_type: str,
    name: str,
    provider: str,
    model: str,
    config: dict,
    inputs: Optional[InputContext],
    hook_info: dict[str, Any],
    before_values: dict[str, str],
    current_values: dict[str, str],
) -> dict[str, Any]:
    """Build a dry-run event with all context.

    Args:
        run_type: "generate" or "agent"
        name: Agent/generator name
        provider: Provider name
        model: Model name
        config: Full config dict
        inputs: InputContext if available
        hook_info: Pre-hook info from _execute_pre_hooks
        before_values: Values before hook execution
        current_values: Values after hook execution

    Returns:
        Complete dry-run event dict
    """
    event: dict[str, Any] = {
        "event": "dry_run",
        "ts": now_ms(),
        "type": run_type,
        "name": name,
        "provider": provider,
        "model": model or "unknown",
        "system_instruction": current_values.get("system_instruction", ""),
        "prompt": current_values.get("prompt", ""),
    }

    # Add agent-specific fields
    if run_type == "agent":
        event["user_instruction"] = current_values.get("user_instruction", "")
        event["extra_context"] = current_values.get("extra_context", "")
        event["tools"] = config.get("tools", [])

    # Add generate-specific fields
    if run_type == "generate":
        event["system_instruction_source"] = (
            inputs.system_prompt_name if inputs else "journal"
        )
        event["prompt_source"] = (
            str(inputs.agent_path) if inputs and inputs.agent_path else "request"
        )

    # Add day-based fields if inputs available
    if inputs:
        event["day"] = inputs.day
        event["segment"] = inputs.segment
        transcript = current_values.get("transcript", "")
        if transcript:
            event["transcript"] = transcript
            event["transcript_chars"] = len(transcript)
            event["transcript_files"] = sum(inputs.source_counts.values())
        if inputs.output_path:
            event["output_path"] = str(inputs.output_path)

    # Add hook before/after info
    if hook_info:
        event["pre_hook"] = hook_info.get("name")
        event["pre_hook_modifications"] = hook_info.get("modifications", [])
        # Include before values for modified fields
        for key, before_val in before_values.items():
            current_val = current_values.get(key, "")
            if current_val != before_val:
                if key == "transcript":
                    event["transcript_before"] = before_val
                    event["transcript_before_chars"] = len(before_val)
                else:
                    event[f"{key}_before"] = before_val

    return event


def _write_output(output_path: Path, result: str, output_format: str) -> None:
    """Write result to output file.

    Args:
        output_path: Path to write to
        result: Content to write
        output_format: Format type ('md' or 'json')
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)
    logging.info("Wrote output to %s", output_path)


def _execute_post_hooks(
    result: str,
    meta: dict,
    transcript: str = "",
    output_path: Optional[Path] = None,
    day: Optional[str] = None,
    segment: Optional[str] = None,
    span_mode: bool = False,
    name: str = "",
) -> str:
    """Execute post-processing hooks and return transformed result.

    Args:
        result: LLM output text
        meta: Agent metadata containing hook config
        transcript: Transcript for context
        output_path: Output path for context
        day: Day for context
        segment: Segment for context
        span_mode: Span mode flag
        name: Agent name

    Returns:
        Transformed result (or original if no hook or hook returns None)
    """
    post_hook = load_post_hook(meta)
    if not post_hook:
        return result

    hook_context = build_hook_context(
        meta,
        name=name,
        day=day,
        segment=segment,
        span=span_mode,
        output_path=str(output_path) if output_path else "",
        transcript=transcript,
    )
    return run_post_hook(result, hook_context, post_hook)


# =============================================================================
# Generator Functions (for transcript analysis without tools)
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
    from think.utils import key_to_context

    context = key_to_context(name) if name else "muse.system.unknown"

    # Try to use cache if display name provided
    # Note: caching is Google-specific, so we check provider first
    provider, model = resolve_provider(context)

    client = None
    cache_name = None
    if cache_display_name and provider == "google":
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions()),
        )
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


def _run_generate(
    config: dict,
    emit_event: Callable[[dict], None],
    *,
    dry_run: bool = False,
    inputs: InputContext | None = None,
) -> None:
    """Execute single-shot generation with optional features based on config.

    This is the generation path for non-tool requests. Uses assemble_inputs() for
    input assembly, which can be shared with the tools path.

    Args:
        config: Merged config from cortex
        emit_event: Callback to emit JSONL events
        dry_run: If True, emit dry_run event instead of calling LLM
        inputs: Pre-assembled InputContext (if None, will call assemble_inputs)
    """
    name = config.get("name", "default")
    force = config.get("force", False)
    provider = config.get("provider", "google")
    model = config.get("model")
    user_prompt = config.get("prompt", "")

    # Assemble inputs first (before start event, so we can skip cleanly)
    if inputs is None:
        inputs = assemble_inputs(config)

    # Emit unified start event
    _emit_start_event(emit_event, name, model, provider, user_prompt)

    # Handle skip conditions using helper
    if _handle_skip(inputs, name, "generate", emit_event):
        return

    # Extract values from InputContext
    day = inputs.day
    segment = inputs.segment
    span_mode = inputs.span_mode
    transcript = inputs.transcript
    prompt = inputs.prompt
    system_instruction = inputs.system_instruction
    system_prompt_name = inputs.system_prompt_name
    output_path = inputs.output_path
    output_format = inputs.output_format
    meta = inputs.meta

    # Check if output exists
    output_exists = False
    is_json_output = output_format == "json"
    if output_path:
        output_exists = output_path.exists() and output_path.stat().st_size > 0

    # Determine cache settings (only for day-based, non-span requests)
    cache_display_name = None
    if day and not span_mode:
        if segment:
            cache_display_name = f"{system_prompt_name}_{day}_{segment}"
        else:
            cache_display_name = f"{system_prompt_name}_{day}"

    # Extract generation parameters from metadata
    meta_thinking_budget = meta.get("thinking_budget")
    meta_max_output_tokens = meta.get("max_output_tokens")

    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY", "")

    usage_data = None

    # Dry-run always goes through prompt assembly, regardless of existing output
    if output_exists and not force and not dry_run:
        # Load existing content (no LLM call)
        logging.info("Output exists, loading: %s", output_path)
        with open(output_path, "r") as f:
            result = f.read()
    else:
        # Generate new content
        if output_exists and force:
            logging.info("Force regenerating: %s", output_path)

        # Capture state before pre-hook for dry-run comparison
        before_values = {
            "transcript": transcript,
            "prompt": prompt,
            "system_instruction": system_instruction,
        }

        # Run pre-processing hooks using helper
        modifications, hook_info = _execute_pre_hooks(
            meta,
            modifiable={
                "prompt": prompt,
                "system_instruction": system_instruction,
                "transcript": transcript,
            },
            output_path=output_path,
            day=day,
            segment=segment,
            span_mode=span_mode,
        )

        # Apply modifications
        transcript = modifications.get("transcript", transcript)
        prompt = modifications.get("prompt", prompt)
        system_instruction = modifications.get("system_instruction", system_instruction)

        # Current values after hook
        current_values = {
            "transcript": transcript,
            "prompt": prompt,
            "system_instruction": system_instruction,
        }

        # Dry-run mode: emit context and return without LLM call
        if dry_run:
            dry_run_event = _build_dry_run_event(
                "generate",
                name,
                provider,
                model,
                config,
                inputs,
                hook_info,
                before_values,
                current_values,
            )
            emit_event(dry_run_event)
            return

        gen_result = generate_agent_output(
            transcript,
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

        # Run post-processing hooks using helper
        result = _execute_post_hooks(
            result,
            meta,
            transcript=transcript,
            output_path=output_path,
            day=day,
            segment=segment,
            span_mode=span_mode,
            name=name,
        )

    # Write output file (agents.py owns output writing)
    if output_path and result:
        _write_output(output_path, result, output_format or "md")

    # Emit finish event with result
    finish_event: dict[str, Any] = {
        "event": "finish",
        "ts": now_ms(),
        "result": result,
    }
    if usage_data:
        finish_event["usage"] = usage_data
    # Include handoff config for cortex to spawn follow-up agent
    if config.get("handoff"):
        finish_event["handoff"] = config["handoff"]

    emit_event(finish_event)

    # Log completion (only for day-based requests)
    if day:
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
    - 'output' field present (no 'tools') -> generator (transcript analysis)
    - Everything else -> agent (with or without tools, via provider)
    """
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

    # Always write to stdout only
    event_writer = JSONEventWriter(None)

    def emit_event(data: Event) -> None:
        if "ts" not in data:
            data["ts"] = now_ms()
        event_writer.emit(data)

    try:
        # NDJSON input mode from stdin only
        app_logger.info("Processing NDJSON input from stdin")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                # Parse NDJSON line - raw request from cortex
                request = json.loads(line)

                # Hydrate config: load agent definition, merge request, resolve provider
                config = hydrate_config(request)

                # Validate config
                error = validate_config(config)
                if error:
                    emit_event({"event": "error", "error": error, "ts": now_ms()})
                    continue

                # Route based on config type: tools → run_tools, else → run_generate
                has_tools = bool(config.get("tools"))

                if not has_tools:
                    # Generate path: single-shot generation with opt-in features
                    app_logger.debug(f"Processing generate: {config.get('name')}")
                    _run_generate(config, emit_event, dry_run=dry_run)

                else:
                    # Agent: with or without tools (conversational or tool-using)
                    # Extract provider to route to correct module
                    from .providers import PROVIDER_REGISTRY, get_provider_module

                    provider = config.get("provider", "google")
                    name = config.get("name", "default")
                    model = config.get("model")

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

                    # Assemble inputs if day is specified (transcript loading)
                    inputs: InputContext | None = None
                    if config.get("day"):
                        inputs = assemble_inputs(config)

                    # Get metadata for hooks (from inputs if available, else from config)
                    meta = inputs.meta if inputs else config

                    # Emit unified start event (agents.py owns this)
                    _emit_start_event(
                        emit_event,
                        name,
                        model,
                        provider,
                        config.get("prompt", ""),
                        continue_from=config.get("continue_from"),
                    )

                    # Handle skip conditions using helper
                    if inputs and _handle_skip(inputs, name, "agent", emit_event):
                        continue

                    # Pass transcript and system instruction to provider if inputs assembled
                    if inputs and not config.get("continue_from"):
                        # Warn if both day and continue_from are specified
                        if config.get("continue_from"):
                            logging.warning(
                                "Both 'day' and 'continue_from' specified; "
                                "continue_from takes precedence, transcript ignored"
                            )
                        else:
                            config["transcript"] = inputs.transcript
                            if not config.get("system_instruction"):
                                config["system_instruction"] = inputs.system_instruction

                    # Capture state before pre-hook for dry-run comparison
                    before_values = {
                        "prompt": config.get("prompt", ""),
                        "system_instruction": config.get("system_instruction", ""),
                        "user_instruction": config.get("user_instruction", ""),
                        "extra_context": config.get("extra_context", ""),
                        "transcript": config.get("transcript", ""),
                    }

                    # Run pre-processing hooks using helper
                    # Note: before_values already contains transcript
                    modifications, hook_info = _execute_pre_hooks(
                        meta,
                        modifiable=before_values.copy(),
                        output_path=inputs.output_path if inputs else None,
                        day=inputs.day if inputs else None,
                        segment=inputs.segment if inputs else None,
                        span_mode=inputs.span_mode if inputs else False,
                    )

                    # Apply modifications to config
                    for key in (
                        "prompt",
                        "system_instruction",
                        "user_instruction",
                        "extra_context",
                        "transcript",
                    ):
                        if key in modifications:
                            config[key] = modifications[key]

                    # Current values after hook
                    current_values = {
                        "prompt": config.get("prompt", ""),
                        "system_instruction": config.get("system_instruction", ""),
                        "user_instruction": config.get("user_instruction", ""),
                        "extra_context": config.get("extra_context", ""),
                        "transcript": config.get("transcript", ""),
                    }

                    # Dry-run mode: emit context and return without LLM call
                    if dry_run:
                        dry_run_event = _build_dry_run_event(
                            "agent",
                            name,
                            provider,
                            model,
                            config,
                            inputs,
                            hook_info,
                            before_values,
                            current_values,
                        )
                        emit_event(dry_run_event)
                        continue

                    handoff_config = config.get("handoff")
                    output_path = inputs.output_path if inputs else None
                    output_format = inputs.output_format if inputs else None

                    # Create event handler that intercepts finish for post-hooks,
                    # output writing, and handoff
                    def agent_emit_event(data: Event) -> None:
                        if data.get("event") == "finish":
                            result = data.get("result", "")

                            # Apply post-processing hooks using helper
                            result = _execute_post_hooks(
                                result,
                                meta,
                                transcript=config.get("transcript", ""),
                                output_path=output_path,
                                day=inputs.day if inputs else None,
                                segment=inputs.segment if inputs else None,
                                span_mode=inputs.span_mode if inputs else False,
                                name=name,
                            )

                            # Update data if result was transformed
                            if result != data.get("result", ""):
                                data = {**data, "result": result}

                            # Write output file (agents.py owns output writing)
                            if output_path and result:
                                _write_output(
                                    output_path, result, output_format or "md"
                                )

                            # Include handoff config for cortex
                            if handoff_config:
                                data = {**data, "handoff": handoff_config}

                        # Filter out start events from providers (we already emitted ours)
                        if data.get("event") == "start":
                            return

                        emit_event(data)

                    # Pass complete config to provider
                    await provider_mod.run_tools(
                        config=config,
                        on_event=agent_emit_event,
                    )

                    # Log completion for day-based requests
                    if inputs and inputs.day:
                        day_log(inputs.day, f"agent {name} ok")

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
