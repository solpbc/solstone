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
import traceback
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
    "JSONEventWriter",
    "format_tool_summary",
    "parse_agent_events_to_turns",
    "load_post_hook",
    "load_pre_hook",
    "build_hook_context",
    "build_pre_hook_context",
    "run_post_hook",
    "run_pre_hook",
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


def _run_generator(
    config: dict, emit_event: Callable[[dict], None], *, dry_run: bool = False
) -> None:
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
        dry_run: If True, emit dry_run event instead of calling LLM
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
            "ts": now_ms(),
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
                "ts": now_ms(),
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

    # Append extra_context (facets, etc.) to system instruction if present
    extra_context = instructions.get("extra_context")
    if extra_context:
        system_instruction = f"{system_instruction}\n\n{extra_context}"

    # Track span mode (multiple sequential segments)
    span_mode = bool(span)

    # Convert sources for clustering (both True and "required" mean load)
    cluster_sources = {k: source_is_enabled(v) for k, v in sources.items()}

    # Build transcript via clustering
    if span:
        markdown, source_counts = cluster_span(day, span, sources=cluster_sources)
    elif segment:
        markdown, source_counts = cluster_period(day, segment, sources=cluster_sources)
    else:
        markdown, source_counts = cluster(day, sources=cluster_sources)

    day_dir = str(day_path(day))
    total_count = sum(source_counts.values())

    # Check required sources have content
    for source_type, mode in sources.items():
        if source_is_required(mode) and source_counts.get(source_type, 0) == 0:
            logging.info(
                "Required source '%s' has no content, skipping",
                source_type,
            )
            emit_event(
                {
                    "event": "finish",
                    "ts": now_ms(),
                    "result": "",
                    "skipped": f"missing_required_{source_type}",
                }
            )
            day_log(day, f"generate {name} skipped (no {source_type})")
            return

    # Skip generation when there's nothing to analyze
    if total_count == 0 or len(markdown.strip()) < MIN_INPUT_CHARS:
        logging.info(
            "Insufficient input (files=%d, chars=%d), skipping",
            total_count,
            len(markdown.strip()),
        )
        emit_event(
            {
                "event": "finish",
                "ts": now_ms(),
                "result": "",
                "skipped": "no_input",
            }
        )
        day_log(day, f"generate {name} skipped (no input)")
        return

    # Prepend input context note for limited recordings
    if total_count < 3:
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
        pre_hook_info: dict[str, Any] = {}
        before_transcript = markdown
        before_prompt = prompt
        before_system = system_instruction

        # Run pre-processing hook if present (before LLM call)
        pre_hook = load_pre_hook(meta)
        if pre_hook:
            hook_config = meta.get("hook", {})
            pre_hook_name = (
                hook_config.get("pre") if isinstance(hook_config, dict) else None
            )
            pre_hook_info["name"] = pre_hook_name

            pre_context = build_pre_hook_context(
                meta,
                name=name,
                day=day,
                segment=segment,
                span=span_mode,
                output_path=str(output_path),
                transcript=markdown,
                prompt=prompt,
                system_instruction=system_instruction,
            )
            modifications = run_pre_hook(pre_context, pre_hook)
            if modifications:
                # Apply modifications to inputs
                markdown = modifications.get("transcript", markdown)
                prompt = modifications.get("prompt", prompt)
                system_instruction = modifications.get(
                    "system_instruction", system_instruction
                )
                # Track what was modified
                pre_hook_info["modifications"] = list(modifications.keys())

        # Dry-run mode: emit context and return without LLM call
        if dry_run:
            dry_run_event: dict[str, Any] = {
                "event": "dry_run",
                "ts": now_ms(),
                "type": "generator",
                "name": name,
                "provider": provider,
                "model": model or "unknown",
                "day": day,
                "segment": segment,
                "system_instruction": system_instruction,
                "system_instruction_source": system_prompt_name,
                "prompt": prompt,
                "prompt_source": str(agent_path),
                "transcript": markdown,
                "transcript_chars": len(markdown),
                "transcript_files": total_count,
                "output_path": str(output_path),
            }
            # Include pre-hook before/after if hook was run
            if pre_hook_info:
                dry_run_event["pre_hook"] = pre_hook_info.get("name")
                dry_run_event["pre_hook_modifications"] = pre_hook_info.get(
                    "modifications", []
                )
                # Include before values if they changed
                if markdown != before_transcript:
                    dry_run_event["transcript_before"] = before_transcript
                    dry_run_event["transcript_before_chars"] = len(before_transcript)
                if prompt != before_prompt:
                    dry_run_event["prompt_before"] = before_prompt
                if system_instruction != before_system:
                    dry_run_event["system_instruction_before"] = before_system

            emit_event(dry_run_event)
            return

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
        post_hook = load_post_hook(meta)
        if post_hook:
            hook_context = build_hook_context(
                meta,
                name=name,
                day=day,
                segment=segment,
                span=span_mode,
                output_path=str(output_path),
                transcript=markdown,
            )
            result = run_post_hook(result, hook_context, post_hook)

    # Emit finish event with result (cortex handles file writing)
    finish_event = {
        "event": "finish",
        "ts": now_ms(),
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
                # Parse NDJSON line - this is the complete merged config from Cortex
                config = json.loads(line)

                # Route based on config type
                has_tools = bool(config.get("tools"))
                has_output = bool(config.get("output"))

                if has_output and not has_tools:
                    # Generator: transcript analysis without tools
                    app_logger.debug(f"Processing generator: {config.get('name')}")
                    _run_generator(config, emit_event, dry_run=dry_run)

                elif has_tools:
                    # Tool-using agent: validate prompt exists
                    prompt = config.get("prompt")
                    if not prompt:
                        emit_event(
                            {
                                "event": "error",
                                "error": "Missing 'prompt' field for tool agent",
                                "ts": now_ms(),
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

                    # Capture state before pre-hook for dry-run comparison
                    pre_hook_info: dict[str, Any] = {}
                    before_prompt = config.get("prompt", "")
                    before_system = config.get("system_instruction", "")
                    before_user = config.get("user_instruction", "")
                    before_extra = config.get("extra_context", "")

                    # Load pre hook if configured (before LLM call)
                    pre_hook = load_pre_hook(config)
                    if pre_hook:
                        hook_config = config.get("hook", {})
                        pre_hook_name = (
                            hook_config.get("pre")
                            if isinstance(hook_config, dict)
                            else None
                        )
                        pre_hook_info["name"] = pre_hook_name

                        pre_context = build_pre_hook_context(config)
                        modifications = run_pre_hook(pre_context, pre_hook)
                        if modifications:
                            # Apply modifications to config
                            for key in (
                                "prompt",
                                "system_instruction",
                                "user_instruction",
                                "extra_context",
                            ):
                                if key in modifications:
                                    config[key] = modifications[key]
                            pre_hook_info["modifications"] = list(modifications.keys())

                    # Dry-run mode: emit context and return without LLM call
                    if dry_run:
                        dry_run_event: dict[str, Any] = {
                            "event": "dry_run",
                            "ts": now_ms(),
                            "type": "agent",
                            "name": config.get("name", "default"),
                            "provider": provider,
                            "model": config.get("model", "unknown"),
                            "system_instruction": config.get("system_instruction", ""),
                            "system_instruction_source": config.get(
                                "system_prompt_name", "journal"
                            ),
                            "user_instruction": config.get("user_instruction", ""),
                            "extra_context": config.get("extra_context", ""),
                            "prompt": config.get("prompt", ""),
                            "tools": config.get("tools", []),
                        }
                        # Include pre-hook before/after if hook was run
                        if pre_hook_info:
                            dry_run_event["pre_hook"] = pre_hook_info.get("name")
                            dry_run_event["pre_hook_modifications"] = pre_hook_info.get(
                                "modifications", []
                            )
                            if config.get("prompt") != before_prompt:
                                dry_run_event["prompt_before"] = before_prompt
                            if config.get("system_instruction") != before_system:
                                dry_run_event["system_instruction_before"] = (
                                    before_system
                                )
                            if config.get("user_instruction") != before_user:
                                dry_run_event["user_instruction_before"] = before_user
                            if config.get("extra_context") != before_extra:
                                dry_run_event["extra_context_before"] = before_extra

                        emit_event(dry_run_event)
                        continue

                    # Load post hook if configured
                    post_hook = load_post_hook(config)

                    # Create event handler that intercepts finish for hooks
                    def agent_emit_event(data: Event) -> None:
                        if post_hook and data.get("event") == "finish":
                            result = data.get("result", "")
                            hook_context = build_hook_context(config)
                            transformed = run_post_hook(result, hook_context, post_hook)
                            if transformed != result:
                                data = {**data, "result": transformed}
                        emit_event(data)

                    # Pass complete config to provider
                    await provider_mod.run_agent(
                        config=config,
                        on_event=agent_emit_event,
                    )

                else:
                    # Neither tools nor output - invalid config
                    emit_event(
                        {
                            "event": "error",
                            "error": "Invalid config: must have 'tools' or 'output' field",
                            "ts": now_ms(),
                        }
                    )

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
