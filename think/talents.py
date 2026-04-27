# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unified talent execution module for solstone.

Spawned by cortex for all talent types:
- Tool-using talents (with configured tools)
- Generators (transcript analysis, no tools)

Both paths share unified config preparation and execution flow.
Reads NDJSON config from stdin, emits JSONL events to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import errno
import json
import logging
import os
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any, Callable, Optional

from think.cluster import cluster, cluster_period, cluster_span
from think.providers.shared import Event
from think.talent import (
    get_output_path,
    get_talent_configs,
    get_talent_filter,
    load_post_hook,
    load_pre_hook,
    load_prompt,
    source_is_enabled,
    source_is_required,
)
from think.utils import (
    day_log,
    day_path,
    format_day,
    format_segment_times,
    get_journal,
    get_project_root,
    now_ms,
    require_solstone,
    segment_parse,
    setup_cli,
)

TALENT_EXECUTION_MODULE = "think.talents"

LOG = logging.getLogger("think.talents")

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
        self._pipe_dead = False
        if path:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                self.file = open(path, "a", encoding="utf-8")
            except OSError as exc:
                LOG.warning("Failed to open JSON event sidecar %s: %s", path, exc)

    def emit(self, data: Event) -> None:
        line = json.dumps(data, ensure_ascii=False)
        if not self._pipe_dead:
            try:
                print(line)
                sys.stdout.flush()  # Ensure immediate output for cortex
            except (BrokenPipeError, OSError) as exc:
                if not isinstance(exc, BrokenPipeError) and exc.errno != errno.EPIPE:
                    raise
                self._pipe_dead = True
        if self.file:
            try:
                self.file.write(line + "\n")
                self.file.flush()
            except OSError as exc:
                LOG.warning("Failed to write JSON event sidecar %s: %s", self.path, exc)

    def close(self) -> None:
        if self.file:
            try:
                self.file.close()
            except OSError as exc:
                LOG.warning("Failed to close JSON event sidecar %s: %s", self.path, exc)


# =============================================================================
# Unified Config Preparation
# =============================================================================


def _stream_content_description(stream: str | None) -> str:
    """Return a human-readable content description for a stream.

    Used in preamble templates so talents know what kind of content they're
    analyzing (live capture vs imported conversations, notes, etc.).
    """
    if not stream:
        return "audio transcription and screen recording"

    STREAM_DESCRIPTIONS = {
        "archon": "audio transcription and screen recording",
        "import.chatgpt": "an imported ChatGPT conversation",
        "import.claude": "an imported Claude conversation",
        "import.gemini": "an imported Gemini conversation",
        "import.ics": "an imported calendar event",
        "import.obsidian": "an imported note from Obsidian",
        "import.document": "an imported document (PDF)",
        "import.kindle": "imported Kindle reading highlights",
    }

    if stream in STREAM_DESCRIPTIONS:
        return STREAM_DESCRIPTIONS[stream]

    # Fallback for unknown import streams
    if stream.startswith("import."):
        source = stream.split(".", 1)[1]
        return f"imported content from {source}"

    return "captured content"


def _stream_import_guidance(stream: str | None) -> str:
    """Return stream-conditional guidance for the activity agent.

    For live capture, returns guidance about frame comparison and spoken audio.
    For imports, returns content-type-specific analysis instructions.
    Returns empty string for unknown streams.
    """
    if not stream or stream == "archon":
        return (
            "## Live Capture Guidance\n\n"
            "ONLY report what CHANGED between screenshots or was SPOKEN in audio. "
            "If content looks the same across frames, skip it entirely.\n\n"
            "### Your Inputs\n\n"
            "- **Screenshots**: Sampled across this segment. Compare frames — what's different?\n"
            "- **Audio**: Transcript of speech. What was said?\n\n"
            "### SKIP Entirely\n\n"
            "- Windows that look identical in first and last frame\n"
            "- Apps open but showing same content throughout\n"
            "- Background windows never brought to focus\n"
            '- Anything you\'d describe as "had open" or "was visible"'
        )

    IMPORT_GUIDANCE = {
        "import.chatgpt": (
            "This is an AI conversation. Summarize the key topics discussed, "
            "questions asked, solutions proposed, and decisions reached. "
            "Focus on what the human was trying to accomplish and what they learned or decided."
        ),
        "import.claude": (
            "This is an AI conversation. Summarize the key topics discussed, "
            "questions asked, solutions proposed, and decisions reached. "
            "Focus on what the human was trying to accomplish and what they learned or decided."
        ),
        "import.gemini": (
            "This is an AI conversation. Summarize the key topics discussed, "
            "questions asked, solutions proposed, and decisions reached. "
            "Focus on what the human was trying to accomplish and what they learned or decided."
        ),
        "import.ics": (
            "This is a calendar event. Describe the event: its purpose, "
            "participants, and any context from the description about why it was scheduled."
        ),
        "import.obsidian": (
            "This is a note. Summarize the key ideas, references, and connections. "
            "What was the author thinking about and working through?"
        ),
        "import.document": (
            "This is an imported document (legal, financial, medical, or personal). "
            "Extract all named parties and their roles (grantor, trustee, beneficiary, "
            "attorney, witness, agent, etc.). Produce a plain-language summary that a "
            "non-expert could understand. Identify key provisions, dates, conditions, "
            "obligations, and deadlines. Note any time-sensitive requirements (renewal "
            "dates, filing deadlines, review periods)."
        ),
        "import.kindle": (
            "These are reading highlights. Describe what was being read and what "
            "the reader found noteworthy. What themes or ideas do these highlights capture?"
        ),
    }

    if stream in IMPORT_GUIDANCE:
        return f"## Content Guidance\n\n{IMPORT_GUIDANCE[stream]}"

    if stream.startswith("import."):
        return (
            "## Content Guidance\n\n"
            "This is imported content. Summarize the key topics, actions, "
            "and takeaways present in this segment."
        )

    return ""


def _build_prompt_context(
    day: str | None,
    segment: str | None,
    span: list[str] | None,
    activity: dict | None = None,
    facet: str | None = None,
) -> dict[str, str]:
    """Build context dict for prompt template substitution.

    Args:
        day: Day in YYYYMMDD format
        segment: Segment key (HHMMSS_LEN)
        span: List of segment keys
        activity: Optional activity record dict for activity-scheduled talents
        facet: Optional facet name for daily multi-facet talents

    Returns:
        Dict with template variables:
        - day: Friendly format (e.g., "Sunday, February 2, 2025")
        - day_YYYYMMDD: Raw day string (e.g., "20250202")
        - segment_start, segment_end: Time strings if segment/span provided
        - stream, content_description: Stream name and human-readable description
        - activity_*: Activity fields if activity record provided
        - facet, activity_md_dir: Facet name and activity markdown dir for daily runs
    """
    context: dict[str, str] = {}
    if not day:
        return context

    context["day"] = format_day(day)
    context["day_YYYYMMDD"] = day

    # Stream-aware content description and import guidance
    stream = os.environ.get("SOL_STREAM")
    context["stream"] = stream or "archon"
    context["content_description"] = _stream_content_description(stream)
    context["import_guidance"] = _stream_import_guidance(stream)

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

    # Activity template variables
    if activity:
        from think.activities import estimate_duration_minutes

        context["activity_id"] = activity.get("id", "")
        context["activity_type"] = activity.get("activity", "")
        context["activity_description"] = activity.get("description", "")
        context["activity_level"] = str(activity.get("level_avg", 0.5))
        entities = activity.get("active_entities", [])
        context["activity_entities"] = ", ".join(entities) if entities else ""
        segments = activity.get("segments", [])
        context["activity_segments"] = ", ".join(segments) if segments else ""
        context["activity_duration"] = str(estimate_duration_minutes(segments))

    if facet:
        context["facet"] = facet
        try:
            context["activity_md_dir"] = (
                f"{get_journal()}/facets/{facet}/activities/{day}/"
            )
        except Exception:
            LOG.debug(
                "Failed to build activity_md_dir for facet=%s day=%s",
                facet,
                day,
                exc_info=True,
            )

    return context


def _build_activity_context(
    activity: dict,
    span: list[str],
    facet: str,
    day: str,
) -> str | None:
    """Build activity context sections for $activity_context.

    Args:
        activity: Activity record dict (from activity records JSONL)
        span: List of segment keys in the activity's span
        facet: Facet name
        day: Day in YYYYMMDD format

    Returns:
        Formatted string for the $activity_context template variable.
    """
    activity_cfg = {"context": True, "state": True, "focus": True}

    parts: list[str] = []
    activity_type = activity.get("activity", "unknown")

    # --- activity.context: Activity metadata section ---
    if activity_cfg.get("context"):
        from think.activities import estimate_duration_minutes

        level_avg = activity.get("level_avg", 0.5)
        level_label = (
            "high" if level_avg >= 0.75 else "medium" if level_avg >= 0.4 else "low"
        )
        segments = activity.get("segments", [])
        duration = estimate_duration_minutes(segments)
        entities = activity.get("active_entities", [])
        entities_str = ", ".join(entities) if entities else "none detected"

        parts.append(
            f"## Activity Context\n"
            f"- **Type:** {activity_type}\n"
            f"- **Description:** {activity.get('description', '')}\n"
            f"- **Engagement Level:** {level_avg} ({level_label})\n"
            f"- **Duration:** ~{duration} minutes ({len(segments)} segments)\n"
            f"- **Active Entities:** {entities_str}"
        )

    # --- activity.state: Per-segment activity descriptions ---
    if activity_cfg.get("state"):
        from think.activities import load_segment_activity_state

        state_lines: list[str] = []
        for seg in span:
            entry = load_segment_activity_state(day, seg, facet, activity_type)
            if entry:
                level = entry.get("level", "")
                desc = entry.get("description", "")
                # Format segment time for readability
                start_str, end_str = format_segment_times(seg)
                time_label = (
                    f" ({start_str} - {end_str})" if start_str and end_str else ""
                )
                state_lines.append(
                    f"### {seg}{time_label}\n{activity_type} [{level}]: {desc}"
                )

        if state_lines:
            parts.append("## Activity State Per Segment\n\n" + "\n\n".join(state_lines))

    # --- activity.focus: Focusing instructions ---
    if activity_cfg.get("focus"):
        parts.append(
            f"## Analysis Focus\n"
            f"You are analyzing ONLY the **{activity_type}** activity within the "
            f"**{facet}** facet. The transcript segments may contain content from "
            f"other concurrent activities (e.g., background meetings, messaging). "
            f"Use the Activity State Per Segment section above to identify which "
            f"content relates to this activity, and ignore unrelated content. "
            f"Your analysis should only cover what happened within this specific activity."
        )

    if not parts:
        return None

    return "\n\n".join(parts)


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
        sources: Source config dict from frontmatter load

    Returns:
        Tuple of (transcript text, source_counts dict)
    """
    # Set segment key for token usage logging
    if segment:
        os.environ["SOL_SEGMENT"] = segment
    elif span:
        os.environ["SOL_SEGMENT"] = span[0]

    # Convert sources config for clustering.
    # Frontmatter now uses ``load.talents`` but cluster still consumes the
    # normalized ``agents`` source key internally.
    cluster_sources: dict = {}
    for k, v in sources.items():
        if k == "talents":
            talent_filter = get_talent_filter(v)
            if talent_filter is None:
                cluster_sources["agents"] = source_is_enabled(v)
            elif not talent_filter:
                cluster_sources["agents"] = False
            else:
                cluster_sources["agents"] = talent_filter
        else:
            cluster_sources[k] = source_is_enabled(v)

    # Build transcript via clustering
    stream = os.environ.get("SOL_STREAM")
    if span:
        return cluster_span(day, span, sources=cluster_sources, stream=stream)
    elif segment:
        return cluster_period(day, segment, sources=cluster_sources, stream=stream)
    else:
        return cluster(day, sources=cluster_sources)


def prepare_config(request: dict) -> dict:
    """Prepare a complete talent config from a request.

    Single unified preparation path for all talent types. Takes raw request
    from cortex and returns fully prepared config ready for execution.

    Config fields produced:
    - name: Talent name
    - provider, model: Resolved from context/request
    - user_instruction: Talent instruction from .md file
    - prompt: User's runtime query/request
    - transcript: Clustered transcript (if day provided)
    - output_path: Where to write output (if output format set)
    - skip_reason: Why to skip (if applicable)

    Args:
        request: Raw request dict from cortex

    Returns:
        Fully prepared config dict
    """
    from think.models import resolve_model_for_provider, resolve_provider
    from think.talent import get_talent, key_to_context

    name = request["name"]
    facet = request.get("facet")
    day = request.get("day")
    segment = request.get("segment")
    span = request.get("span")
    activity = request.get("activity")
    output_format = request.get("output")
    output_path_override = request.get("output_path")
    user_prompt = request.get("prompt", "")

    # Load complete talent config
    config = get_talent(name, facet=facet, analysis_day=day)

    # Config now contains all frontmatter fields plus:
    # - path: Path to the .md file
    # - sources: Source config for transcript loading
    # - All frontmatter: tools, hook, disabled, thinking_budget, max_output_tokens, etc.

    # Convert path string to Path object for convenience
    talent_path = Path(config["path"]) if config.get("path") else None
    sources = config.get("sources", {})
    talent_cwd = config.get("cwd")

    # Merge request values (request overrides talent defaults)
    config.update({k: v for k, v in request.items() if v is not None})
    request_cwd = request.get("cwd")
    if request_cwd is not None and request_cwd != talent_cwd:
        raise ValueError(
            f"Request overrides 'cwd' for talent '{name}' are not allowed "
            f"({talent_cwd!r} != {request_cwd!r})"
        )

    cwd_value = config.get("cwd")
    if cwd_value == "journal":
        try:
            journal_path = Path(get_journal())
        except Exception as exc:
            raise RuntimeError(
                f"Cannot resolve cwd for talent '{name}' — journal path unavailable"
            ) from exc
        if not journal_path.exists():
            raise RuntimeError(
                f"Cannot resolve cwd for talent '{name}' — journal path unavailable"
            )
        config["cwd"] = str(journal_path)
    elif cwd_value == "repo":
        config["cwd"] = get_project_root()

    # Populate stream from env if not already in config (think passes it as
    # SOL_STREAM env var but not as a top-level request key — hooks need it)
    if "stream" not in config:
        sol_stream = os.environ.get("SOL_STREAM")
        if sol_stream:
            config["stream"] = sol_stream

    # Track additional state
    config["span_mode"] = bool(span)
    config["source_counts"] = {}

    # Resolve provider and model from context
    context = key_to_context(name)
    talent_type = config["type"]
    default_provider, default_model = resolve_provider(context, talent_type)

    provider = config.get("provider") or default_provider
    model = config.get("model")
    if not model:
        if provider != default_provider:
            model = resolve_model_for_provider(context, provider, talent_type)
        else:
            model = default_model

    config["provider"] = provider
    config["model"] = model
    config["context"] = context

    # --- Provider fallback: preflight swap if primary is unhealthy ---
    from think.models import (
        get_backup_provider,
        is_provider_healthy,
        load_health_status,
        should_recheck_health,
    )
    from think.providers import PROVIDER_METADATA

    health_data = load_health_status()
    config["health_stale"] = should_recheck_health(health_data)

    if not is_provider_healthy(provider, health_data):
        backup = get_backup_provider(talent_type)
        if backup and backup != provider:
            env_key = PROVIDER_METADATA.get(backup, {}).get("env_key")
            if not env_key or os.getenv(env_key):
                config["fallback_from"] = provider
                config["provider"] = backup
                config["model"] = resolve_model_for_provider(
                    context, backup, talent_type
                )

    # Check if disabled
    if config.get("disabled"):
        config["skip_reason"] = "disabled"
        return config

    # Day-based processing: load transcript and apply template substitution
    if day:
        # Load transcript (only when the talent has enabled sources to consume)
        if any(source_is_enabled(v) for v in sources.values()):
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

        # Reload talent instruction with template substitution for day/segment context
        if talent_path and talent_path.exists():
            from think.prompts import _resolve_facets

            prompt_context = _build_prompt_context(
                day, segment, span, activity=activity, facet=facet
            )
            prompt_context["facets"] = _resolve_facets(facet)

            if activity and span and facet:
                activity_ctx = _build_activity_context(activity, span, facet, day)
                if activity_ctx:
                    prompt_context["activity_context"] = activity_ctx

            talent_prompt_obj = load_prompt(
                talent_path.stem, base_dir=talent_path.parent, context=prompt_context
            )
            config["user_instruction"] = talent_prompt_obj.text

    # Set prompt (user's runtime query)
    # For tool talents: prompt is the user's question
    # For generators: prompt is typically empty (instruction is in user_instruction)
    config["prompt"] = user_prompt

    # Determine output path
    if output_format:
        if output_path_override:
            config["output_path"] = Path(output_path_override)
        elif day:
            stream = os.environ.get("SOL_STREAM")
            day_dir = str(day_path(day))
            config["output_path"] = get_output_path(
                day_dir,
                name,
                segment=segment,
                output_format=output_format,
                facet=facet,
                stream=stream,
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

    if is_cogitate and not (has_prompt or has_user_instruction):
        return "Cogitate talent requires non-empty 'prompt' or 'user_instruction'"

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


def _apply_template_vars(config: dict, template_vars: dict) -> None:
    """Substitute template_vars into text fields of config in-place.

    Expands each key with auto-capitalize convention (matching load_prompt):
      {"foo": "bar"} -> $foo="bar", $Foo="Bar"
    """
    expanded = {}
    for key, value in template_vars.items():
        str_value = str(value)
        expanded[key] = str_value
        expanded[key.capitalize()] = str_value.capitalize()

    for field in ("user_instruction", "transcript", "prompt"):
        text = config.get(field)
        if text:
            config[field] = Template(text).safe_substitute(expanded)


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
# Unified Talent Execution
# =============================================================================


def _write_output(output_path: Path, result: str) -> None:
    """Write result to output file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)
    LOG.info("Wrote output to %s", output_path)


def _build_dry_run_event(config: dict, before_values: dict) -> dict:
    """Build a dry-run event with all context."""
    talent_type = config["type"]

    event: dict[str, Any] = {
        "event": "dry_run",
        "ts": now_ms(),
        "type": talent_type,
        "name": config["name"],
        "provider": config.get("provider", ""),
        "model": config.get("model") or "unknown",
        "system_instruction": config.get("system_instruction", ""),
        "user_instruction": config.get("user_instruction", ""),
        "prompt": config.get("prompt", ""),
    }

    extra_context = config.get("extra_context", "")
    if extra_context:
        event["extra_context"] = extra_context

    # Day-based fields
    if config.get("day"):
        event["day"] = config["day"]
        event["segment"] = config.get("segment")
        transcript = config.get("transcript", "")
        if transcript:
            event["transcript"] = transcript
            event["transcript_chars"] = len(transcript)
            event["transcript_files"] = sum(config.get("source_counts", {}).values())
        output_path = Path(config["output_path"]) if config.get("output_path") else None
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


_NON_RETRYABLE_ERRORS = (
    ValueError,
    json.JSONDecodeError,
    KeyError,
    TypeError,
    AttributeError,
    FileNotFoundError,
    PermissionError,
    NotImplementedError,
)


def _is_retryable_error(exc: Exception) -> bool:
    """Check if an exception is likely a provider error worth retrying.

    Returns False for local/code errors (ValueError, KeyError, etc.).
    Returns True for everything else (SDK connection, timeout, server errors).
    """
    return not isinstance(exc, _NON_RETRYABLE_ERRORS)


async def _execute_with_tools(
    config: dict,
    emit_event: Callable[[dict], None],
) -> None:
    """Execute a tool-using talent via the provider's run_cogitate.

    Args:
        config: Prepared config dict
        emit_event: Event emission callback
    """
    from .providers import PROVIDER_REGISTRY, get_provider_module

    provider = config.get("provider", "google")
    output_path = Path(config["output_path"]) if config.get("output_path") else None

    if provider not in PROVIDER_REGISTRY:
        valid = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
        raise ValueError(f"Unknown provider: {provider!r}. Valid providers: {valid}")

    provider_mod = get_provider_module(provider)

    # Wrapper to intercept finish event for post-processing
    def talent_emit_event(data: Event) -> None:
        if data.get("event") == "finish":
            result = data.get("result", "")
            result = _run_post_hooks(result, config)
            if result != data.get("result", ""):
                data = {**data, "result": result}
            if output_path and result:
                _write_output(output_path, result)

        # Filter out start events from providers (we already emitted ours)
        if data.get("event") == "start":
            return

        emit_event(data)

    try:
        await provider_mod.run_cogitate(config=config, on_event=talent_emit_event)
    except Exception as exc:
        if not _is_retryable_error(exc) or config.get("fallback_from"):
            raise
        from think.models import (
            get_backup_provider,
            resolve_model_for_provider,
        )
        from think.providers import PROVIDER_METADATA

        backup = get_backup_provider("cogitate")
        if not backup or backup == provider:
            raise
        env_key = PROVIDER_METADATA.get(backup, {}).get("env_key")
        if env_key and not os.getenv(env_key):
            raise

        context = config.get("context")
        if not context:
            from think.talent import key_to_context

            context = key_to_context(config["name"])
        backup_model = resolve_model_for_provider(context, backup, "cogitate")

        emit_event(
            {
                "event": "fallback",
                "ts": now_ms(),
                "original_provider": provider,
                "backup_provider": backup,
                "reason": "on_failure",
                "error": str(exc),
            }
        )

        config["fallback_from"] = provider
        config["provider"] = backup
        config["model"] = backup_model

        backup_mod = get_provider_module(backup)

        # Suppress error events from backup provider — if backup also fails
        # we report the original error, not the backup's error.
        def backup_emit(data: Event) -> None:
            if data.get("event") == "error":
                return
            talent_emit_event(data)

        try:
            await backup_mod.run_cogitate(config=config, on_event=backup_emit)
        except Exception:
            # Ensure the original error is reported by the caller even if the
            # primary provider already emitted its own error event (_evented).
            if hasattr(exc, "_evented"):
                delattr(exc, "_evented")
            raise exc
    finally:
        if config.get("health_stale"):
            from think.models import request_health_recheck

            request_health_recheck()
            config["health_stale"] = False


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
    from think.talent import key_to_context

    name = config["name"]
    messages = config.get("messages")
    transcript = config.get("transcript", "")
    user_instruction = config.get("user_instruction", "")
    prompt = config.get("prompt", "")
    system_instruction = config.get("system_instruction") or None
    output_path = Path(config["output_path"]) if config.get("output_path") else None
    output_format = config.get("output")

    # Get generation parameters from config (set in frontmatter)
    thinking_budget = config.get("thinking_budget") or 8192 * 2
    max_output_tokens = config.get("max_output_tokens") or 8192 * 6
    is_json_output = output_format == "json"

    # Derive LLM request timeout from token budget: scale with output size,
    # floor at 120s, cap at 480s (well under cortex's 600s subprocess kill).
    timeout_s = config.get("timeout_s") or min(
        480, max(120, (max_output_tokens + thinking_budget) // 100)
    )

    if messages and isinstance(messages, list):
        contents = messages
    else:
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
    try:
        gen_result = generate_with_result(
            contents=contents,
            context=context,
            temperature=0.3,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
            system_instruction=system_instruction,
            json_output=is_json_output,
            json_schema=config.get("json_schema"),
            timeout_s=timeout_s,
        )
    except Exception as exc:
        if not _is_retryable_error(exc) or config.get("fallback_from"):
            raise
        from think.models import (
            get_backup_provider,
            resolve_model_for_provider,
        )
        from think.providers import PROVIDER_METADATA

        provider = config.get("provider", "google")
        backup = get_backup_provider("generate")
        if not backup or backup == provider:
            raise
        env_key = PROVIDER_METADATA.get(backup, {}).get("env_key")
        if env_key and not os.getenv(env_key):
            raise

        backup_model = resolve_model_for_provider(context, backup, "generate")

        emit_event(
            {
                "event": "fallback",
                "ts": now_ms(),
                "original_provider": provider,
                "backup_provider": backup,
                "reason": "on_failure",
                "error": str(exc),
            }
        )

        config["fallback_from"] = provider
        config["provider"] = backup
        config["model"] = backup_model

        try:
            gen_result = generate_with_result(
                contents=contents,
                context=context,
                temperature=0.3,
                max_output_tokens=max_output_tokens,
                thinking_budget=thinking_budget,
                system_instruction=system_instruction,
                json_output=is_json_output,
                json_schema=config.get("json_schema"),
                timeout_s=timeout_s,
                provider=backup,
                model=backup_model,
            )
        except Exception:
            raise exc
    finally:
        if config.get("health_stale"):
            from think.models import request_health_recheck

            request_health_recheck()
            config["health_stale"] = False

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
    if "schema_validation" in gen_result:
        finish_event["schema_validation"] = gen_result["schema_validation"]
    emit_event(finish_event)


async def _run_talent(
    config: dict,
    emit_event: Callable[[dict], None],
    dry_run: bool = False,
) -> None:
    """Execute a talent based on config.

    Unified execution path for all talent types. Handles:
    - Skip conditions (disabled, no input, etc.)
    - Output existence checking (skip if exists unless refresh)
    - Pre/post hooks
    - Dry-run mode
    - Routing to tool or generate execution

    Args:
        config: Fully prepared config dict
        emit_event: Callback to emit JSONL events
        dry_run: If True, emit dry_run event instead of calling LLM
    """
    name = config["name"]
    provider = config.get("provider", "google")
    model = config.get("model")
    is_cogitate = config["type"] == "cogitate"
    refresh = config.get("refresh", False)
    output_path = Path(config["output_path"]) if config.get("output_path") else None

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

    # Emit preflight fallback event if provider was swapped
    if config.get("fallback_from"):
        emit_event(
            {
                "event": "fallback",
                "ts": now_ms(),
                "original_provider": config["fallback_from"],
                "backup_provider": config["provider"],
                "reason": "preflight",
            }
        )

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
            day_log(config["day"], f"talent {name} skipped ({skip_reason})")
        return

    # Check if output already exists (applies to both tool talents and generators)
    if output_path and not refresh and not dry_run:
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
    before_values["extra_context"] = config.get("extra_context", "")

    # Run pre-hooks
    modifications = _run_pre_hooks(config)
    template_vars = modifications.pop("template_vars", None)
    for key, value in modifications.items():
        config[key] = value
    if template_vars:
        LOG.info("Pre-hook template_vars: %s", list(template_vars.keys()))
        _apply_template_vars(config, template_vars)

    # Handle skip conditions set by pre-hooks
    skip_reason = config.get("skip_reason")
    if skip_reason:
        LOG.info("Config %s skipped by pre-hook: %s", name, skip_reason)
        emit_event(
            {
                "event": "finish",
                "ts": now_ms(),
                "result": "",
                "skipped": skip_reason,
            }
        )
        if config.get("day"):
            day_log(config["day"], f"talent {name} skipped ({skip_reason})")
        return

    # Dry-run mode
    if dry_run:
        emit_event(_build_dry_run_event(config, before_values))
        return

    # Execute based on talent type
    if is_cogitate:
        await _execute_with_tools(config, emit_event)
    else:
        await _execute_generate(config, emit_event)

    # Log completion
    if config.get("day"):
        day_log(config["day"], f"talent {name} ok")


# =============================================================================
# Utility Functions
# =============================================================================


def scan_day(day: str) -> dict[str, list[str]]:
    """Return lists of processed and pending daily generator output files.

    Only scans daily generators (schedule='daily'). Segment generators are
    stored within segment directories and are not included here.
    """
    day_dir = day_path(day)
    daily_generators = get_talent_configs(
        type="generate", schedule="daily", include_disabled=True
    )
    processed: list[str] = []
    pending: list[str] = []
    for key, meta in sorted(daily_generators.items()):
        output_format = meta.get("output")
        output_file = get_output_path(day_dir, key, output_format=output_format)
        if output_file.exists():
            processed.append(os.path.join("talents", output_file.name))
        else:
            pending.append(os.path.join("talents", output_file.name))
    return {"processed": sorted(processed), "repairable": sorted(pending)}


# =============================================================================
# Main Entry Point
# =============================================================================


async def main_async() -> None:
    """NDJSON-based CLI for talents."""

    parser = argparse.ArgumentParser(
        description="solstone Talent CLI - Accepts NDJSON input via stdin"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be sent to the provider without calling the LLM",
    )
    args = setup_cli(parser)
    require_solstone()
    dry_run = args.dry_run

    app_logger = setup_logging(args.verbose)
    event_writer = JSONEventWriter(None)
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()
    registered_signals: list[signal.Signals] = []
    if main_task:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, main_task.cancel)
                registered_signals.append(sig)
            except (NotImplementedError, RuntimeError):
                LOG.debug("Signal handler registration unavailable for %s", sig)

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

                await _run_talent(config, emit_event, dry_run=dry_run)

            except json.JSONDecodeError as e:
                emit_event(
                    {
                        "event": "error",
                        "error": f"Invalid JSON: {str(e)}",
                        "ts": now_ms(),
                    }
                )
            except Exception as e:
                if getattr(e, "_evented", False):
                    continue
                from think.models import IncompleteJSONError

                event = {
                    "event": "error",
                    "error": str(e),
                    "trace": traceback.format_exc(),
                    "ts": now_ms(),
                }
                if isinstance(e, IncompleteJSONError):
                    from think._extraction_utils import log_extraction_failure

                    event["partial_text_length"] = len(e.partial_text)
                    event["partial_text_tail"] = e.partial_text[-500:]
                    name = config.get("name", "unknown") if config else "unknown"
                    log_extraction_failure(e, name)
                emit_event(event)

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
        for sig in registered_signals:
            loop.remove_signal_handler(sig)
        event_writer.close()


def main() -> None:
    """Entry point wrapper."""
    asyncio.run(main_async())


__all__ = [
    "prepare_config",
    "validate_config",
    "scan_day",
]

if __name__ == "__main__":
    main()
