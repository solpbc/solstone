#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""OpenAI provider for agents and direct LLM generation.

This module provides the OpenAI provider for the ``sol providers check`` CLI
and run_generate/run_agenerate functions returning GenerateResult.

Common Parameters
-----------------
contents : str or list
    The content to send to the model.
model : str
    Model name, optionally with a reasoning effort suffix.
    Supported suffixes: ``-none``, ``-low``, ``-medium``, ``-high``, ``-xhigh``.
    Example: ``"gpt-5.2-high"`` sends ``reasoning={"effort": "high"}`` to the API.
    Without a suffix, ``reasoning`` is omitted (OpenAI model default).
max_output_tokens : int
    Maximum tokens for the model's response output.
system_instruction : str, optional
    System instruction for the model.
json_output : bool
    Whether to request JSON response format.
timeout_s : float, optional
    Request timeout in seconds.
**kwargs
    Additional provider-specific options.

Note: GPT-5+ reasoning models don't support custom temperature (fixed at 1.0).
"""

from __future__ import annotations

import functools
import logging
import os
import re
import traceback
from pathlib import Path
from typing import Any, Callable

from think.models import GPT_5, OPENAI_EFFORT_SUFFIXES
from think.providers.cli import (
    CLIRunner,
    ThinkingAggregator,
    assemble_prompt,
    build_cogitate_env,
)
from think.utils import now_ms

from .shared import (
    GenerateResult,
    JSONEventCallback,
    safe_raw,
)

# Agent configuration is now loaded via get_talent() in cortex.py

LOG = logging.getLogger("think.providers.openai")
_SCHEMA_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _parse_model_effort(model: str) -> tuple[str, str | None]:
    """Extract reasoning effort suffix from a model name.

    Returns (api_model, effort) where api_model has the suffix stripped
    and effort is the reasoning effort value (or None if no suffix).
    """
    for suffix in OPENAI_EFFORT_SUFFIXES:
        if model.endswith(suffix):
            return model[: -len(suffix)], suffix[1:]
    return model, None


def _translate_codex(
    event: dict[str, Any],
    aggregator: ThinkingAggregator,
    callback: JSONEventCallback,
    usage_holder: list[dict[str, Any]] | None = None,
) -> str | None:
    """Translate a Codex JSONL event into our standard Event format.

    Returns the thread_id from ``thread.started`` events so CLIRunner can
    capture it as ``cli_session_id``.  All other events return ``None``.
    """
    event_type = event.get("type", "")
    item = event.get("item") or {}
    item_type = item.get("type", "")

    # -- thread.started: capture session ID --------------------------------
    if event_type == "thread.started":
        return event.get("thread_id")

    # -- turn.started: no-op -----------------------------------------------
    if event_type == "turn.started":
        return None

    # -- item.started: command_execution → tool_start ----------------------
    if event_type == "item.started" and item_type == "command_execution":
        aggregator.flush_as_thinking(raw_events=[event])
        callback.emit(
            {
                "event": "tool_start",
                "tool": "bash",
                "args": {"command": item.get("command", "")},
                "call_id": item.get("id", ""),
                "raw": safe_raw([event]),
            }
        )
        return None

    # -- item.completed ----------------------------------------------------
    if event_type == "item.completed":
        if item_type == "reasoning":
            thinking_event: dict[str, Any] = {
                "event": "thinking",
                "summary": item.get("text", ""),
                "raw": safe_raw([event]),
                "ts": now_ms(),
            }
            if aggregator._model:
                thinking_event["model"] = aggregator._model
            callback.emit(thinking_event)
            return None

        if item_type == "agent_message":
            aggregator.accumulate(item.get("text", ""))
            return None

        if item_type == "command_execution":
            callback.emit(
                {
                    "event": "tool_end",
                    "tool": "bash",
                    "args": {"command": item.get("command", "")},
                    "result": item.get("aggregated_output", ""),
                    "call_id": item.get("id", ""),
                    "raw": safe_raw([event]),
                }
            )
            return None

    # -- turn.completed: capture usage -------------------------------------
    if event_type == "turn.completed":
        if usage_holder is not None and event.get("usage"):
            usage_holder[0] = event["usage"]
        return None

    return None


async def run_cogitate(
    config: dict[str, Any],
    on_event: Callable[[dict], None] | None = None,
) -> str:
    """Run a prompt with Codex CLI subprocess tool-calling support.

    Uses streaming and emits JSON events.

    Args:
        config: Complete configuration dictionary including prompt, system_instruction,
            user_instruction, extra_context, model, etc.
        on_event: Optional event callback
    """
    raw_model = config.get("model") or GPT_5
    model, effort = _parse_model_effort(raw_model)  # strip suffix for CLI
    LOG.info("Running agent with model %s", model)
    cb = JSONEventCallback(on_event)

    # Note: Start event is emitted by agents.py (unified event ownership)

    # Assemble prompt — Codex has no --system-prompt flag, so prepend it
    prompt_body, system_instruction = assemble_prompt(config)
    if system_instruction:
        prompt_text = system_instruction + "\n\n" + prompt_body
    else:
        prompt_text = prompt_body

    # Build command — sandbox is read-only; "sol" commands bypass
    # the sandbox via exec-policy rules in .codex/rules/solstone.rules
    # Write-enabled agents get full sandbox access
    sandbox = "workspace-write" if config.get("write") else "read-only"

    session_id = config.get("session_id")
    sandbox = "workspace-write" if config.get("write") else "read-only"
    if session_id:
        cmd = [
            "codex",
            "exec",
            "resume",
            session_id,
            "--json",
            "-s",
            sandbox,
            "-m",
            model,
        ]
    else:
        cmd = ["codex", "exec", "--json", "-s", sandbox, "-m", model]

    if effort:
        cmd.extend(["-c", f'model_reasoning_effort="{effort}"'])

    cmd.append("-")  # read prompt from stdin

    # Create runner
    usage_holder: list[dict[str, Any]] = [{}]
    aggregator = ThinkingAggregator(cb, model)
    translate = functools.partial(_translate_codex, usage_holder=usage_holder)
    cwd_value = config.get("cwd")
    runner = CLIRunner(
        cmd=cmd,
        prompt_text=prompt_text,
        translate=translate,
        callback=cb,
        aggregator=aggregator,
        cwd=Path(cwd_value) if cwd_value else None,
        env=build_cogitate_env("OPENAI_API_KEY"),
    )

    try:
        result = await runner.run()
    except Exception as exc:
        if not getattr(exc, "_evented", False):
            cb.emit(
                {
                    "event": "error",
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                }
            )
            setattr(exc, "_evented", True)
        raise

    # Emit finish event
    finish_event: dict[str, Any] = {
        "event": "finish",
        "result": result,
    }
    if runner.cli_session_id:
        finish_event["cli_session_id"] = runner.cli_session_id
    if usage_holder[0]:
        finish_event["usage"] = usage_holder[0]
    else:
        LOG.warning("No usage data captured from Codex CLI for model %s", raw_model)
    cb.emit(finish_event)

    return result


# ---------------------------------------------------------------------------
# run_generate / run_agenerate functions
# ---------------------------------------------------------------------------

# Cache for OpenAI clients
_openai_client = None
_async_openai_client = None


def _get_openai_client():
    """Get or create sync OpenAI client."""
    global _openai_client
    if _openai_client is None:
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        _openai_client = openai.OpenAI(api_key=api_key)
    return _openai_client


def _get_async_openai_client():
    """Get or create async OpenAI client."""
    global _async_openai_client
    if _async_openai_client is None:
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        _async_openai_client = openai.AsyncOpenAI(api_key=api_key)
    return _async_openai_client


def _build_input(
    contents: Any,
    system_instruction: str | None = None,
) -> tuple[Any, str | None]:
    """Build OpenAI Responses input and system instructions."""
    if isinstance(contents, str):
        return contents, system_instruction
    if isinstance(contents, list):
        if contents and isinstance(contents[0], dict) and "role" in contents[0]:
            return contents, system_instruction
        return "\n".join(str(c) for c in contents), system_instruction
    return str(contents), system_instruction


def _derive_schema_name(schema: dict | None) -> str:
    """Return a valid schema name for OpenAI structured outputs."""
    if isinstance(schema, dict):
        title = schema.get("title")
        if isinstance(title, str) and title and _SCHEMA_NAME_RE.fullmatch(title):
            return title
    return "response"


def _normalize_finish_reason(response: Any) -> str | None:
    """Normalize OpenAI finish_reason to standard values.

    Returns normalized string: "stop", "max_tokens", "content_filter", or None.
    """
    if not response or not getattr(response, "status", None):
        return None

    status = response.status
    if status == "completed":
        return "stop"
    if status == "incomplete":
        incomplete_details = getattr(response, "incomplete_details", None)
        if (
            incomplete_details is not None
            and getattr(incomplete_details, "reason", None) == "content_filter"
        ):
            return "content_filter"
        return "max_tokens"
    if status == "failed":
        return "error"
    return status


def _extract_thinking(response: Any) -> list | None:
    """Extract reasoning summaries from Responses API output.

    Returns list of thinking block dicts or None if no reasoning.
    """
    if not hasattr(response, "output") or not response.output:
        return None

    thinking_blocks = []
    for item in response.output:
        if getattr(item, "type", None) != "reasoning":
            continue
        for summary in getattr(item, "summary", None) or []:
            text = getattr(summary, "text", None)
            if text:
                thinking_blocks.append({"summary": text})

    return thinking_blocks if thinking_blocks else None


def _extract_usage(response: Any) -> dict | None:
    """Extract normalized usage dict from OpenAI response."""
    if not response.usage:
        return None

    usage: dict[str, int] = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    # Extract optional detail fields
    input_details = getattr(response.usage, "input_tokens_details", None)
    if input_details:
        cached = getattr(input_details, "cached_tokens", 0)
        if cached:
            usage["cached_tokens"] = cached
    output_details = getattr(response.usage, "output_tokens_details", None)
    if output_details:
        reasoning = getattr(output_details, "reasoning_tokens", 0)
        if reasoning:
            usage["reasoning_tokens"] = reasoning
    return usage


def run_generate(
    contents: Any,
    model: str = GPT_5,
    max_output_tokens: int = 8192 * 2,
    system_instruction: str | None = None,
    json_output: bool = False,
    json_schema: dict | None = None,
    timeout_s: float | None = None,
    **kwargs: Any,
) -> GenerateResult:
    """Generate text synchronously.

    Returns GenerateResult with text, usage, finish_reason, and thinking.
    See module docstring for parameter details.
    """
    client = _get_openai_client()
    input_content, instructions = _build_input(contents, system_instruction)

    # Parse effort suffix from model name (e.g., "gpt-5.2-high" → "gpt-5.2", "high")
    api_model, effort = _parse_model_effort(model)

    # Build request kwargs
    request_kwargs: dict[str, Any] = {
        "model": api_model,
        "input": input_content,
        "max_output_tokens": max_output_tokens,
    }
    if instructions is not None:
        request_kwargs["instructions"] = instructions
    if effort is not None:
        request_kwargs["reasoning"] = {"effort": effort}

    if json_schema is not None:
        request_kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": _derive_schema_name(json_schema),
                "schema": json_schema,
                "strict": True,
            }
        }
    elif json_output:
        request_kwargs["text"] = {"format": {"type": "json_object"}}

    if timeout_s:
        request_kwargs["timeout"] = timeout_s

    response = client.responses.create(**request_kwargs)
    return GenerateResult(
        text=response.output_text or "",
        usage=_extract_usage(response),
        finish_reason=_normalize_finish_reason(response),
        thinking=_extract_thinking(response),
    )


async def run_agenerate(
    contents: Any,
    model: str = GPT_5,
    max_output_tokens: int = 8192 * 2,
    system_instruction: str | None = None,
    json_output: bool = False,
    json_schema: dict | None = None,
    timeout_s: float | None = None,
    **kwargs: Any,
) -> GenerateResult:
    """Generate text asynchronously.

    Returns GenerateResult with text, usage, finish_reason, and thinking.
    See module docstring for parameter details.
    """
    client = _get_async_openai_client()
    input_content, instructions = _build_input(contents, system_instruction)

    # Parse effort suffix from model name (e.g., "gpt-5.2-high" → "gpt-5.2", "high")
    api_model, effort = _parse_model_effort(model)

    # Build request kwargs
    request_kwargs: dict[str, Any] = {
        "model": api_model,
        "input": input_content,
        "max_output_tokens": max_output_tokens,
    }
    if instructions is not None:
        request_kwargs["instructions"] = instructions
    if effort is not None:
        request_kwargs["reasoning"] = {"effort": effort}

    if json_schema is not None:
        request_kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": _derive_schema_name(json_schema),
                "schema": json_schema,
                "strict": True,
            }
        }
    elif json_output:
        request_kwargs["text"] = {"format": {"type": "json_object"}}

    if timeout_s:
        request_kwargs["timeout"] = timeout_s

    response = await client.responses.create(**request_kwargs)
    return GenerateResult(
        text=response.output_text or "",
        usage=_extract_usage(response),
        finish_reason=_normalize_finish_reason(response),
        thinking=_extract_thinking(response),
    )


def list_models() -> list[dict]:
    """List available OpenAI models.

    Returns
    -------
    list[dict]
        List of raw model info objects from the OpenAI API.
    """
    client = _get_openai_client()
    return [m.model_dump() for m in client.models.list()]


def validate_key(api_key: str) -> dict:
    """Validate an OpenAI API key by listing models.

    Creates a temporary client with the provided key. Never uses
    the cached client or environment variables.

    Returns {"valid": True} or {"valid": False, "error": "..."}.
    """
    try:
        import openai

        client = openai.OpenAI(api_key=api_key, timeout=10)
        list(client.models.list())
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": str(e)}


__all__ = [
    "run_cogitate",
    "run_generate",
    "run_agenerate",
    "list_models",
    "validate_key",
]
