#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""OpenAI provider for agents and direct LLM generation.

This module provides the OpenAI provider for the ``sol agents`` CLI
and run_generate/run_agenerate functions returning GenerateResult.

Common Parameters
-----------------
contents : str or list
    The content to send to the model.
model : str
    Model name, optionally with a reasoning effort suffix.
    Supported suffixes: ``-none``, ``-low``, ``-medium``, ``-high``, ``-xhigh``.
    Example: ``"gpt-5.2-high"`` sends ``reasoning_effort="high"`` to the API.
    Without a suffix, ``reasoning_effort`` is omitted (OpenAI model default).
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
import traceback
from typing import Any, Callable

from think.models import GPT_5, OPENAI_EFFORT_SUFFIXES
from think.providers.cli import (
    CLIRunner,
    ThinkingAggregator,
    assemble_prompt,
)
from think.utils import now_ms

from .shared import (
    GenerateResult,
    JSONEventCallback,
    safe_raw,
)

# Agent configuration is now loaded via get_agent() in cortex.py

LOG = logging.getLogger("think.providers.openai")


def _parse_model_effort(model: str) -> tuple[str, str | None]:
    """Extract reasoning effort suffix from a model name.

    Returns (api_model, effort) where api_model has the suffix stripped
    and effort is the reasoning_effort value (or None if no suffix).
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
    model, _effort = _parse_model_effort(raw_model)  # strip suffix for CLI
    LOG.info("Running agent with model %s", model)
    cb = JSONEventCallback(on_event)

    # Note: Start event is emitted by agents.py (unified event ownership)

    # Assemble prompt — Codex has no --system-prompt flag, so prepend it
    prompt_body, system_instruction = assemble_prompt(config)
    if system_instruction:
        prompt_text = system_instruction + "\n\n" + prompt_body
    else:
        prompt_text = prompt_body

    # Build command — sandbox is read-only; "sol call" commands bypass
    # the sandbox via exec-policy rules in .codex/rules/solstone.rules
    session_id = config.get("session_id")
    if session_id:
        cmd = [
            "codex",
            "exec",
            "resume",
            session_id,
            "--json",
            "-s",
            "read-only",
            "-m",
            model,
        ]
    else:
        cmd = ["codex", "exec", "--json", "-s", "read-only", "-m", model]

    cmd.append("-")  # read prompt from stdin

    # Create runner
    usage_holder: list[dict[str, Any]] = [{}]
    aggregator = ThinkingAggregator(cb, model)
    translate = functools.partial(_translate_codex, usage_holder=usage_holder)
    runner = CLIRunner(
        cmd=cmd,
        prompt_text=prompt_text,
        translate=translate,
        callback=cb,
        aggregator=aggregator,
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


def _convert_contents_to_messages(
    contents: Any,
    system_instruction: str | None = None,
) -> list[dict]:
    """Convert contents to OpenAI messages format."""
    messages = []

    # Add system message if provided
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    # Handle different content formats
    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
    elif isinstance(contents, list):
        # Check if it's a list of messages or a list of content parts
        if contents and isinstance(contents[0], dict) and "role" in contents[0]:
            # Already in messages format
            messages.extend(contents)
        else:
            # List of content parts - combine into single user message
            combined = "\n".join(str(c) for c in contents)
            messages.append({"role": "user", "content": combined})
    else:
        messages.append({"role": "user", "content": str(contents)})

    return messages


def _normalize_finish_reason(finish_reason: str | None) -> str | None:
    """Normalize OpenAI finish_reason to standard values.

    Returns normalized string: "stop", "max_tokens", "content_filter", or None.
    """
    if not finish_reason:
        return None

    reason = finish_reason.lower()
    if reason == "stop":
        return "stop"
    elif reason == "length":
        return "max_tokens"
    elif reason == "content_filter":
        return "content_filter"
    else:
        return reason


def _extract_usage(response: Any) -> dict | None:
    """Extract normalized usage dict from OpenAI response."""
    if not response.usage:
        return None

    return {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }


def run_generate(
    contents: Any,
    model: str = GPT_5,
    max_output_tokens: int = 8192 * 2,
    system_instruction: str | None = None,
    json_output: bool = False,
    timeout_s: float | None = None,
    **kwargs: Any,
) -> GenerateResult:
    """Generate text synchronously.

    Returns GenerateResult with text, usage, finish_reason, and thinking.
    See module docstring for parameter details.
    """
    client = _get_openai_client()
    messages = _convert_contents_to_messages(contents, system_instruction)

    # Parse effort suffix from model name (e.g., "gpt-5.2-high" → "gpt-5.2", "high")
    api_model, effort = _parse_model_effort(model)

    # Build request kwargs
    # Note: GPT-5+ models require max_completion_tokens instead of max_tokens
    # Note: Reasoning models don't support custom temperature (only 1.0)
    request_kwargs: dict[str, Any] = {
        "model": api_model,
        "messages": messages,
        "max_completion_tokens": max_output_tokens,
    }
    if effort is not None:
        request_kwargs["reasoning_effort"] = effort

    if json_output:
        request_kwargs["response_format"] = {"type": "json_object"}

    if timeout_s:
        request_kwargs["timeout"] = timeout_s

    response = client.chat.completions.create(**request_kwargs)

    choice = response.choices[0]
    return GenerateResult(
        text=choice.message.content or "",
        usage=_extract_usage(response),
        finish_reason=_normalize_finish_reason(choice.finish_reason),
        thinking=None,  # OpenAI reasoning not exposed in generate response
    )


async def run_agenerate(
    contents: Any,
    model: str = GPT_5,
    max_output_tokens: int = 8192 * 2,
    system_instruction: str | None = None,
    json_output: bool = False,
    timeout_s: float | None = None,
    **kwargs: Any,
) -> GenerateResult:
    """Generate text asynchronously.

    Returns GenerateResult with text, usage, finish_reason, and thinking.
    See module docstring for parameter details.
    """
    client = _get_async_openai_client()
    messages = _convert_contents_to_messages(contents, system_instruction)

    # Parse effort suffix from model name (e.g., "gpt-5.2-high" → "gpt-5.2", "high")
    api_model, effort = _parse_model_effort(model)

    # Build request kwargs
    # Note: GPT-5+ models require max_completion_tokens instead of max_tokens
    # Note: Reasoning models don't support custom temperature (only 1.0)
    request_kwargs: dict[str, Any] = {
        "model": api_model,
        "messages": messages,
        "max_completion_tokens": max_output_tokens,
    }
    if effort is not None:
        request_kwargs["reasoning_effort"] = effort

    if json_output:
        request_kwargs["response_format"] = {"type": "json_object"}

    if timeout_s:
        request_kwargs["timeout"] = timeout_s

    response = await client.chat.completions.create(**request_kwargs)

    choice = response.choices[0]
    return GenerateResult(
        text=choice.message.content or "",
        usage=_extract_usage(response),
        finish_reason=_normalize_finish_reason(choice.finish_reason),
        thinking=None,  # OpenAI reasoning not exposed in generate response
    )


__all__ = [
    "run_cogitate",
    "run_generate",
    "run_agenerate",
]
