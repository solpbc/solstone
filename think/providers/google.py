#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Gemini provider for agents and direct LLM generation.

This module provides the Google Gemini provider for the ``sol agents`` CLI
and run_generate/run_agenerate functions returning GenerateResult.

Common Parameters
-----------------
contents : str or list
    The content to send to the model.
model : str
    Model name to use.
temperature : float
    Temperature for generation (default: 0.3).
max_output_tokens : int
    Maximum tokens for the model's response output.
system_instruction : str, optional
    System instruction for the model.
json_output : bool
    Whether to request JSON response format.
thinking_budget : int, optional
    Token budget for model thinking.
timeout_s : float, optional
    Request timeout in seconds.
**kwargs
    Provider-specific options (client).
"""

from __future__ import annotations

import logging
import os
import traceback
from typing import Any, Callable

from google import genai
from google.genai import types

from think.models import GEMINI_FLASH
from think.utils import now_ms

from .cli import (
    CLIRunner,
    ThinkingAggregator,
    assemble_prompt,
    lookup_cli_session_id,
)
from .shared import (
    GenerateResult,
    JSONEventCallback,
    ThinkingEvent,
    extract_tool_result,
)

_DEFAULT_MAX_TOKENS = 8192
_DEFAULT_MODEL = GEMINI_FLASH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client and helper functions for generate/agenerate
# ---------------------------------------------------------------------------


def get_or_create_client(client: genai.Client | None = None) -> genai.Client:
    """Get existing client or create new one.

    Parameters
    ----------
    client : genai.Client, optional
        Existing client to reuse. If not provided, creates a new one
        using GOOGLE_API_KEY from environment.

    Returns
    -------
    genai.Client
        The provided client or a newly created one.
    """
    if client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions()),
        )
    return client


def _compute_agent_thinking_params(
    max_output_tokens: int, thinking_budget: int | None
) -> tuple[int, int]:
    """Compute total tokens and effective thinking budget for agent run.

    Args:
        max_output_tokens: Maximum output tokens from config.
        thinking_budget: Thinking budget from config, or None for dynamic.

    Returns:
        Tuple of (total_tokens, effective_thinking_budget).
        total_tokens = max_output_tokens + (thinking_budget or 0)
        effective_thinking_budget = thinking_budget if provided, else -1 (dynamic)
    """
    total_tokens = max_output_tokens + (thinking_budget or 0)
    effective_thinking_budget = thinking_budget if thinking_budget is not None else -1
    return total_tokens, effective_thinking_budget


def _build_generate_config(
    temperature: float,
    max_output_tokens: int,
    system_instruction: str | None,
    json_output: bool,
    thinking_budget: int | None,
    timeout_s: float | None = None,
) -> types.GenerateContentConfig:
    """Build the GenerateContentConfig.

    Note: Gemini's max_output_tokens is actually the total budget (thinking + output).
    We compute this internally: total = max_output_tokens + thinking_budget.
    """
    # Compute total tokens: output + thinking budget
    total_tokens = max_output_tokens + (thinking_budget or 0)

    config_args: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": total_tokens,
    }

    if system_instruction:
        config_args["system_instruction"] = system_instruction

    if json_output:
        config_args["response_mime_type"] = "application/json"

    # Only enable thinking if explicitly requested with a positive budget
    if thinking_budget and thinking_budget > 0:
        config_args["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )

    if timeout_s:
        # Convert seconds to milliseconds for the SDK
        timeout_ms = int(timeout_s * 1000)
        config_args["http_options"] = types.HttpOptions(timeout=timeout_ms)

    return types.GenerateContentConfig(**config_args)


def _extract_response_text(response: Any) -> str:
    """Extract text from response.

    Returns response.text if available, or a friendly completion message
    if the response is empty. Raises on safety filter blocks.

    Parameters
    ----------
    response
        The response from the model.
    """
    if response is None:
        raise ValueError("No response from model")

    # Check for error conditions in candidates
    finish_reason = _extract_finish_reason(response)
    if finish_reason and "SAFETY" in finish_reason.upper():
        raise ValueError(f"Response blocked by safety filters: {finish_reason}")

    # Extract text, or generate friendly message if empty
    text = response.text if response.text else ""
    if text:
        return text

    # Empty text - generate user-friendly completion message
    return _format_completion_message(finish_reason, had_tool_calls=False)


def _normalize_finish_reason(response: Any) -> str | None:
    """Normalize finish_reason to standard values.

    Returns normalized string: "stop", "max_tokens", "safety", or None.
    """
    raw = _extract_finish_reason(response)
    if not raw:
        return None

    # Normalize (handle both enum names and string values)
    reason = raw.upper().replace("FINISHREASON.", "")

    if reason == "STOP":
        return "stop"
    elif reason == "MAX_TOKENS":
        return "max_tokens"
    elif "SAFETY" in reason:
        return "safety"
    elif reason == "RECITATION":
        return "recitation"
    else:
        return reason.lower()


def _extract_usage(response: Any) -> dict | None:
    """Extract normalized usage dict from response."""
    if not hasattr(response, "usage_metadata") or not response.usage_metadata:
        return None

    metadata = response.usage_metadata
    usage: dict[str, int] = {
        "input_tokens": getattr(metadata, "prompt_token_count", 0),
        "output_tokens": getattr(metadata, "candidates_token_count", 0),
        "total_tokens": getattr(metadata, "total_token_count", 0),
    }
    # Only include optional fields if non-zero
    cached = getattr(metadata, "cached_content_token_count", 0)
    if cached:
        usage["cached_tokens"] = cached
    reasoning = getattr(metadata, "thoughts_token_count", 0)
    if reasoning:
        usage["reasoning_tokens"] = reasoning
    return usage


def _extract_thinking(response: Any) -> list | None:
    """Extract thinking blocks from response.

    Returns list of ThinkingBlock dicts or None if no thinking.
    """
    if not hasattr(response, "candidates") or not response.candidates:
        return None

    thinking_blocks = []
    for candidate in response.candidates:
        if not candidate.content or not candidate.content.parts:
            continue
        for part in candidate.content.parts:
            if getattr(part, "thought", False) and getattr(part, "text", None):
                thinking_blocks.append({"summary": part.text})

    return thinking_blocks if thinking_blocks else None


def _extract_finish_reason(response: Any) -> str | None:
    """Extract finish_reason from response candidates.

    Returns the finish_reason string (e.g., "STOP", "MAX_TOKENS") or None
    if not available.
    """
    if not hasattr(response, "candidates") or not response.candidates:
        return None

    candidate = response.candidates[0]
    if hasattr(candidate, "finish_reason") and candidate.finish_reason:
        # Convert enum to string if needed
        reason = candidate.finish_reason
        if hasattr(reason, "name"):
            return reason.name
        return str(reason)
    return None


def _format_completion_message(finish_reason: str | None, had_tool_calls: bool) -> str:
    """Create a user-friendly completion message based on finish reason.

    Parameters
    ----------
    finish_reason
        The finish_reason from the response (e.g., "STOP", "MAX_TOKENS").
    had_tool_calls
        Whether tool calls were executed during this run.

    Returns
    -------
    str
        A concise, user-friendly completion message.
    """
    if not finish_reason:
        finish_reason = "UNKNOWN"

    # Normalize finish reason (handle both enum names and string values)
    reason = finish_reason.upper().replace("FINISHREASON.", "")

    if reason == "STOP":
        if had_tool_calls:
            return "Completed via tools."
        return "Completed."
    elif reason == "MAX_TOKENS":
        return "Reached token limit."
    elif "SAFETY" in reason:
        return "Blocked by safety filters."
    elif reason == "RECITATION":
        return "Stopped due to recitation."
    elif "TOOL" in reason or "FUNCTION" in reason:
        # UNEXPECTED_TOOL_CALL, MALFORMED_FUNCTION_CALL, etc.
        return "Tool execution incomplete."
    else:
        # Unknown reason - include it for debugging
        return f"Completed ({reason.lower()})."


def _log_empty_response_diagnostics(
    response: Any, finish_reason: str | None, had_tool_calls: bool
) -> None:
    """Log diagnostic information when response.text is empty.

    Helps debug intermittent empty response issues with Gemini models.
    """
    # Build diagnostic info
    diag = {
        "finish_reason": finish_reason,
        "had_tool_calls": had_tool_calls,
        "has_candidates": hasattr(response, "candidates") and bool(response.candidates),
    }

    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        diag["has_content"] = candidate.content is not None
        if candidate.content:
            diag["has_parts"] = bool(getattr(candidate.content, "parts", None))
            if hasattr(candidate.content, "parts") and candidate.content.parts:
                diag["num_parts"] = len(candidate.content.parts)
                # Check what types of parts we have
                part_types = []
                for part in candidate.content.parts:
                    if getattr(part, "thought", False):
                        part_types.append("thinking")
                    elif getattr(part, "text", None):
                        part_types.append("text")
                    elif hasattr(part, "function_call"):
                        part_types.append("function_call")
                    elif hasattr(part, "function_response"):
                        part_types.append("function_response")
                    else:
                        part_types.append("other")
                diag["part_types"] = part_types

    # Check for AFC history (indicates tools were auto-called)
    if hasattr(response, "automatic_function_calling_history"):
        afc_history = response.automatic_function_calling_history
        diag["afc_history_length"] = len(afc_history) if afc_history else 0

    logger.info(f"Empty response.text diagnostics: {diag}")


# ---------------------------------------------------------------------------
# run_generate / run_agenerate functions
# ---------------------------------------------------------------------------


def run_generate(
    contents: str | list[Any],
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: str | None = None,
    json_output: bool = False,
    thinking_budget: int | None = None,
    timeout_s: float | None = None,
    **kwargs: Any,
) -> GenerateResult:
    """Generate text synchronously.

    Returns GenerateResult with text, usage, finish_reason, and thinking.
    See module docstring for parameter details.
    """
    client = kwargs.get("client")

    client = get_or_create_client(client)
    if isinstance(contents, str):
        contents = [contents]
    config = _build_generate_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        timeout_s=timeout_s,
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    return GenerateResult(
        text=_extract_response_text(response),
        usage=_extract_usage(response),
        finish_reason=_normalize_finish_reason(response),
        thinking=_extract_thinking(response),
    )


async def run_agenerate(
    contents: str | list[Any],
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: str | None = None,
    json_output: bool = False,
    thinking_budget: int | None = None,
    timeout_s: float | None = None,
    **kwargs: Any,
) -> GenerateResult:
    """Generate text asynchronously.

    Returns GenerateResult with text, usage, finish_reason, and thinking.
    See module docstring for parameter details.
    """
    client = kwargs.get("client")

    client = get_or_create_client(client)
    if isinstance(contents, str):
        contents = [contents]
    config = _build_generate_config(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        json_output=json_output,
        thinking_budget=thinking_budget,
        timeout_s=timeout_s,
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    return GenerateResult(
        text=_extract_response_text(response),
        usage=_extract_usage(response),
        finish_reason=_normalize_finish_reason(response),
        thinking=_extract_thinking(response),
    )


# ---------------------------------------------------------------------------
# Agent functions
# ---------------------------------------------------------------------------


def _emit_thinking_events(
    response: Any, model: str, callback: JSONEventCallback
) -> None:
    """Extract and emit thinking events from a response.

    In the Google GenAI SDK, thinking content appears in response.candidates[].content.parts[]
    where each Part has a `thought` boolean indicating if it's thinking content, and the
    actual thinking text is in `part.text`.
    """
    if not hasattr(response, "candidates") or not response.candidates:
        return

    for candidate in response.candidates:
        if not candidate.content or not candidate.content.parts:
            continue

        for part in candidate.content.parts:
            # part.thought is a boolean indicating this is thinking content
            # part.text contains the actual thinking summary
            if getattr(part, "thought", False) and getattr(part, "text", None):
                thinking_event: ThinkingEvent = {
                    "event": "thinking",
                    "ts": now_ms(),
                    "summary": part.text,
                    "model": model,
                }
                callback.emit(thinking_event)


class ToolLoggingHooks:
    """Wrap ``session.call_tool`` to emit events."""

    def __init__(
        self,
        writer: JSONEventCallback,
        agent_id: str | None = None,
        name: str | None = None,
        day: str | None = None,
    ) -> None:
        self.writer = writer
        self._counter = 0
        self.session = None
        self.agent_id = agent_id
        self.name = name
        self.day = day

    def attach(self, session: Any) -> None:
        self.session = session
        original = session.call_tool

        async def wrapped(name: str, arguments: dict | None = None, **kwargs) -> Any:
            self._counter += 1
            call_id = f"{name}-{self._counter}"
            self.writer.emit(
                {
                    "event": "tool_start",
                    "tool": name,
                    "args": arguments,
                    "call_id": call_id,
                }
            )

            # Build _meta dict for passing agent identity and context
            meta = {}
            if self.agent_id:
                meta["agent_id"] = self.agent_id
            if self.name:
                meta["name"] = self.name
            if self.day:
                meta["day"] = self.day

            result = await original(
                name=name,
                arguments=arguments,
                meta=meta,
                **kwargs,
            )

            result_data = extract_tool_result(result)

            self.writer.emit(
                {
                    "event": "tool_end",
                    "tool": name,
                    "args": arguments,
                    "result": result_data,
                    "call_id": call_id,
                }
            )
            return result

        session.call_tool = wrapped  # type: ignore[assignment]


def _translate_gemini(
    event: dict[str, Any],
    aggregator: ThinkingAggregator,
    callback: JSONEventCallback,
    usage_out: dict[str, Any] | None = None,
) -> str | None:
    """Translate a Gemini CLI JSONL event into our standard Event types.

    Args:
        event: Raw JSONL event dict from the Gemini CLI.
        aggregator: ThinkingAggregator for buffering text.
        callback: JSONEventCallback for emitting events.
        usage_out: Optional mutable dict to receive usage stats from result events.

    Returns:
        The CLI session ID from init events, or None.
    """
    event_type = event.get("type")

    if event_type == "init":
        return event.get("session_id")

    if event_type == "message":
        role = event.get("role")
        if role == "user":
            return None
        if role == "assistant" and event.get("delta"):
            content = event.get("content", "")
            if content:
                aggregator.accumulate(content)
            return None
        return None

    if event_type == "tool_use":
        aggregator.flush_as_thinking(raw_events=[event])
        callback.emit(
            {
                "event": "tool_start",
                "tool": event.get("tool_name", ""),
                "args": event.get("parameters"),
                "call_id": event.get("tool_id"),
                "raw": [event],
                "ts": now_ms(),
            }
        )
        return None

    if event_type == "tool_result":
        callback.emit(
            {
                "event": "tool_end",
                "call_id": event.get("tool_id"),
                "result": event.get("output"),
                "raw": [event],
                "ts": now_ms(),
            }
        )
        return None

    if event_type == "result":
        stats = event.get("stats") or {}
        if usage_out is not None and stats:
            usage_out.update(
                {
                    "input_tokens": stats.get("input_tokens", 0),
                    "output_tokens": stats.get("output_tokens", 0),
                    "total_tokens": stats.get("total_tokens", 0),
                }
            )
            if stats.get("cached"):
                usage_out["cached_tokens"] = stats["cached"]
            if stats.get("duration_ms"):
                usage_out["duration_ms"] = stats["duration_ms"]
        return None

    # Unknown event type — log and skip
    logger.debug("Unknown Gemini CLI event type: %s", event_type)
    return None


async def run_tools(
    config: dict[str, Any],
    on_event: Callable[[dict], None] | None = None,
) -> str:
    """Run a prompt with MCP tool-calling support via Google Gemini.

    Args:
        config: Complete configuration dictionary including prompt, system_instruction,
            user_instruction, extra_context, model, etc.
        on_event: Optional event callback
    """
    model = config.get("model", _DEFAULT_MODEL)
    continue_from = config.get("continue_from")
    callback = JSONEventCallback(on_event)

    try:
        # Assemble prompt from config fields
        prompt_body, system_instruction = assemble_prompt(config)

        # Gemini CLI has no --system-prompt flag; prepend to prompt body
        if system_instruction:
            prompt_body = system_instruction + "\n\n" + prompt_body

        # Build CLI command
        cmd = [
            "gemini",
            "-p",
            "-",
            "-o",
            "stream-json",
            "--approval-mode",
            "plan",
            "-m",
            model,
        ]

        # Resume from previous session if continuing
        if continue_from:
            session_id = lookup_cli_session_id(continue_from)
            if session_id:
                cmd.extend(["--resume", session_id])

        # Mutable container for usage stats from result event
        usage: dict[str, Any] = {}

        def translate(
            event: dict[str, Any], agg: ThinkingAggregator, cb: JSONEventCallback
        ) -> str | None:
            return _translate_gemini(event, agg, cb, usage)

        aggregator = ThinkingAggregator(callback, model=model)
        runner = CLIRunner(
            cmd=cmd,
            prompt_text=prompt_body,
            translate=translate,
            callback=callback,
            aggregator=aggregator,
        )

        result = await runner.run()

        # Emit finish event (CLIRunner does not emit one)
        finish_event: dict[str, Any] = {
            "event": "finish",
            "result": result,
            "ts": now_ms(),
        }
        if usage:
            finish_event["usage"] = usage
        if runner.cli_session_id:
            finish_event["cli_session_id"] = runner.cli_session_id
        callback.emit(finish_event)
        return result
    except Exception as exc:
        callback.emit(
            {
                "event": "error",
                "error": str(exc),
                "trace": traceback.format_exc(),
            }
        )
        setattr(exc, "_evented", True)
        raise


__all__ = [
    "run_tools",
    "run_generate",
    "run_agenerate",
    "get_or_create_client",
]
