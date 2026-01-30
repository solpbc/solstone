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
    Provider-specific options (cached_content, client).
"""

from __future__ import annotations

import logging
import os
import traceback
from typing import Any, Callable

from google import genai
from google.genai import errors as google_errors
from google.genai import types

from think.models import GEMINI_FLASH
from think.utils import create_mcp_client, now_ms

from .shared import (
    GenerateResult,
    JSONEventCallback,
    ThinkingEvent,
    extract_agent_config,
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
    cached_content: str | None,
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

    if cached_content:
        config_args["cached_content"] = cached_content

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
    cached_content = kwargs.get("cached_content")
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
        cached_content=cached_content,
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
    cached_content = kwargs.get("cached_content")
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
        cached_content=cached_content,
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
    ) -> None:
        self.writer = writer
        self._counter = 0
        self.session = None
        self.agent_id = agent_id
        self.name = name

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

            # Build _meta dict for passing agent identity
            meta = {}
            if self.agent_id:
                meta["agent_id"] = self.agent_id
            if self.name:
                meta["name"] = self.name

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
    ac = extract_agent_config(config, default_max_tokens=_DEFAULT_MAX_TOKENS)
    callback = JSONEventCallback(on_event)

    try:
        # Check API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        start_event: dict = {
            "event": "start",
            "prompt": ac.prompt,
            "name": ac.name,
            "model": ac.model,
            "provider": "google",
        }
        if ac.continue_from:
            start_event["continue_from"] = ac.continue_from
        callback.emit(start_event)

        # Build history - check for continuation first
        if ac.continue_from:
            # Load previous conversation history using shared function
            from ..agents import parse_agent_events_to_turns

            turns = parse_agent_events_to_turns(ac.continue_from)
            # Convert to Google's format
            history = []
            for turn in turns:
                role = "model" if turn["role"] == "assistant" else turn["role"]
                history.append(
                    types.Content(role=role, parts=[types.Part(text=turn["content"])])
                )
        else:
            # Fresh conversation - convert generic turns to Google format
            history = []
            if ac.extra_context:
                history.append(
                    types.Content(
                        role="user", parts=[types.Part(text=ac.extra_context)]
                    )
                )
            if ac.user_instruction:
                history.append(
                    types.Content(
                        role="user", parts=[types.Part(text=ac.user_instruction)]
                    )
                )

        # Create client with retry enabled
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions()),
        )

        # Create fresh chat session
        chat = client.aio.chats.create(
            model=ac.model,
            config=types.GenerateContentConfig(
                system_instruction=ac.system_instruction
            ),
            history=history,
        )

        # Track tool usage for diagnostics
        tool_call_count = 0

        # Configure tools if MCP server URL provided
        if ac.mcp_server_url:
            # Create MCP client and attach hooks
            async with create_mcp_client(str(ac.mcp_server_url)) as mcp:
                # Attach tool logging hooks to the MCP session
                tool_hooks = ToolLoggingHooks(
                    callback, agent_id=ac.agent_id, name=ac.name
                )
                tool_hooks.attach(mcp.session)

                # Configure function calling mode based on tool filtering
                if ac.tools and isinstance(ac.tools, list):
                    logger.info(f"Filtering tools to: {ac.tools}")
                    function_calling_config = types.FunctionCallingConfig(
                        mode="ANY",  # Restrict to only allowed functions
                        allowed_function_names=ac.tools,
                    )
                else:
                    function_calling_config = types.FunctionCallingConfig(mode="AUTO")

                total_tokens, effective_thinking_budget = (
                    _compute_agent_thinking_params(
                        ac.max_output_tokens, ac.thinking_budget
                    )
                )

                cfg = types.GenerateContentConfig(
                    max_output_tokens=total_tokens,
                    tools=[mcp.session],
                    tool_config=types.ToolConfig(
                        function_calling_config=function_calling_config
                    ),
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_budget=effective_thinking_budget,
                    ),
                )

                # Send the message - SDK handles automatic function calling
                response = await chat.send_message(ac.prompt, config=cfg)
                _emit_thinking_events(response, ac.model, callback)

                # Capture tool call count from hooks
                tool_call_count = tool_hooks._counter
        else:
            # No MCP tools - just basic config
            total_tokens, effective_thinking_budget = _compute_agent_thinking_params(
                ac.max_output_tokens, ac.thinking_budget
            )

            cfg = types.GenerateContentConfig(
                max_output_tokens=total_tokens,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=effective_thinking_budget,
                ),
            )

            response = await chat.send_message(ac.prompt, config=cfg)
            _emit_thinking_events(response, ac.model, callback)

        # Extract finish reason for diagnostics and user-friendly messages
        finish_reason = _extract_finish_reason(response)
        had_tool_calls = tool_call_count > 0

        text = response.text
        tool_only = False
        if not text:
            # Log diagnostics for empty response debugging
            _log_empty_response_diagnostics(response, finish_reason, had_tool_calls)

            # Generate user-friendly completion message
            text = _format_completion_message(finish_reason, had_tool_calls)
            tool_only = True
            logger.info(
                f"Empty response.text: finish_reason={finish_reason}, "
                f"tool_calls={tool_call_count}, message={text!r}"
            )

        # Extract usage from response
        usage_dict = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            metadata = response.usage_metadata
            usage_dict = {
                "input_tokens": getattr(metadata, "prompt_token_count", 0),
                "output_tokens": getattr(metadata, "candidates_token_count", 0),
                "total_tokens": getattr(metadata, "total_token_count", 0),
            }
            # Only include optional fields if non-zero
            cached = getattr(metadata, "cached_content_token_count", 0)
            if cached:
                usage_dict["cached_tokens"] = cached
            reasoning = getattr(metadata, "thoughts_token_count", 0)
            if reasoning:
                usage_dict["reasoning_tokens"] = reasoning

        finish_event = {
            "event": "finish",
            "result": text,
            "usage": usage_dict,
            "ts": now_ms(),
        }
        if tool_only:
            finish_event["tool_only"] = True
        if finish_reason:
            finish_event["finish_reason"] = finish_reason
        callback.emit(finish_event)
        return text
    except google_errors.ServerError as exc:
        # Google API server error (5xx) - transient, may be retried
        error_msg = f"Google API server error: {exc}"
        logger.warning(error_msg)
        callback.emit(
            {
                "event": "error",
                "error": error_msg,
                "error_type": "server_error",
                "retryable": True,
                "trace": traceback.format_exc(),
            }
        )
        setattr(exc, "_evented", True)
        raise RuntimeError(error_msg) from exc
    except google_errors.ClientError as exc:
        # Google API client error (4xx) - likely a config or request issue
        error_msg = f"Google API client error: {exc}"
        logger.error(error_msg)
        callback.emit(
            {
                "event": "error",
                "error": error_msg,
                "error_type": "client_error",
                "retryable": False,
                "trace": traceback.format_exc(),
            }
        )
        setattr(exc, "_evented", True)
        raise RuntimeError(error_msg) from exc
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
