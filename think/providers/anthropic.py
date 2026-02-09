#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Anthropic Claude provider for agents and direct LLM generation.

This module provides the Anthropic Claude provider for the ``sol agents`` CLI
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
    Additional provider-specific options.
"""

from __future__ import annotations

import logging
import os
import traceback
from typing import Any, Callable

from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    RedactedThinkingBlock,
    ThinkingBlock,
)

from think.models import CLAUDE_SONNET_4
from think.utils import now_ms

from .cli import (
    CLIRunner,
    ThinkingAggregator,
    assemble_prompt,
    check_cli_binary,
)
from .shared import (
    GenerateResult,
    JSONEventCallback,
)

# Default values are now handled internally
_DEFAULT_MODEL = CLAUDE_SONNET_4

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 8096 * 2
_MIN_THINKING_BUDGET = 1024  # Anthropic minimum
_DEFAULT_THINKING_BUDGET = 10000


def _compute_thinking_params(max_tokens: int) -> tuple[int, int]:
    """Compute thinking budget and adjusted max_tokens.

    Returns (thinking_budget, adjusted_max_tokens) ensuring:
    - thinking_budget >= _MIN_THINKING_BUDGET
    - thinking_budget < adjusted_max_tokens
    """
    # Budget is the lesser of default or what fits in max_tokens
    thinking_budget = min(_DEFAULT_THINKING_BUDGET, max(max_tokens - 1000, 0))

    # Ensure minimum thinking budget
    if thinking_budget < _MIN_THINKING_BUDGET:
        thinking_budget = _MIN_THINKING_BUDGET
        # Increase max_tokens to accommodate thinking + output
        max_tokens = max(max_tokens, thinking_budget + 1000)

    return thinking_budget, max_tokens


def _resolve_agent_thinking_params(
    max_output_tokens: int, thinking_budget_config: int | None
) -> tuple[int, int]:
    """Resolve thinking budget and max tokens for agent run.

    Args:
        max_output_tokens: Maximum output tokens from config.
        thinking_budget_config: Explicit thinking budget from config, or None.

    Returns:
        Tuple of (thinking_budget, effective_max_tokens).
        If thinking_budget_config is provided and > 0, uses it directly.
        Otherwise computes defaults via _compute_thinking_params.
    """
    if thinking_budget_config is not None and thinking_budget_config > 0:
        return thinking_budget_config, max_output_tokens
    return _compute_thinking_params(max_output_tokens)


def _translate_claude(
    event: dict[str, Any],
    aggregator: ThinkingAggregator,
    callback: JSONEventCallback,
    pending_tools: dict[str, dict[str, Any]],
    result_meta: dict[str, Any],
) -> str | None:
    """Translate a Claude CLI JSONL event into our Event format.

    Args:
        event: Raw parsed JSON event from Claude CLI stdout.
        aggregator: ThinkingAggregator for text buffering.
        callback: JSONEventCallback for emitting events.
        pending_tools: Mutable dict tracking active tool calls (tool_use_id -> {tool, args}).
        result_meta: Mutable dict for storing cost/usage from result event.

    Returns:
        Session ID string from init events, None otherwise.
    """
    event_type = event.get("type")

    if event_type == "system":
        if event.get("subtype") == "init":
            return event.get("session_id")

    elif event_type == "assistant":
        message = event.get("message", {})
        content_blocks = message.get("content", [])

        # Two-pass: text/thinking first, then tool_use
        tool_use_blocks = []
        for block in content_blocks:
            block_type = block.get("type")
            if block_type == "text":
                aggregator.accumulate(block.get("text", ""))
            elif block_type == "thinking":
                thinking_event: dict[str, Any] = {
                    "event": "thinking",
                    "summary": block.get("thinking", ""),
                    "raw": [event],
                }
                if aggregator._model:
                    thinking_event["model"] = aggregator._model
                callback.emit(thinking_event)
            elif block_type == "tool_use":
                tool_use_blocks.append(block)

        for block in tool_use_blocks:
            aggregator.flush_as_thinking(raw_events=[event])

            tool_id = block.get("id", "")
            tool_name = block.get("name", "")
            tool_args = block.get("input", {})

            pending_tools[tool_id] = {"tool": tool_name, "args": tool_args}

            callback.emit(
                {
                    "event": "tool_start",
                    "tool": tool_name,
                    "args": tool_args,
                    "call_id": tool_id,
                    "raw": [event],
                }
            )

    elif event_type == "user":
        message = event.get("message", {})
        content_blocks = message.get("content", [])

        for block in content_blocks:
            if block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                tool_info = pending_tools.pop(tool_use_id, {})

                callback.emit(
                    {
                        "event": "tool_end",
                        "tool": tool_info.get("tool", ""),
                        "args": tool_info.get("args"),
                        "result": block.get("content", ""),
                        "call_id": tool_use_id,
                        "raw": [event],
                    }
                )

    elif event_type == "result":
        result_meta["cost_usd"] = event.get("total_cost_usd")
        usage = event.get("usage")
        if usage:
            result_meta["usage"] = {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": (
                    (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
                ),
            }

    return None


async def run_cogitate(
    config: dict[str, Any],
    on_event: Callable[[dict], None] | None = None,
) -> str:
    """Run a prompt with tool-calling support via Claude CLI subprocess.

    Spawns the Claude CLI in streaming JSON mode and translates its
    JSONL output into our standard Event format.

    Args:
        config: Complete configuration dictionary including prompt, system_instruction,
            user_instruction, extra_context, model, etc.
        on_event: Optional event callback
    """
    model = config.get("model", _DEFAULT_MODEL)
    session_id = config.get("session_id")

    callback = JSONEventCallback(on_event)

    try:
        check_cli_binary("claude")

        prompt_body, system_instruction = assemble_prompt(config)

        cmd = [
            "claude",
            "-p",
            "-",
            "--output-format",
            "stream-json",
            "--permission-mode",
            "plan",
            "--allowedTools",
            "Bash(sol call *)",
            "--model",
            model,
        ]

        if system_instruction:
            cmd.extend(["--system-prompt", system_instruction])

        if session_id:
            cmd.extend(["--resume", session_id])

        aggregator = ThinkingAggregator(callback, model=model)
        pending_tools: dict[str, dict[str, Any]] = {}
        result_meta: dict[str, Any] = {}

        def translate(
            event: dict[str, Any],
            agg: ThinkingAggregator,
            cb: JSONEventCallback,
        ) -> str | None:
            return _translate_claude(event, agg, cb, pending_tools, result_meta)

        runner = CLIRunner(
            cmd=cmd,
            prompt_text=prompt_body,
            translate=translate,
            callback=callback,
            aggregator=aggregator,
        )

        result = await runner.run()

        # Build finish event with usage from result meta
        usage_dict = result_meta.get("usage")
        cost_usd = result_meta.get("cost_usd")
        if usage_dict and cost_usd is not None:
            usage_dict["cost_usd"] = cost_usd

        callback.emit(
            {
                "event": "finish",
                "result": result,
                "cli_session_id": runner.cli_session_id,
                "usage": usage_dict,
                "ts": now_ms(),
            }
        )

        return result
    except Exception as exc:
        callback.emit(
            {
                "event": "error",
                "error": str(exc),
                "trace": traceback.format_exc(),
                "ts": now_ms(),
            }
        )
        setattr(exc, "_evented", True)
        raise


# ---------------------------------------------------------------------------
# run_generate / run_agenerate functions
# ---------------------------------------------------------------------------


def _extract_usage_dict(response: Any) -> dict[str, Any] | None:
    """Extract usage dict from Anthropic response.

    Returns normalized usage dict or None if usage unavailable.
    """
    if not hasattr(response, "usage") or not response.usage:
        return None

    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    usage_dict: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    # Add cache tokens if available and non-zero
    cache_creation = getattr(usage, "cache_creation_input_tokens", None)
    if cache_creation:
        usage_dict["cache_creation_tokens"] = cache_creation
    cache_read = getattr(usage, "cache_read_input_tokens", None)
    if cache_read:
        usage_dict["cached_tokens"] = cache_read
    return usage_dict


def _normalize_finish_reason(stop_reason: str | None) -> str | None:
    """Normalize Anthropic stop_reason to standard values.

    Returns normalized string: "stop", "max_tokens", or None.
    """
    if not stop_reason:
        return None

    reason = stop_reason.lower()
    if reason == "end_turn":
        return "stop"
    elif reason == "max_tokens":
        return "max_tokens"
    elif reason == "stop_sequence":
        return "stop"
    else:
        return reason


def _extract_text_and_thinking(response: Any) -> tuple[str, list | None]:
    """Extract text and thinking blocks from Anthropic response.

    Returns tuple of (text, thinking_blocks).
    """
    text = ""
    thinking_blocks = []

    for block in response.content:
        if getattr(block, "type", None) == "text":
            text += block.text
        elif isinstance(block, ThinkingBlock):
            thinking_blocks.append(
                {
                    "summary": block.thinking,
                    "signature": block.signature,
                }
            )
        elif isinstance(block, RedactedThinkingBlock):
            thinking_blocks.append(
                {
                    "summary": "[redacted]",
                    "redacted_data": block.data,
                }
            )

    return text, thinking_blocks if thinking_blocks else None


# Cache for Anthropic clients
_anthropic_client = None
_async_anthropic_client = None


def _get_anthropic_client():
    """Get or create sync Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


def _get_async_anthropic_client():
    """Get or create async Anthropic client."""
    global _async_anthropic_client
    if _async_anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        _async_anthropic_client = AsyncAnthropic(api_key=api_key)
    return _async_anthropic_client


def _convert_contents_to_messages(contents: Any) -> list[MessageParam]:
    """Convert contents to Anthropic messages format."""
    # Handle different content formats
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}]
    elif isinstance(contents, list):
        # Check if it's already in messages format
        if contents and isinstance(contents[0], dict) and "role" in contents[0]:
            return contents
        else:
            # List of content parts - combine into single user message
            combined = "\n".join(str(c) for c in contents)
            return [{"role": "user", "content": combined}]
    else:
        return [{"role": "user", "content": str(contents)}]


def run_generate(
    contents: Any,
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
    client = _get_anthropic_client()
    messages = _convert_contents_to_messages(contents)

    # Handle JSON output by adding to system instruction
    system = system_instruction or ""
    if json_output:
        json_instruction = "Respond with valid JSON only. No explanation or markdown."
        system = f"{system}\n\n{json_instruction}" if system else json_instruction

    # Build request kwargs
    request_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_output_tokens,
        "messages": messages,
    }

    if system:
        request_kwargs["system"] = system

    # Note: Anthropic doesn't support temperature with thinking enabled
    # Configure thinking if budget is provided
    if thinking_budget and thinking_budget > 0:
        request_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
    else:
        request_kwargs["temperature"] = temperature

    if timeout_s:
        request_kwargs["timeout"] = timeout_s

    response = client.messages.create(**request_kwargs)

    text, thinking = _extract_text_and_thinking(response)
    return GenerateResult(
        text=text,
        usage=_extract_usage_dict(response),
        finish_reason=_normalize_finish_reason(response.stop_reason),
        thinking=thinking,
    )


async def run_agenerate(
    contents: Any,
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
    client = _get_async_anthropic_client()
    messages = _convert_contents_to_messages(contents)

    # Handle JSON output by adding to system instruction
    system = system_instruction or ""
    if json_output:
        json_instruction = "Respond with valid JSON only. No explanation or markdown."
        system = f"{system}\n\n{json_instruction}" if system else json_instruction

    # Build request kwargs
    request_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_output_tokens,
        "messages": messages,
    }

    if system:
        request_kwargs["system"] = system

    # Note: Anthropic doesn't support temperature with thinking enabled
    # Configure thinking if budget is provided
    if thinking_budget and thinking_budget > 0:
        request_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
    else:
        request_kwargs["temperature"] = temperature

    if timeout_s:
        request_kwargs["timeout"] = timeout_s

    response = await client.messages.create(**request_kwargs)

    text, thinking = _extract_text_and_thinking(response)
    return GenerateResult(
        text=text,
        usage=_extract_usage_dict(response),
        finish_reason=_normalize_finish_reason(response.stop_reason),
        thinking=thinking,
    )


__all__ = [
    "run_cogitate",
    "run_generate",
    "run_agenerate",
]
