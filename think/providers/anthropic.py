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

import json
import logging
import os
import traceback
from typing import Any, Callable

from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    RedactedThinkingBlock,
    ThinkingBlock,
    ToolParam,
    ToolUseBlock,
)

from think.models import CLAUDE_SONNET_4
from think.utils import create_mcp_client, now_ms

from .shared import (
    GenerateResult,
    JSONEventCallback,
    ThinkingEvent,
    extract_tool_result,
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


def _emit_thinking_event(
    block: ThinkingBlock | RedactedThinkingBlock,
    model: str,
    callback: JSONEventCallback,
) -> None:
    """Emit a thinking event for a ThinkingBlock or RedactedThinkingBlock."""
    if isinstance(block, ThinkingBlock):
        thinking_event: ThinkingEvent = {
            "event": "thinking",
            "ts": now_ms(),
            "summary": block.thinking,
            "model": model,
            "signature": block.signature,
        }
        callback.emit(thinking_event)
    elif isinstance(block, RedactedThinkingBlock):
        redacted_event: ThinkingEvent = {
            "event": "thinking",
            "ts": now_ms(),
            "summary": "[redacted]",
            "model": model,
            "redacted_data": block.data,
        }
        callback.emit(redacted_event)


_MAX_TOOL_ITERATIONS = 25  # Safety limit for agentic loop iterations


class ToolExecutor:
    """Handle MCP tool execution and result formatting for Anthropic."""

    def __init__(
        self,
        mcp_client: Any,
        callback: JSONEventCallback,
        agent_id: str | None = None,
        name: str | None = None,
    ) -> None:
        self.mcp = mcp_client
        self.callback = callback
        self.agent_id = agent_id
        self.name = name

    async def execute_tool(self, tool_use: ToolUseBlock) -> dict:
        """Execute ``tool_use`` and return a Claude ``tool_result`` block."""
        call_id = tool_use.id  # Use Claude's tool_use_id as call_id
        self.callback.emit(
            {
                "event": "tool_start",
                "tool": tool_use.name,
                "args": tool_use.input,
                "call_id": call_id,
            }
        )

        # Build _meta dict for passing agent identity
        meta = {}
        if self.agent_id:
            meta["agent_id"] = self.agent_id
        if self.name:
            meta["name"] = self.name

        try:
            try:
                result = await self.mcp.session.call_tool(
                    name=tool_use.name,
                    arguments=tool_use.input,
                    meta=meta,
                )
            except RuntimeError:
                await self.mcp.__aenter__()
                result = await self.mcp.session.call_tool(
                    name=tool_use.name,
                    arguments=tool_use.input,
                    meta=meta,
                )
            result_data = extract_tool_result(result)
            self.callback.emit(
                {
                    "event": "tool_end",
                    "tool": tool_use.name,
                    "args": tool_use.input,
                    "result": result_data,
                    "call_id": call_id,
                }
            )
            content = (
                result_data if isinstance(result_data, str) else json.dumps(result_data)
            )
        except Exception as exc:  # pragma: no cover - unexpected
            self.callback.emit(
                {
                    "event": "tool_end",
                    "tool": tool_use.name,
                    "args": tool_use.input,
                    "result": {"error": str(exc)},
                    "call_id": call_id,
                }
            )
            content = f"Error: {exc}"

        return {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": content,
        }


async def _get_mcp_tools(
    mcp: Any, allowed_tools: list[str] | None = None
) -> list[ToolParam]:
    """Return a list of MCP tools formatted for Claude using ``mcp``.

    Args:
        mcp: MCP client instance
        allowed_tools: Optional list of allowed tool names to filter
    """

    if not hasattr(mcp, "list_tools"):
        return []

    tools = []
    tool_list = await mcp.list_tools()

    for tool in tool_list:
        # Filter by allowed tools if specified
        if allowed_tools and tool.name not in allowed_tools:
            continue

        tools.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema
                or {"type": "object", "properties": {}, "required": []},
            }
        )

    return tools


async def run_tools(
    config: dict[str, Any],
    on_event: Callable[[dict], None] | None = None,
) -> str:
    """Run a prompt with MCP tool-calling support via Anthropic Claude.

    Args:
        config: Complete configuration dictionary including prompt, system_instruction,
            user_instruction, extra_context, model, etc.
        on_event: Optional event callback
    """
    # Extract config values directly
    prompt = config.get("prompt", "")
    model = config.get("model", _DEFAULT_MODEL)
    system_instruction = config.get("system_instruction")
    user_instruction = config.get("user_instruction")
    extra_context = config.get("extra_context")
    transcript = config.get("transcript")
    mcp_server_url = config.get("mcp_server_url")
    tools_filter = config.get("tools")
    max_output_tokens = config.get("max_output_tokens", _DEFAULT_MAX_TOKENS)
    thinking_budget_config = config.get("thinking_budget")
    continue_from = config.get("continue_from")
    agent_id = config.get("agent_id")
    name = config.get("name")

    callback = JSONEventCallback(on_event)

    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        client = AsyncAnthropic(api_key=api_key)

        # Note: Start event is emitted by agents.py (unified event ownership)

        # Build initial messages - check for continuation first
        if continue_from:
            # Load previous conversation history using shared function
            from ..agents import parse_agent_events_to_turns

            messages = parse_agent_events_to_turns(continue_from)
            # Add new prompt as continuation
            messages.append({"role": "user", "content": prompt})
        else:
            # Fresh conversation
            messages: list[MessageParam] = []
            # Prepend transcript if provided (from day/segment input assembly)
            if transcript:
                messages.append({"role": "user", "content": transcript})
            if extra_context:
                messages.append({"role": "user", "content": extra_context})
            if user_instruction:
                messages.append({"role": "user", "content": user_instruction})
            messages.append({"role": "user", "content": prompt})

        # Initialize tools and executor if MCP server URL provided
        if mcp_server_url:
            async with create_mcp_client(str(mcp_server_url)) as mcp:
                if tools_filter and isinstance(tools_filter, list):
                    logger.info(f"Using tool filter with allowed tools: {tools_filter}")

                tools = await _get_mcp_tools(mcp, tools_filter)
                tool_executor = ToolExecutor(
                    mcp, callback, agent_id=agent_id, name=name
                )

                thinking_budget, effective_max_tokens = _resolve_agent_thinking_params(
                    max_output_tokens, thinking_budget_config
                )

                for _ in range(_MAX_TOOL_ITERATIONS):
                    # Build request params - thinking always enabled
                    create_params = {
                        "model": model,
                        "max_tokens": effective_max_tokens,
                        "system": system_instruction,
                        "messages": messages,
                        "thinking": {
                            "type": "enabled",
                            "budget_tokens": thinking_budget,
                        },
                    }
                    if tools:
                        create_params["tools"] = tools

                    response = await client.messages.create(**create_params)

                    tool_uses = []
                    final_text = ""
                    for block in response.content:
                        if getattr(block, "type", None) == "text":
                            final_text += block.text
                        elif getattr(block, "type", None) == "tool_use":
                            tool_uses.append(block)
                        elif isinstance(block, (ThinkingBlock, RedactedThinkingBlock)):
                            _emit_thinking_event(block, model, callback)

                    messages.append({"role": "assistant", "content": response.content})

                    if not tool_uses:
                        # Model is done - check for tool-only completion
                        tool_only = False
                        if not final_text:
                            final_text = "Done."
                            tool_only = True
                            logger.info(
                                "Tool-only completion, using synthetic response"
                            )
                        finish_event = {
                            "event": "finish",
                            "result": final_text,
                            "usage": _extract_usage_dict(response),
                            "ts": now_ms(),
                        }
                        if tool_only:
                            finish_event["tool_only"] = True
                        finish_reason = _normalize_finish_reason(
                            getattr(response, "stop_reason", None)
                        )
                        if finish_reason:
                            finish_event["reason"] = finish_reason
                        callback.emit(finish_event)
                        return final_text

                    results = []
                    for tool_use in tool_uses:
                        result = await tool_executor.execute_tool(tool_use)
                        results.append(result)

                    messages.append({"role": "user", "content": results})
                else:
                    # Hit iteration limit - treat as tool-only completion
                    logger.warning(
                        f"Hit max iterations ({_MAX_TOOL_ITERATIONS}), completing"
                    )
                    callback.emit(
                        {
                            "event": "finish",
                            "result": "Done.",
                            "tool_only": True,
                            "reason": "max_iterations",
                            "ts": now_ms(),
                        }
                    )
                    return "Done."
        else:
            # No MCP tools - single response only
            thinking_budget, effective_max_tokens = _resolve_agent_thinking_params(
                max_output_tokens, thinking_budget_config
            )
            create_params = {
                "model": model,
                "max_tokens": effective_max_tokens,
                "system": system_instruction,
                "messages": messages,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                },
            }

            response = await client.messages.create(**create_params)

            final_text = ""
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    final_text += block.text
                elif isinstance(block, (ThinkingBlock, RedactedThinkingBlock)):
                    _emit_thinking_event(block, model, callback)

            finish_event = {
                "event": "finish",
                "result": final_text,
                "usage": _extract_usage_dict(response),
                "ts": now_ms(),
            }
            finish_reason = _normalize_finish_reason(
                getattr(response, "stop_reason", None)
            )
            if finish_reason:
                finish_event["reason"] = finish_reason
            callback.emit(finish_event)
            return final_text
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
    "run_tools",
    "run_generate",
    "run_agenerate",
]
