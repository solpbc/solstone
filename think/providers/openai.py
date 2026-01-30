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
    Model name to use.
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

Note: GPT-5+ reasoning models don't support custom temperature (fixed at 1.0)
and reasoning is always enabled with medium effort.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from typing import Any, Callable
from urllib.parse import urlparse, urlunparse

from agents import (
    Agent,
    Runner,
    SQLiteSession,
)
from agents.items import (
    MessageOutputItem,
    ReasoningItem,
    ToolCallItem,
    ToolCallOutputItem,
)
from agents.mcp.server import MCPServerStreamableHttp

# Try to import ToolFilterStatic if available
try:
    from agents.mcp.server import ToolFilterStatic
except ImportError:
    ToolFilterStatic = None  # type: ignore
from agents.model_settings import ModelSettings, Reasoning
from agents.run import RunConfig

# Optional: used only for raw text deltas if available
try:
    from openai.types.responses import ResponseTextDeltaEvent  # type: ignore
except Exception:  # pragma: no cover
    ResponseTextDeltaEvent = object  # type: ignore

# Agent configuration is now loaded via get_agent() in cortex.py

from think.models import GPT_5
from think.utils import now_ms

from .shared import (
    GenerateResult,
    JSONEventCallback,
    ThinkingEvent,
    extract_agent_config,
)


def _convert_turns_to_items(turns: list[dict]) -> list[dict]:
    """Convert turn history to OpenAI SDK item format.

    Args:
        turns: List of dicts with 'role' and 'content' keys

    Returns:
        List of items in SDK format with type/role/content structure
    """
    items = []
    for turn in turns:
        role = turn.get("role")
        content = turn.get("content", "")

        if role in ("user", "assistant") and content:
            # Responses API requires input_text for user, output_text for assistant
            content_type = "input_text" if role == "user" else "output_text"
            items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [{"type": content_type, "text": content}],
                }
            )

    return items


# Default values
_DEFAULT_MAX_TOKENS = 16384
_DEFAULT_MAX_TURNS = 64

LOG = logging.getLogger("think.providers.openai")

# Default reasoning config for all GPT-5 models
_DEFAULT_REASONING = Reasoning(effort="medium", summary="detailed")


def _normalize_streamable_http_uri(http_uri: str) -> str:
    """
    Ensure the Streamable HTTP MCP URL points at the MCP endpoint.

    If no path or '/', append '/mcp'.
    If already '/mcp' (with or without trailing '/'), keep unchanged.
    Otherwise, leave as-is (user may have a reverse-proxy path).
    """
    try:
        parsed = urlparse(http_uri.strip())
        path = parsed.path or ""
        if path in ("", "/"):
            path = "/mcp"
        elif path.rstrip("/") == "/mcp":
            path = "/mcp"
        new_parsed = parsed._replace(path=path)
        return urlunparse(new_parsed)
    except Exception:
        return http_uri


def _json_maybe_loads(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def _extract_tool_name(raw_call: Any) -> str:
    for attr in ("name", "tool", "tool_name"):
        if hasattr(raw_call, attr):
            val = getattr(raw_call, attr)
            if isinstance(val, str) and val:
                return val
    return type(raw_call).__name__


def _extract_tool_call_id(raw_call: Any) -> str | None:
    for attr in ("id", "call_id", "tool_call_id"):
        if hasattr(raw_call, attr):
            val = getattr(raw_call, attr)
            if isinstance(val, str) and val:
                return val
    return None


def _extract_tool_args(raw_call: Any) -> Any:
    """
    Function tools usually expose `.arguments` (often JSON string).
    Some tools expose `.input_json` / `.input`.
    """
    if hasattr(raw_call, "arguments"):
        return _json_maybe_loads(getattr(raw_call, "arguments"))
    for attr in ("input_json", "input", "arguments_json"):
        if hasattr(raw_call, attr):
            return getattr(raw_call, attr)
    if isinstance(raw_call, dict):
        if "arguments" in raw_call:
            return _json_maybe_loads(raw_call["arguments"])
        if "input" in raw_call:
            return raw_call["input"]
    return None


def _extract_text_parts(raw: Any, attr: str) -> str | None:
    """Extract text from a list of text parts on a raw item attribute.

    Args:
        raw: Raw item object
        attr: Attribute name to extract (e.g., "summary", "content")

    Returns:
        Joined text or None if not found/empty
    """
    if not hasattr(raw, attr):
        return None
    parts = getattr(raw, attr)
    if not parts:
        return None
    try:
        texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
        return "".join(texts) if texts else None
    except Exception:
        return None


async def run_tools(
    config: dict[str, Any],
    on_event: Callable[[dict], None] | None = None,
) -> str:
    """Run a prompt with MCP tool-calling support via OpenAI Agents SDK.

    Uses streaming and emits JSON events.

    Args:
        config: Complete configuration dictionary including prompt, system_instruction,
            user_instruction, extra_context, model, etc.
        on_event: Optional event callback
    """
    ac = extract_agent_config(config, default_max_tokens=_DEFAULT_MAX_TOKENS)
    max_turns = config.get("max_turns", _DEFAULT_MAX_TURNS)

    LOG.info("Running agent with model %s", ac.model)
    cb = JSONEventCallback(on_event)
    start_event: dict = {
        "event": "start",
        "prompt": ac.prompt,
        "name": ac.name,
        "model": ac.model,
        "provider": "openai",
        "ts": now_ms(),
    }
    if ac.continue_from:
        start_event["continue_from"] = ac.continue_from
    cb.emit(start_event)

    # Model settings: always enable reasoning with detailed summaries
    model_settings = ModelSettings(
        max_tokens=ac.max_output_tokens,
        reasoning=_DEFAULT_REASONING,
    )

    # Initialize MCP server
    mcp_server = None
    if ac.mcp_server_url:
        http_uri = _normalize_streamable_http_uri(str(ac.mcp_server_url))

        # Configure tool filter if tools are specified
        tool_filter = None
        if ac.tools and isinstance(ac.tools, list) and ToolFilterStatic:
            # Create a tool filter with allowed tools
            tool_filter = ToolFilterStatic(allowed_tool_names=ac.tools)
            LOG.info(f"Using tool filter with allowed tools: {ac.tools}")
        elif ac.tools:
            LOG.warning(
                "Tool filtering requested but ToolFilterStatic not available in this version"
            )

        mcp_params = {"url": http_uri}
        headers: dict[str, str] = {}
        if ac.agent_id:
            headers["X-Agent-Id"] = str(ac.agent_id)
        if ac.name:
            headers["X-Agent-Name"] = ac.name
        if headers:
            mcp_params["headers"] = headers

        mcp_server = MCPServerStreamableHttp(
            params=mcp_params,
            cache_tools_list=True,
            tool_filter=tool_filter,
            # Increase tool invocation timeout to avoid premature cancellations
            client_session_timeout_seconds=15.0,
        )

    # Keep a map of in-flight tools so we can pair outputs with args
    pending_tools: dict[str, dict[str, Any]] = {}

    # Accumulate streamed text chunks as a fallback (if final_output missing)
    streamed_text: list[str] = []

    # Create session and load history if continuing conversation
    session_id = ac.continue_from or ac.agent_id or f"session-{int(time.time())}"
    session = SQLiteSession(session_id=session_id, db_path=":memory:")

    # Load conversation history if continuing
    if ac.continue_from:
        from ..agents import parse_agent_events_to_turns

        turns = parse_agent_events_to_turns(ac.continue_from)
        if turns:
            items = _convert_turns_to_items(turns)
            await session.add_items(items)
    else:
        # Fresh conversation - add transcript, context and user instruction as initial messages
        initial_turns = []
        # Prepend transcript if provided (from day/segment input assembly)
        if ac.transcript:
            initial_turns.append({"role": "user", "content": ac.transcript})
        if ac.extra_context:
            initial_turns.append({"role": "user", "content": ac.extra_context})
        if ac.user_instruction:
            initial_turns.append({"role": "user", "content": ac.user_instruction})
        if initial_turns:
            initial_items = _convert_turns_to_items(initial_turns)
            await session.add_items(initial_items)

    try:
        # Handle MCP server context manager conditionally
        if mcp_server:
            mcp_context = mcp_server
        else:
            # Create a dummy context manager when no MCP server URL provided
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def dummy_context():
                yield

            mcp_context = dummy_context()

        async with mcp_context:
            # Create agent with or without MCP servers
            mcp_servers_list = [mcp_server] if mcp_server else []
            agent = Agent(
                name="solstoneCLI",
                instructions=ac.system_instruction,
                model=ac.model,
                model_settings=model_settings,
                mcp_servers=mcp_servers_list,
            )

            result = Runner.run_streamed(
                agent,
                input=ac.prompt,
                session=session,
                run_config=RunConfig(tracing_disabled=True),  # per docs
                max_turns=max_turns,
            )

            async for ev in result.stream_events():
                # 1) Raw deltas (Responses API events)
                if ev.type == "raw_response_event":
                    data = getattr(ev, "data", None)
                    # If we have text deltas, capture them (optional)
                    if isinstance(data, ResponseTextDeltaEvent) and isinstance(
                        getattr(data, "delta", None), str
                    ):
                        streamed_text.append(data.delta)
                    continue

                # 2) Agent updates (handoffs)
                if ev.type == "agent_updated_stream_event":
                    new_agent = getattr(ev, "new_agent", None)
                    cb.emit(
                        {
                            "event": "agent_updated",
                            "agent": getattr(new_agent, "name", None),
                            "ts": now_ms(),
                        }
                    )
                    continue

                # 3) Run items (messages, tools, reasoning, etc.)
                if ev.type == "run_item_stream_event":
                    name = ev.name
                    item = ev.item

                    # Tool call started — capture args
                    if name == "tool_called" and isinstance(item, ToolCallItem):
                        raw = item.raw_item
                        tool_name = _extract_tool_name(raw)
                        call_id = _extract_tool_call_id(raw) or tool_name
                        args = _extract_tool_args(raw)
                        pending_tools[call_id] = {"tool": tool_name, "args": args}
                        cb.emit(
                            {
                                "event": "tool_start",
                                "tool": tool_name,
                                "args": args,
                                "call_id": call_id,
                                "ts": now_ms(),
                            }
                        )

                    # Tool output finished — mirror end, include original args if we have them
                    elif name == "tool_output" and isinstance(item, ToolCallOutputItem):
                        raw = item.raw_item
                        # raw_item is a TypedDict (dict), not an object - use dict access
                        if isinstance(raw, dict):
                            call_id = (
                                raw.get("tool_call_id")
                                or raw.get("call_id")
                                or raw.get("id")
                            )
                        else:
                            call_id = (
                                getattr(raw, "tool_call_id", None)
                                or getattr(raw, "call_id", None)
                                or getattr(raw, "id", None)
                            )
                        meta = pending_tools.pop(call_id, {}) if call_id else {}
                        cb.emit(
                            {
                                "event": "tool_end",
                                "tool": meta.get("tool", "tool"),
                                "args": meta.get("args"),
                                "result": item.output,
                                "call_id": call_id,
                                "ts": now_ms(),
                            }
                        )

                    # Reasoning / "thinking" item created
                    elif name == "reasoning_item_created" and isinstance(
                        item, ReasoningItem
                    ):
                        raw = item.raw_item
                        # Try summary first, then fall back to content
                        summary_text = _extract_text_parts(
                            raw, "summary"
                        ) or _extract_text_parts(raw, "content")

                        if summary_text:
                            thinking_event: ThinkingEvent = {
                                "event": "thinking",
                                "summary": summary_text,
                                "model": ac.model,
                                "ts": now_ms(),
                            }
                            cb.emit(thinking_event)

                    # Completed assistant message (final text will be read from result)
                    elif name == "message_output_created" and isinstance(
                        item, MessageOutputItem
                    ):
                        pass  # no-op

            # Done streaming — prefer result.final_output, else join deltas
            final_text = getattr(result, "final_output", None)
            if not isinstance(final_text, str) or not final_text:
                final_text = "".join(streamed_text)

            # Check for tool-only completion (no text output)
            tool_only = False
            if not final_text:
                final_text = "Done."
                tool_only = True
                LOG.info("Tool-only completion, using synthetic response")

            # Extract usage information from result
            usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
            usage_dict = None
            if usage:
                usage_dict = {
                    "requests": getattr(usage, "requests", None),
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "details": {
                        "input": getattr(usage, "input_tokens_details", None)
                        and {
                            "cached_tokens": getattr(
                                usage.input_tokens_details, "cached_tokens", None
                            )
                        },
                        "output": getattr(usage, "output_tokens_details", None)
                        and {
                            "reasoning_tokens": getattr(
                                usage.output_tokens_details, "reasoning_tokens", None
                            )
                        },
                    },
                }

            finish_event = {
                "event": "finish",
                "result": final_text,
                "usage": usage_dict,
                "ts": now_ms(),
            }
            if tool_only:
                finish_event["tool_only"] = True
            cb.emit(finish_event)
            return final_text

    except Exception as exc:
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        cb.emit({"event": "error", "error": str(exc), "trace": trace, "ts": now_ms()})
        setattr(exc, "_evented", True)
        raise
    finally:
        # IMPORTANT: Don't explicitly close the SQLiteSession while streaming;
        # the SDK continues to read from it in background tasks and closing can race.
        pass


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

    # Build request kwargs with reasoning enabled
    # Note: GPT-5+ models require max_completion_tokens instead of max_tokens
    # Note: Reasoning models don't support custom temperature (only 1.0)
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_output_tokens,
        "reasoning_effort": "medium",
    }

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

    # Build request kwargs with reasoning enabled
    # Note: GPT-5+ models require max_completion_tokens instead of max_tokens
    # Note: Reasoning models don't support custom temperature (only 1.0)
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_output_tokens,
        "reasoning_effort": "medium",
    }

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
    "run_tools",
    "run_generate",
    "run_agenerate",
]
