# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Shared utilities and types for AI providers.

This module contains:
- Event TypedDicts emitted by providers during agent execution
- GenerateResult TypedDict returned by run_generate/run_agenerate
- JSONEventCallback for event emission
- Utility functions for common provider operations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union

from typing_extensions import Required, TypedDict

from think.utils import now_ms

# ---------------------------------------------------------------------------
# Event Types
# ---------------------------------------------------------------------------


class ToolStartEvent(TypedDict, total=False):
    """Event emitted when a tool starts."""

    event: Literal["tool_start"]
    ts: int
    tool: str
    args: Optional[dict[str, Any]]
    call_id: Optional[str]  # Unique ID to pair with tool_end event


class ToolEndEvent(TypedDict, total=False):
    """Event emitted when a tool finishes."""

    event: Literal["tool_end"]
    ts: int
    tool: str
    args: Optional[dict[str, Any]]
    result: Any
    call_id: Optional[str]  # Matches the call_id from tool_start


class StartEvent(TypedDict):
    """Event emitted when an agent run begins."""

    event: Literal["start"]
    ts: int
    prompt: str
    name: str
    model: str
    provider: str


class FinishEvent(TypedDict):
    """Event emitted when an agent run finishes successfully."""

    event: Literal["finish"]
    ts: int
    result: str


class ErrorEvent(TypedDict, total=False):
    """Event emitted when an error occurs."""

    event: Literal["error"]
    ts: int
    error: str
    trace: Optional[str]


class AgentUpdatedEvent(TypedDict):
    """Event emitted when the agent context changes."""

    event: Literal["agent_updated"]
    ts: int
    agent: str


class ThinkingEvent(TypedDict, total=False):
    """Event emitted when thinking/reasoning summaries are available.

    For Anthropic models, may include a signature for verification when
    passing thinking blocks back during tool use continuations.
    For redacted thinking, summary will contain "[redacted]" and
    redacted_data will contain the encrypted content.
    """

    event: Required[Literal["thinking"]]
    ts: Required[int]
    summary: Required[str]
    model: Optional[str]
    signature: Optional[str]  # Anthropic thinking block signature
    redacted_data: Optional[str]  # Encrypted data for redacted thinking


Event = Union[
    ToolStartEvent,
    ToolEndEvent,
    StartEvent,
    FinishEvent,
    ErrorEvent,
    ThinkingEvent,
    AgentUpdatedEvent,
]


# ---------------------------------------------------------------------------
# GenerateResult
# ---------------------------------------------------------------------------


class GenerateResult(TypedDict, total=False):
    """Result from provider run_generate/run_agenerate functions.

    Structured result that allows the wrapper to handle cross-cutting concerns
    like token logging and JSON validation centrally.

    The thinking field contains dicts with: summary (str), signature (optional str),
    redacted_data (optional str for Anthropic redacted thinking).
    """

    text: Required[str]  # Response text
    usage: Optional[dict]  # Normalized usage dict (input_tokens, output_tokens, etc.)
    finish_reason: Optional[str]  # Normalized: "stop", "max_tokens", "safety", etc.
    thinking: Optional[list]  # List of thinking block dicts


# ---------------------------------------------------------------------------
# JSONEventCallback
# ---------------------------------------------------------------------------


class JSONEventCallback:
    """Emit JSON events via a callback."""

    def __init__(self, callback: Optional[Callable[[Event], None]] = None) -> None:
        self.callback = callback

    def emit(self, data: Event) -> None:
        if "ts" not in data:
            data = {**data, "ts": now_ms()}
        if self.callback:
            self.callback(data)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Agent Config Extraction
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Validated agent configuration extracted from config dict."""

    prompt: str
    model: str
    name: str
    agent_id: Optional[str]
    max_output_tokens: int
    thinking_budget: Optional[int]
    mcp_server_url: Optional[str]
    continue_from: Optional[str]
    system_instruction: str
    extra_context: str
    user_instruction: str
    tools: Optional[list[str]]
    provider: str

    # Original config for provider-specific access
    raw_config: dict


def extract_agent_config(config: dict, default_max_tokens: int = 8192) -> AgentConfig:
    """Extract and validate agent configuration.

    Parameters
    ----------
    config
        Raw config dict from cortex.
    default_max_tokens
        Default max_output_tokens if not specified in config.

    Returns
    -------
    AgentConfig
        Validated configuration dataclass.

    Raises
    ------
    ValueError
        If required fields are missing.
    """
    prompt = config.get("prompt", "")
    if not prompt:
        raise ValueError("Missing 'prompt' in config")

    model = config.get("model")
    if not model:
        raise ValueError("Missing 'model' in config - should be set by Cortex")

    return AgentConfig(
        prompt=prompt,
        model=model,
        name=config.get("name", "default"),
        agent_id=config.get("agent_id"),
        max_output_tokens=config.get("max_output_tokens", default_max_tokens),
        thinking_budget=config.get("thinking_budget"),
        mcp_server_url=config.get("mcp_server_url"),
        continue_from=config.get("continue_from"),
        system_instruction=config.get("system_instruction", ""),
        extra_context=config.get("extra_context", ""),
        user_instruction=config.get("user_instruction", ""),
        tools=config.get("tools"),
        provider=config.get("provider", "google"),
        raw_config=config,
    )


# ---------------------------------------------------------------------------
# MCP Tool Result Extraction
# ---------------------------------------------------------------------------


def extract_tool_result(result: Any) -> Any:
    """Extract content from MCP CallToolResult.

    Handles:
    - CallToolResult with content list of TextContent objects
    - CallToolResult with single content
    - Direct result values (dict, string, etc.)

    Parameters
    ----------
    result
        Raw result from MCP tool call.

    Returns
    -------
    Any
        Normalized result suitable for event logging and LLM responses.
    """
    if not hasattr(result, "content"):
        return result

    content = result.content
    if not isinstance(content, list):
        return content

    # Extract text from TextContent objects
    extracted = []
    for item in content:
        if hasattr(item, "text"):
            extracted.append(item.text)
        else:
            extracted.append(item)

    # Return single item directly, otherwise list
    return extracted[0] if len(extracted) == 1 else extracted


__all__ = [
    "Event",
    "GenerateResult",
    "JSONEventCallback",
    "ThinkingEvent",
    "extract_agent_config",
    "extract_tool_result",
]
