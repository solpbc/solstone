#!/usr/bin/env python3
"""Anthropic Claude backed agent implementation.

This module exposes :class:`AgentSession` for interacting with Anthropic's
Claude API and is used by the ``muse-agents`` CLI.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from typing import Any, Callable, Dict, Optional

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolParam, ToolUseBlock
from mcp.types import RequestParams

from think.models import CLAUDE_SONNET_4
from think.utils import create_mcp_client

from .agents import JSONEventCallback, ThinkingEvent

# Default values are now handled internally
_DEFAULT_MODEL = CLAUDE_SONNET_4
_DEFAULT_MAX_TOKENS = 8096 * 2


def setup_logging(verbose: bool) -> logging.Logger:
    """Return app logger configured for ``verbose``."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        os.environ["ANTHROPIC_LOG"] = "debug"
    else:
        logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


class ToolExecutor:
    """Handle MCP tool execution and result formatting for Anthropic."""

    def __init__(
        self,
        mcp_client: Any,
        callback: JSONEventCallback,
        agent_id: str | None = None,
        persona: str | None = None,
    ) -> None:
        self.mcp = mcp_client
        self.callback = callback
        self.agent_id = agent_id
        self.persona = persona

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
        if self.persona:
            meta["persona"] = self.persona

        try:
            try:
                result = await self.mcp.session.call_tool(
                    name=tool_use.name,
                    arguments=tool_use.input,
                    params=RequestParams(_meta=meta) if meta else None,
                )
            except RuntimeError:
                await self.mcp.__aenter__()
                result = await self.mcp.session.call_tool(
                    name=tool_use.name,
                    arguments=tool_use.input,
                    params=RequestParams(_meta=meta) if meta else None,
                )
            # Extract content from CallToolResult if needed
            if hasattr(result, "content"):
                # MCP CallToolResult object - extract text from TextContent objects
                if isinstance(result.content, list):
                    # Handle array of content items
                    extracted_content = []
                    for item in result.content:
                        if hasattr(item, "text"):
                            # TextContent object - extract the text
                            extracted_content.append(item.text)
                        else:
                            # Other content types - keep as is
                            extracted_content.append(item)
                    # If single text content, return as string, otherwise as list
                    result_data = (
                        extracted_content[0]
                        if len(extracted_content) == 1
                        else extracted_content
                    )
                else:
                    result_data = result.content
            else:
                # Direct result (dict, string, etc.)
                result_data = result
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
    mcp: Any, allowed_tools: Optional[list[str]] = None
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


async def run_agent(
    config: Dict[str, Any],
    on_event: Optional[Callable[[dict], None]] = None,
) -> str:
    """Run a single prompt through the Anthropic Claude agent and return the response.

    Args:
        config: Complete configuration dictionary including prompt, instruction, model, etc.
        on_event: Optional event callback
    """
    # Extract values from unified config
    prompt = config.get("prompt", "")
    if not prompt:
        raise ValueError("Missing 'prompt' in config")

    model = config.get("model", _DEFAULT_MODEL)
    max_tokens = config.get("max_tokens", _DEFAULT_MAX_TOKENS)
    thinking_budget_tokens = config.get("thinking_budget_tokens", None)
    disable_mcp = config.get("disable_mcp", False)
    persona = config.get("persona", "default")

    callback = JSONEventCallback(on_event)

    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        client = AsyncAnthropic(api_key=api_key)

        callback.emit(
            {
                "event": "start",
                "prompt": prompt,
                "persona": persona,
                "model": model,
                "backend": "anthropic",
            }
        )

        # Extract instruction and extra_context from config
        system_instruction = config.get("instruction", "")
        first_user = config.get("extra_context", "")

        # Build initial messages - check for continuation first
        conversation_id = config.get("conversation_id")
        if conversation_id:
            # Load previous conversation history using shared function
            from .agents import parse_agent_events_to_turns

            messages = parse_agent_events_to_turns(conversation_id)
            # Add new prompt as continuation
            messages.append({"role": "user", "content": prompt})
        else:
            # Fresh conversation
            messages: list[MessageParam] = []
            if first_user:
                messages.append({"role": "user", "content": first_user})
            messages.append({"role": "user", "content": prompt})

        # Initialize tools and executor based on disable_mcp flag
        if not disable_mcp:
            mcp_url = config.get("mcp_server_url")
            if not mcp_url:
                raise RuntimeError("MCP server URL not provided in config")

            async with create_mcp_client(str(mcp_url)) as mcp:
                # Extract allowed tools from config
                allowed_tools = config.get("tools", None)
                if allowed_tools and isinstance(allowed_tools, list):
                    logging.getLogger(__name__).info(
                        f"Using tool filter with allowed tools: {allowed_tools}"
                    )

                tools = await _get_mcp_tools(mcp, allowed_tools)
                agent_id = config.get("agent_id")
                tool_executor = ToolExecutor(
                    mcp, callback, agent_id=agent_id, persona=persona
                )

                while True:
                    # Configure thinking for supported models
                    thinking_config = None
                    if model in [
                        "claude-opus-4-20250514",
                        "claude-sonnet-4-20250514",
                        "claude-sonnet-3-7-20241124",
                    ]:
                        # Use config value if provided, otherwise calculate default
                        if thinking_budget_tokens is not None:
                            budget_tokens = thinking_budget_tokens
                        elif max_tokens >= 2048:
                            # Only enable thinking if we have enough tokens
                            budget_tokens = min(10000, max_tokens - 1000)
                        else:
                            # Skip thinking for small token limits
                            thinking_config = None
                            budget_tokens = None

                        if budget_tokens is not None:
                            thinking_config = {
                                "type": "enabled",
                                "budget_tokens": budget_tokens,
                            }

                    # Only include tools parameter if we have tools
                    create_params = {
                        "model": model,
                        "max_tokens": max_tokens,
                        "system": system_instruction,
                        "messages": messages,
                    }
                    if tools:
                        create_params["tools"] = tools
                    if thinking_config is not None:
                        create_params["thinking"] = thinking_config

                    response = await client.messages.create(**create_params)

                    tool_uses = []
                    final_text = ""
                    for block in response.content:
                        if getattr(block, "type", None) == "text":
                            final_text += block.text
                        elif getattr(block, "type", None) == "tool_use":
                            tool_uses.append(block)
                        elif getattr(block, "type", None) == "thinking":
                            # Emit thinking event with the reasoning content
                            thinking_event: ThinkingEvent = {
                                "event": "thinking",
                                "ts": int(time.time() * 1000),
                                "summary": block.thinking,
                                "model": model,
                            }
                            callback.emit(thinking_event)

                    messages.append({"role": "assistant", "content": response.content})

                    if not tool_uses:
                        # Use agent_id as conversation_id for continuations
                        agent_id = config.get("agent_id", "unknown")
                        callback.emit(
                            {
                                "event": "finish",
                                "result": final_text,
                                "conversation_id": agent_id,
                            }
                        )
                        return final_text

                    results = []
                    for tool_use in tool_uses:
                        result = await tool_executor.execute_tool(tool_use)
                        results.append(result)

                    messages.append({"role": "user", "content": results})
        else:
            # No MCP tools - single response only
            # Configure thinking for supported models
            thinking_config = None
            if model in [
                "claude-opus-4-20250514",
                "claude-sonnet-4-20250514",
                "claude-sonnet-3-7-20241124",
            ]:
                # Use config value if provided, otherwise calculate default
                if thinking_budget_tokens is not None:
                    budget_tokens = thinking_budget_tokens
                elif max_tokens >= 2048:
                    # Only enable thinking if we have enough tokens
                    budget_tokens = min(10000, max_tokens - 1000)
                else:
                    # Skip thinking for small token limits
                    thinking_config = None
                    budget_tokens = None

                if budget_tokens is not None:
                    thinking_config = {
                        "type": "enabled",
                        "budget_tokens": budget_tokens,
                    }

            # Create params without tools parameter when MCP is disabled
            create_params = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_instruction,
                "messages": messages,
            }
            if thinking_config is not None:
                create_params["thinking"] = thinking_config

            response = await client.messages.create(**create_params)

            final_text = ""
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    final_text += block.text
                elif getattr(block, "type", None) == "thinking":
                    # Emit thinking event with the reasoning content
                    thinking_event: ThinkingEvent = {
                        "event": "thinking",
                        "ts": int(time.time() * 1000),
                        "summary": block.thinking,
                        "model": model,
                    }
                    callback.emit(thinking_event)

            # Use agent_id as conversation_id for continuations
            agent_id = config.get("agent_id", "unknown")
            callback.emit(
                {"event": "finish", "result": final_text, "conversation_id": agent_id}
            )
            return final_text
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


async def run_prompt(
    prompt: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    persona: str = "default",
) -> str:
    """Convenience helper to run ``prompt`` (alias for run_agent)."""
    return await run_agent(prompt, config=config, on_event=on_event, persona=persona)


__all__ = [
    "run_agent",
    "run_prompt",
    "setup_logging",
]
