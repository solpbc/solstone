#!/usr/bin/env python3
"""Anthropic Claude backed agent implementation.

This module exposes :class:`AgentSession` for interacting with Anthropic's
Claude API and is used by the ``think-agents`` CLI.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from typing import Any, Callable, Optional

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolParam, ToolUseBlock

from .agents import JSONEventCallback, ThinkingEvent
from .models import CLAUDE_OPUS_4, CLAUDE_SONNET_4
from .utils import agent_instructions, create_mcp_client

DEFAULT_MODEL = CLAUDE_SONNET_4
DEFAULT_MAX_TOKENS = 8096*2


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

    def __init__(self, mcp_client: Any, callback: JSONEventCallback) -> None:
        self.mcp = mcp_client
        self.callback = callback

    async def execute_tool(self, tool_use: ToolUseBlock) -> dict:
        """Execute ``tool_use`` and return a Claude ``tool_result`` block."""
        self.callback.emit({
            "event": "tool_start",
            "tool": tool_use.name,
            "args": tool_use.input,
        })

        try:
            try:
                result = await self.mcp.session.call_tool(
                    name=tool_use.name,
                    arguments=tool_use.input,
                )
            except RuntimeError:
                await self.mcp.__aenter__()
                result = await self.mcp.session.call_tool(
                    name=tool_use.name,
                    arguments=tool_use.input,
                )
            # Extract content from CallToolResult if needed
            if hasattr(result, 'content'):
                # MCP CallToolResult object - extract text from TextContent objects
                if isinstance(result.content, list):
                    # Handle array of content items
                    extracted_content = []
                    for item in result.content:
                        if hasattr(item, 'text'):
                            # TextContent object - extract the text
                            extracted_content.append(item.text)
                        else:
                            # Other content types - keep as is
                            extracted_content.append(item)
                    # If single text content, return as string, otherwise as list
                    result_data = extracted_content[0] if len(extracted_content) == 1 else extracted_content
                else:
                    result_data = result.content
            else:
                # Direct result (dict, string, etc.)
                result_data = result
            self.callback.emit({
                "event": "tool_end",
                "tool": tool_use.name,
                "result": result_data,
            })
            content = result_data if isinstance(result_data, str) else json.dumps(result_data)
        except Exception as exc:  # pragma: no cover - unexpected
            self.callback.emit({
                "event": "tool_end",
                "tool": tool_use.name,
                "result": {"error": str(exc)},
            })
            content = f"Error: {exc}"

        return {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": content,
        }


async def _get_mcp_tools(mcp: Any) -> list[ToolParam]:
    """Return a list of MCP tools formatted for Claude using ``mcp``."""

    if not hasattr(mcp, "list_tools"):
        return []

    tools = []
    tool_list = await mcp.list_tools()

    for tool in tool_list:
        tools.append({
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema
            or {"type": "object", "properties": {}, "required": []},
        })

    return tools


async def run_agent(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_event: Optional[Callable[[dict], None]] = None,
    persona: str = "default",
) -> str:
    """Run a single prompt through the Anthropic Claude agent and return the response."""
    callback = JSONEventCallback(on_event)

    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        client = AsyncAnthropic(api_key=api_key)

        callback.emit({
            "event": "start",
            "prompt": prompt,
            "persona": persona,
            "model": model,
        })

        async with create_mcp_client() as mcp:
            system_instruction, first_user, _ = agent_instructions(persona)
            tools = await _get_mcp_tools(mcp)
            tool_executor = ToolExecutor(mcp, callback)

            # Build initial messages
            messages: list[MessageParam] = []
            if first_user:
                messages.append({"role": "user", "content": first_user})
            messages.append({"role": "user", "content": prompt})

            while True:
                # Configure thinking for supported models
                thinking_config = None
                if model in ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-sonnet-3-7-20241124"]:
                    thinking_config = {
                        "type": "enabled",
                        "budget_tokens": min(10000, max_tokens - 1000)  # Reserve some tokens for final response
                    }

                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_instruction,
                    messages=messages,
                    tools=tools if tools else None,
                    thinking=thinking_config,
                )

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
                            "model": model
                        }
                        callback.emit(thinking_event)

                messages.append({"role": "assistant", "content": response.content})

                if not tool_uses:
                    callback.emit({"event": "finish", "result": final_text})
                    return final_text

                results = []
                for tool_use in tool_uses:
                    result = await tool_executor.execute_tool(tool_use)
                    results.append(result)

                messages.append({"role": "user", "content": results})
    except Exception as exc:
        callback.emit({
            "event": "error",
            "error": str(exc),
            "trace": traceback.format_exc(),
        })
        setattr(exc, "_evented", True)
        raise


async def run_prompt(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_event: Optional[Callable[[dict], None]] = None,
    persona: str = "default",
) -> str:
    """Convenience helper to run ``prompt`` (alias for run_agent)."""
    return await run_agent(
        prompt, model=model, max_tokens=max_tokens, on_event=on_event, persona=persona
    )


__all__ = [
    "run_agent",
    "run_prompt",
    "setup_logging",
    "DEFAULT_MODEL",
    "DEFAULT_MAX_TOKENS",
]
