#!/usr/bin/env python3
"""Anthropic Claude backed agent implementation.

This module exposes :class:`AgentSession` for interacting with Anthropic's
Claude API and is used by the ``think-agents`` CLI.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from typing import Any, Callable, Optional

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolParam, ToolUseBlock

from .agents import BaseAgentSession, JSONEventCallback
from .models import CLAUDE_OPUS_4
from .utils import agent_instructions, create_mcp_client

DEFAULT_MODEL = CLAUDE_OPUS_4
DEFAULT_MAX_TOKENS = 4096


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
        self.callback.emit(
            {
                "event": "tool_start",
                "tool": tool_use.name,
                "args": tool_use.input,
            }
        )

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
            self.callback.emit(
                {
                    "event": "tool_end",
                    "tool": tool_use.name,
                    "result": result,
                }
            )
            content = result if isinstance(result, str) else json.dumps(result)
        except Exception as exc:  # pragma: no cover - unexpected
            self.callback.emit(
                {
                    "event": "tool_end",
                    "tool": tool_use.name,
                    "result": {"error": str(exc)},
                }
            )
            content = f"Error: {exc}"

        return {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": content,
        }


class AgentSession(BaseAgentSession):
    """Context manager running Claude with MCP tools."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        on_event: Optional[Callable[[dict], None]] = None,
        persona: str = "default",
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._callback = JSONEventCallback(on_event)
        self.client: AsyncAnthropic | None = None
        self.messages: list[MessageParam] = []
        self.system_instruction = ""
        self._history: list[dict[str, str]] = []
        self.persona = persona

    async def __aenter__(self) -> "AgentSession":
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self.client = AsyncAnthropic(api_key=api_key)

            self.system_instruction, first_user, _ = agent_instructions(self.persona)

            if first_user:
                self.messages.append({"role": "user", "content": first_user})
                self._history.append({"role": "user", "content": first_user})

            return self
        except Exception as exc:
            self._callback.emit(
                {
                    "event": "error",
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                }
            )
            setattr(exc, "_evented", True)
            raise

    @property
    def history(self) -> list[dict[str, str]]:
        """Return the accumulated chat history."""
        return list(self._history)

    def add_history(self, role: str, text: str) -> None:
        """Queue a history message for the next run."""
        self.messages.append({"role": role, "content": text})
        self._history.append({"role": role, "content": text})

    async def __aexit__(self, exc_type, exc, tb) -> None:
        pass

    async def _get_mcp_tools(self, mcp: Any) -> list[ToolParam]:
        """Return a list of MCP tools formatted for Claude using ``mcp``."""

        if not hasattr(mcp, "list_tools"):
            return []

        tools = []
        tool_list = await mcp.list_tools()

        for tool in tool_list:
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema
                    or {"type": "object", "properties": {}, "required": []},
                }
            )

        return tools

    async def run(self, prompt: str) -> str:
        """Run ``prompt`` through Claude and return the result."""
        try:
            if self.client is None:
                raise RuntimeError("AgentSession not initialized")

            self.messages.append({"role": "user", "content": prompt})
            self._history.append({"role": "user", "content": prompt})

            self._callback.emit(
                {
                    "event": "start",
                    "prompt": prompt,
                    "persona": self.persona,
                    "model": self.model,
                }
            )

            async with create_mcp_client("fastmcp") as mcp:
                tools = await self._get_mcp_tools(mcp)
                tool_executor = ToolExecutor(mcp, self._callback)

                while True:
                    response = await self.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        system=self.system_instruction,
                        messages=self.messages,
                        tools=tools if tools else None,
                    )

                    tool_uses = []
                    final_text = ""
                    for block in response.content:
                        if getattr(block, "type", None) == "text":
                            final_text += block.text
                        elif getattr(block, "type", None) == "tool_use":
                            tool_uses.append(block)

                    self.messages.append(
                        {"role": "assistant", "content": response.content}
                    )

                    if not tool_uses:
                        self._callback.emit({"event": "finish", "result": final_text})
                        self._history.append(
                            {"role": "assistant", "content": final_text}
                        )
                        return final_text

                    results = []
                    for tool_use in tool_uses:
                        result = await tool_executor.execute_tool(tool_use)
                        results.append(result)

                    self.messages.append({"role": "user", "content": results})
        except Exception as exc:
            self._callback.emit(
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
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_event: Optional[Callable[[dict], None]] = None,
    persona: str = "default",
) -> str:
    """Convenience helper to run ``prompt`` with a temporary session."""
    async with AgentSession(
        model, max_tokens=max_tokens, on_event=on_event, persona=persona
    ) as ag:
        return await ag.run(prompt)


__all__ = [
    "AgentSession",
    "run_prompt",
    "setup_logging",
    "DEFAULT_MODEL",
    "DEFAULT_MAX_TOKENS",
]
