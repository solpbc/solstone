#!/usr/bin/env python3
"""Gemini backed agent implementation.

This module exposes :class:`AgentSession` for interacting with Google's Gemini
API. It is utilised by the unified ``think-agents`` CLI.
"""

from __future__ import annotations

import logging
import os
import traceback
from typing import Any, Callable, Optional

from fastmcp import Client
from google import genai
from google.genai import types

from .agents import BaseAgentSession, JSONEventCallback
from .models import GEMINI_FLASH
from .utils import agent_instructions, create_mcp_client

DEFAULT_MODEL = GEMINI_FLASH
DEFAULT_MAX_TOKENS = 8192


def setup_logging(verbose: bool) -> logging.Logger:
    """Return app logger configured for ``verbose``."""

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    return logging.getLogger(__name__)


class ToolLoggingHooks:
    """Wrap ``session.call_tool`` to emit events."""

    def __init__(self, writer: JSONEventCallback) -> None:
        self.writer = writer

    def attach(self, session: Any) -> None:
        original = session.call_tool

        async def wrapped(name: str, arguments: dict | None = None, **kwargs) -> Any:
            self.writer.emit({"event": "tool_start", "tool": name, "args": arguments})
            result = await original(name=name, arguments=arguments, **kwargs)
            self.writer.emit({"event": "tool_end", "tool": name, "result": result})
            return result

        session.call_tool = wrapped  # type: ignore[assignment]


class AgentSession(BaseAgentSession):
    """Context manager running Gemini with MCP tools."""

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
        self._mcp: Client | None = None
        self.client: genai.Client | None = None
        self.chat: genai.chats.AsyncChat | None = None
        self.system_instruction = ""
        self._history: list[dict[str, str]] = []
        self.persona = persona

    async def __aenter__(self) -> "AgentSession":
        try:
            self._mcp = create_mcp_client("fastmcp")
            await self._mcp.__aenter__()

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY not set")
            self.client = genai.Client(api_key=api_key)

            self.system_instruction, first_user, _ = agent_instructions(self.persona)

            ToolLoggingHooks(self._callback).attach(self._mcp.session)
            self.chat = self.client.aio.chats.create(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction
                ),
                history=[
                    types.Content(role="user", parts=[types.Part(text=first_user)])
                ],
            )
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
        """Return the accumulated chat history as ``role``/``content`` dicts."""

        return list(self._history)

    def add_history(self, role: str, text: str) -> None:
        """Record a message to the chat history."""
        if self.chat is not None:
            self.chat.record_history(
                types.Content(role=role, parts=[types.Part(text=text)])
            )
        self._history.append({"role": role, "content": text})

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._mcp:
            await self._mcp.__aexit__(exc_type, exc, tb)

    async def run(self, prompt: str) -> str:
        """Run ``prompt`` through Gemini and return the result."""
        try:
            if self._mcp is None or self.client is None or self.chat is None:
                raise RuntimeError("AgentSession not initialized")
            self._callback.emit(
                {
                    "event": "start",
                    "prompt": prompt,
                    "persona": self.persona,
                    "model": self.model,
                }
            )
            session = self._mcp.session
            cfg = types.GenerateContentConfig(
                max_output_tokens=self.max_tokens,
                tools=[session],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                ),
            )
            response = await self.chat.send_message(prompt, config=cfg)
            text = response.text
            self._callback.emit({"event": "finish", "result": text})
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": text})
            return text
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
    """Convenience helper to run ``prompt`` with a temporary :class:`AgentSession`."""

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
