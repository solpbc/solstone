#!/usr/bin/env python3
"""Gemini backed agent implementation.

This module exposes :class:`AgentSession` for interacting with Google's Gemini
API. It is utilised by the unified ``think-agents`` CLI.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from typing import Any, Callable, Optional

from google import genai
from google.genai import types

from .agents import JSONEventCallback, ThinkingEvent
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


async def run_agent(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_event: Optional[Callable[[dict], None]] = None,
    persona: str = "default",
) -> str:
    """Run a single prompt through the Google Gemini agent and return the response."""
    callback = JSONEventCallback(on_event)

    try:
        # Check API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        callback.emit(
            {
                "event": "start",
                "prompt": prompt,
                "persona": persona,
                "model": model,
            }
        )

        # Create fresh client and MCP for each run (prevents event loop issues)
        async with create_mcp_client() as mcp:
            client = genai.Client(api_key=api_key)

            # Get system instruction
            system_instruction, first_user, _ = agent_instructions(persona)

            # Build minimal history for chat
            history = []
            if first_user:
                history.append(
                    types.Content(role="user", parts=[types.Part(text=first_user)])
                )

            # Create fresh chat session
            chat = client.aio.chats.create(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                ),
                history=history,
            )

            # Attach tool logging hooks to the MCP session
            ToolLoggingHooks(callback).attach(mcp.session)

            cfg = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                tools=[mcp.session],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                ),
                thinking_config=(
                    types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_budget=-1,  # Enable dynamic thinking
                    )
                    if hasattr(types, "ThinkingConfig")
                    else None
                ),
            )

            response = await chat.send_message(prompt, config=cfg)

            # Extract thinking content from response
            if hasattr(response, "candidates") and response.candidates:
                for candidate in response.candidates:
                    # Check for thinking content in candidate
                    if hasattr(candidate, "thought") and candidate.thought:
                        thinking_event: ThinkingEvent = {
                            "event": "thinking",
                            "ts": int(time.time() * 1000),
                            "summary": candidate.thought,
                            "model": model,
                        }
                        callback.emit(thinking_event)

            # Also check for thinking at the response level
            if hasattr(response, "thought") and response.thought:
                thinking_event: ThinkingEvent = {
                    "event": "thinking",
                    "ts": int(time.time() * 1000),
                    "summary": response.thought,
                    "model": model,
                }
                callback.emit(thinking_event)

            text = response.text
            callback.emit({"event": "finish", "result": text})
            return text
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
