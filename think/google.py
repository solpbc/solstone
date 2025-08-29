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
from typing import Any, Callable, Dict, Optional

from google import genai
from google.genai import types

from .agents import JSONEventCallback, ThinkingEvent
from .models import GEMINI_FLASH
from .utils import agent_instructions, create_mcp_client

# Default values are now handled internally
_DEFAULT_MODEL = GEMINI_FLASH
_DEFAULT_MAX_TOKENS = 8192


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
        self._counter = 0

    def attach(self, session: Any) -> None:
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
            result = await original(name=name, arguments=arguments, **kwargs)
            self.writer.emit(
                {
                    "event": "tool_end",
                    "tool": name,
                    "args": arguments,
                    "result": result,
                    "call_id": call_id,
                }
            )
            return result

        session.call_tool = wrapped  # type: ignore[assignment]


async def run_agent(
    prompt: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    persona: str = "default",
) -> str:
    """Run a single prompt through the Google Gemini agent and return the response.

    Args:
        prompt: The prompt to run
        config: Configuration dictionary (supports 'model', 'max_tokens', and backend-specific options)
        on_event: Optional event callback
        persona: Persona instructions to load
    """
    # Extract config values with defaults
    config = config or {}
    model = config.get("model", _DEFAULT_MODEL)
    max_tokens = config.get("max_tokens", _DEFAULT_MAX_TOKENS)
    disable_mcp = config.get("disable_mcp", False)

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
                "backend": "google",
            }
        )

        # Get system instruction (always needed)
        system_instruction, first_user, _ = agent_instructions(persona)

        # Build minimal history for chat
        history = []
        if first_user:
            history.append(
                types.Content(role="user", parts=[types.Part(text=first_user)])
            )

        # Create client
        client = genai.Client(api_key=api_key)

        # Create fresh chat session
        chat = client.aio.chats.create(
            model=model,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
            history=history,
        )

        # Configure tools based on disable_mcp flag
        if not disable_mcp:
            # Create MCP client and attach hooks
            async with create_mcp_client() as mcp:
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
        else:
            # No MCP tools - just basic config
            cfg = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
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

        # Extract thinking content from response (works for both MCP and non-MCP cases)
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
