#!/usr/bin/env python3
"""OpenAI backed agent implementation.

This module provides :class:`AgentSession` which wraps the OpenAI Agents API
and exposes a simple ``run(prompt)`` coroutine. It is used by the unified
``think-agents`` CLI and can be imported directly for programmatic access.
"""

from __future__ import annotations

import logging
import sys
import traceback
from typing import Callable, Optional

from agents import (
    Agent,
    AgentHooks,
    ModelSettings,
    RunConfig,
    Runner,
    SQLiteSession,
    enable_verbose_stdout_logging,
)

from think.utils import agent_instructions, create_mcp_client

from .agents import BaseAgentSession, JSONEventCallback
from .models import GPT_O4_MINI

DEFAULT_MODEL = GPT_O4_MINI
DEFAULT_MAX_TOKENS = 1024 * 32


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration for agent and tools."""

    if verbose:
        enable_verbose_stdout_logging()

        openai_agents_logger = logging.getLogger("openai.agents")
        openai_agents_logger.setLevel(logging.DEBUG)
        if not openai_agents_logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            openai_agents_logger.addHandler(handler)

        tracing_logger = logging.getLogger("openai.agents.tracing")
        tracing_logger.setLevel(logging.DEBUG)
        if not tracing_logger.handlers:
            tracing_logger.addHandler(handler)

    app_logger = logging.getLogger(__name__)
    if verbose:
        app_logger.setLevel(logging.DEBUG)
    else:
        app_logger.setLevel(logging.INFO)
    return app_logger


class ToolLoggingHooks(AgentHooks):
    """Hooks forwarding agent events to :class:`JSONEventCallback`."""

    def __init__(self, writer: JSONEventCallback) -> None:
        self.writer = writer

    def on_tool_call_start(self, context, tool_name: str, arguments: dict) -> None:
        self.writer.emit({"event": "tool_start", "tool": tool_name, "args": arguments})

    def on_tool_call_end(self, context, tool_name: str, result) -> None:

        self.writer.emit({"event": "tool_end", "tool": tool_name, "result": result})

    def on_agent_start(self, context, agent_name: str) -> None:
        self.writer.emit({"event": "agent_start", "agent": agent_name})

    def on_agent_end(self, context, agent_name: str, result) -> None:
        self.writer.emit({"event": "agent_end", "agent": agent_name})


class AgentSession(BaseAgentSession):
    """Context manager wrapping the Sunstone agent with optional session reuse."""

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
        self.session = SQLiteSession("sunstone_cli_session")
        self.run_config = RunConfig()
        self._history: list[dict[str, str]] = []
        self._pending_history: list[dict[str, str]] = []
        self.persona = persona

    async def __aenter__(self) -> "AgentSession":
        return self

    @property
    def history(self) -> list[dict[str, str]]:
        """Return chat history added and generated during this session."""

        return list(self._history)

    def add_history(self, role: str, text: str) -> None:
        """Queue a history message for the next run."""

        self._pending_history.append({"role": role, "content": text})
        self._history.append({"role": role, "content": text})

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if hasattr(self.session, "close"):
            self.session.close()

    async def run(self, prompt: str) -> str:
        """Run ``prompt`` through the agent and return the result."""
        try:
            if self._pending_history:
                await self.session.add_items(self._pending_history)
                self._pending_history.clear()

            self._history.append({"role": "user", "content": prompt})

            model_name = self.model
            if self.model.startswith("o4-mini"):
                parts = self.model.split("-")
                if len(parts) >= 2:
                    model_name = "-".join(parts[:2])
            self._callback.emit(
                {
                    "event": "start",
                    "prompt": prompt,
                    "persona": self.persona,
                    "model": model_name,
                }
            )

            async with create_mcp_client() as mcp:
                system_instruction, extra_context, _ = agent_instructions(self.persona)
                agent = Agent(
                    name="SunstoneCLI",
                    instructions=f"{system_instruction}\n\n{extra_context}",
                    model=self.model,
                    model_settings=ModelSettings(max_tokens=self.max_tokens),
                    mcp_servers=[mcp],
                    hooks=ToolLoggingHooks(self._callback),
                )

                result = await Runner.run(
                    agent, prompt, session=self.session, run_config=self.run_config
                )

            self._callback.emit({"event": "finish", "result": result.final_output})
            self._history.append({"role": "assistant", "content": result.final_output})
            return result.final_output
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
