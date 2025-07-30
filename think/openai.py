#!/usr/bin/env python3
"""OpenAI backed agent implementation.

This module provides :class:`AgentSession` which wraps the OpenAI Agents API
and exposes a simple ``run(prompt)`` coroutine. It is used by the unified
``think-agents`` CLI and can be imported directly for programmatic access.
"""

from __future__ import annotations

import logging
import sys
import time
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

from .agents import JSONEventCallback, ThinkingEvent
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


async def run_agent(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_event: Optional[Callable[[dict], None]] = None,
    persona: str = "default",
) -> str:
    """Run a single prompt through the OpenAI agent and return the response."""
    callback = JSONEventCallback(on_event)
    
    try:
        model_name = model
        if model.startswith("o4-mini"):
            parts = model.split("-")
            if len(parts) >= 2:
                model_name = "-".join(parts[:2])
        
        callback.emit({
            "event": "start",
            "prompt": prompt,
            "persona": persona,
            "model": model_name,
        })

        async with create_mcp_client() as mcp:
            system_instruction, extra_context, _ = agent_instructions(persona)
            agent = Agent(
                name="SunstoneCLI",
                instructions=f"{system_instruction}\n\n{extra_context}",
                model=model,
                model_settings=ModelSettings(max_tokens=max_tokens),
                mcp_servers=[mcp],
                hooks=ToolLoggingHooks(callback),
            )

            # Create temporary session for this run
            session = SQLiteSession("sunstone_oneshot")
            try:
                result = await Runner.run(agent, prompt, session=session, run_config=RunConfig())
            finally:
                if hasattr(session, "close"):
                    session.close()

        # Extract thinking summaries from reasoning items
        if hasattr(result, 'new_items') and result.new_items:
            for item in result.new_items:
                if hasattr(item, 'reasoning') and item.reasoning:
                    if hasattr(item.reasoning, 'summary') and item.reasoning.summary:
                        thinking_event: ThinkingEvent = {
                            "event": "thinking",
                            "ts": int(time.time() * 1000),
                            "summary": item.reasoning.summary,
                            "model": model
                        }
                        callback.emit(thinking_event)
                    elif hasattr(item.reasoning, 'content') and item.reasoning.content:
                        thinking_event: ThinkingEvent = {
                            "event": "thinking",
                            "ts": int(time.time() * 1000),
                            "summary": item.reasoning.content,
                            "model": model
                        }
                        callback.emit(thinking_event)

        # Alternative: Extract thinking summaries from raw responses
        elif hasattr(result, 'raw_responses') and result.raw_responses:
            for response in result.raw_responses:
                if hasattr(response, 'output') and response.output:
                    for output_item in response.output:
                        if hasattr(output_item, 'summary') and output_item.summary:
                            for summary_item in output_item.summary:
                                if hasattr(summary_item, 'text') and summary_item.text:
                                    thinking_event: ThinkingEvent = {
                                        "event": "thinking",
                                        "ts": int(time.time() * 1000),
                                        "summary": summary_item.text,
                                        "model": model
                                    }
                                    callback.emit(thinking_event)

        callback.emit({"event": "finish", "result": result.final_output})
        return result.final_output
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
