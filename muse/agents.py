# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypedDict, Union

from think.utils import setup_cli

LOG = logging.getLogger("muse.agents")


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for agent CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, stream=sys.stdout)
    return LOG


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
    persona: str
    model: str
    backend: str


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


class ThinkingEvent(TypedDict):
    """Event emitted when thinking/reasoning summaries are available."""

    event: Literal["thinking"]
    ts: int
    summary: str
    model: Optional[str]


Event = Union[
    ToolStartEvent,
    ToolEndEvent,
    StartEvent,
    FinishEvent,
    ErrorEvent,
    ThinkingEvent,
    AgentUpdatedEvent,
]


class JSONEventWriter:
    """Write JSONL events to stdout and optionally to a file."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.file = None
        if path:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                self.file = open(path, "a", encoding="utf-8")
            except Exception:
                pass  # Fail silently if can't open file

    def emit(self, data: Event) -> None:
        line = json.dumps(data, ensure_ascii=False)
        print(line)
        sys.stdout.flush()  # Ensure immediate output for cortex
        if self.file:
            try:
                self.file.write(line + "\n")
                self.file.flush()
            except Exception:
                pass  # Fail silently on write errors

    def close(self) -> None:
        if self.file:
            try:
                self.file.close()
            except Exception:
                pass


def _journal_emit(event: Event) -> None:
    """Emit event to stdout for cortex to capture."""
    # No longer manages files - just ensure event goes to stdout
    pass


def _close_journal_writer() -> None:
    """No-op - cortex manages all file handles."""
    pass


class JSONEventCallback:
    """Emit JSON events via a callback."""

    def __init__(self, callback: Optional[Callable[[Event], None]] = None) -> None:
        self.callback = callback

    def emit(self, data: Event) -> None:
        if "ts" not in data:
            data = {**data, "ts": int(time.time() * 1000)}
        if self.callback:
            self.callback(data)

    def close(self) -> None:
        pass


def format_tool_summary(tool_calls: list) -> str:
    """Format tool calls into a readable summary string.

    Args:
        tool_calls: List of dicts with 'name' and 'args' keys

    Returns:
        Formatted string like "Tools used: fetch_data(url='...'), process(id=123)"
        or empty string if no tool calls
    """
    if not tool_calls:
        return ""

    tool_summaries = []
    for tool in tool_calls:
        name = tool.get("name", "unknown")
        args = tool.get("args", {})
        # Format args as compact string
        args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in args.items())
        tool_summaries.append(f"{name}({args_str})")

    return "\n\nTools used: " + ", ".join(tool_summaries)


def parse_agent_events_to_turns(conversation_id: str) -> list:
    """Parse agent event log into conversation turns.

    Converts agent event logs into simple conversation turns with role and content.
    Automatically combines assistant text with tool usage summaries.

    Args:
        conversation_id: Agent ID whose conversation to load

    Returns:
        List of dicts like [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        Returns empty list if conversation not found

    Note:
        Incomplete turns (missing finish event) are skipped
    """
    import logging

    from muse.cortex_client import read_agent_events

    logger = logging.getLogger(__name__)

    try:
        events = read_agent_events(conversation_id)
    except FileNotFoundError:
        logger.warning(f"Cannot continue from {conversation_id}: log not found")
        return []

    turns = []
    current_tool_calls: list = []

    for event in events:
        event_type = event.get("event")

        if event_type == "start":
            # User's initial prompt
            prompt = event.get("prompt", "")
            if prompt:
                turns.append({"role": "user", "content": prompt})

        elif event_type == "tool_start":
            # Track tool calls for current assistant turn
            tool_name = event.get("tool", "")
            tool_args = event.get("args", {})
            current_tool_calls.append({"name": tool_name, "args": tool_args})

        elif event_type == "finish":
            # Assistant's response with optional tool summary
            result_text = event.get("result", "").strip()

            # Build content combining response and tool usage
            content_parts = []
            if result_text:
                content_parts.append(result_text)

            if current_tool_calls:
                tool_summary = format_tool_summary(current_tool_calls)
                content_parts.append(tool_summary)

            if content_parts:
                turns.append({"role": "assistant", "content": "\n".join(content_parts)})

            # Reset tool tracking for next turn
            current_tool_calls = []

    return turns


__all__ = [
    "ToolStartEvent",
    "ToolEndEvent",
    "StartEvent",
    "FinishEvent",
    "ErrorEvent",
    "AgentUpdatedEvent",
    "ThinkingEvent",
    "Event",
    "JSONEventWriter",
    "JournalEventWriter",
    "JSONEventCallback",
    "format_tool_summary",
    "parse_agent_events_to_turns",
]


async def main_async() -> None:
    """NDJSON-based CLI for agent backends."""

    parser = argparse.ArgumentParser(
        description="solstone Agent CLI - Accepts NDJSON input via stdin"
    )

    args = setup_cli(parser)

    app_logger = setup_logging(args.verbose)

    # Always write to stdout only
    event_writer = JSONEventWriter(None)

    def emit_event(data: Event) -> None:
        if "ts" not in data:
            data["ts"] = int(time.time() * 1000)
        event_writer.emit(data)

    try:
        # NDJSON input mode from stdin only
        app_logger.info("Processing NDJSON input from stdin")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                # Parse NDJSON line - this is the complete merged config from Cortex
                config = json.loads(line)

                # Validate prompt exists
                prompt = config.get("prompt")
                if not prompt:
                    emit_event(
                        {
                            "event": "error",
                            "error": "Missing 'prompt' field in NDJSON input",
                            "ts": int(time.time() * 1000),
                        }
                    )
                    continue

                # Extract backend to route to correct module
                backend = config.get("backend", "google")

                # Set OpenAI key if needed
                if backend == "openai":
                    api_key = os.getenv("OPENAI_API_KEY", "")
                    if api_key:
                        from agents import set_default_openai_key

                        set_default_openai_key(api_key)

                app_logger.debug(f"Processing request: backend={backend}")

                # Route to appropriate backend
                if backend == "google":
                    from . import google as backend_mod
                elif backend == "anthropic":
                    from . import anthropic as backend_mod
                elif backend == "claude":
                    from . import claude as backend_mod
                else:
                    from . import openai as backend_mod

                # Pass complete config to backend
                await backend_mod.run_agent(
                    config=config,
                    on_event=emit_event,
                )

            except json.JSONDecodeError as e:
                emit_event(
                    {
                        "event": "error",
                        "error": f"Invalid JSON: {str(e)}",
                        "ts": int(time.time() * 1000),
                    }
                )
            except Exception as e:
                emit_event(
                    {
                        "event": "error",
                        "error": str(e),
                        "trace": traceback.format_exc(),
                        "ts": int(time.time() * 1000),
                    }
                )

    except Exception as exc:  # pragma: no cover - unexpected
        err = {
            "event": "error",
            "error": str(exc),
            "trace": traceback.format_exc(),
        }
        if not getattr(exc, "_evented", False):
            emit_event(err)
        raise
    finally:
        event_writer.close()


def main() -> None:
    """Entry point wrapper."""

    asyncio.run(main_async())
