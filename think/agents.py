from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, TypedDict, Union

from think.utils import setup_cli


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


class JournalEventWriter(JSONEventWriter):
    """Deprecated - journal logging now handled by cortex."""

    def __init__(self) -> None:
        # Don't create journal files - cortex handles journal logging
        # Only stdout output is used
        super().__init__(path=None)

    def emit(self, data: Event) -> None:
        # Just emit to stdout, cortex will handle journal logging
        super().emit(data)


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


async def run_agent(
    prompt: str,
    *,
    backend: str = "openai",
    config: Optional[Dict[str, Any]] = None,
    on_event: Optional[Callable[[Event], None]] = None,
    persona: str = "default",
) -> str:
    """Run a single prompt through an agent backend and return the response.

    Args:
        prompt: The prompt to run
        backend: Backend provider ("openai", "google", "anthropic", "claude")
        config: Configuration dictionary for the backend (supports 'model', 'max_tokens', and backend-specific options)
        on_event: Optional event callback
        persona: Persona instructions to load

    Returns:
        The agent's response text
    """
    if backend == "google":
        from . import google as backend_mod
    elif backend == "anthropic":
        from . import anthropic as backend_mod
    elif backend == "claude":
        from . import claude as backend_mod
    else:
        from . import openai as backend_mod

    return await backend_mod.run_agent(
        prompt,
        config=config or {},
        on_event=on_event,
        persona=persona,
    )


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
    "run_agent",
]


async def main_async() -> None:
    """NDJSON-based CLI for agent backends."""

    parser = argparse.ArgumentParser(
        description="Sunstone Agent CLI - Accepts NDJSON input via stdin"
    )

    args = setup_cli(parser)

    # Import openai for logging setup
    from . import openai as openai_mod

    app_logger = openai_mod.setup_logging(args.verbose)

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
                # Parse NDJSON line
                request = json.loads(line)

                # Extract parameters
                prompt = request.get("prompt")
                if not prompt:
                    emit_event(
                        {
                            "event": "error",
                            "error": "Missing 'prompt' field in NDJSON input",
                            "ts": int(time.time() * 1000),
                        }
                    )
                    continue

                # Run agent with provided parameters
                backend = request.get("backend", "openai")
                persona = request.get("persona", "default")

                # Get config from request - expect it to be a dict
                config = request.get("config", {})

                # Set OpenAI key if needed
                if backend == "openai":
                    api_key = os.getenv("OPENAI_API_KEY", "")
                    if api_key:
                        from agents import set_default_openai_key

                        set_default_openai_key(api_key)

                app_logger.debug(
                    f"Processing request: backend={backend}, config={config}"
                )

                await run_agent(
                    prompt,
                    backend=backend,
                    config=config,
                    on_event=emit_event,
                    persona=persona,
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
