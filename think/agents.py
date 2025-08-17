from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypedDict, Union

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
    model: str = "",
    max_tokens: int = 0,
    on_event: Optional[Callable[[Event], None]] = None,
    persona: str = "default",
) -> str:
    """Run a single prompt through an agent backend and return the response.

    Args:
        prompt: The prompt to run
        backend: Backend provider ("openai", "google", "anthropic")
        model: Model to use (defaults to backend default)
        max_tokens: Maximum tokens (defaults to backend default)
        on_event: Optional event callback
        persona: Persona instructions to load

    Returns:
        The agent's response text
    """
    if backend == "google":
        from . import google as backend_mod
    elif backend == "anthropic":
        from . import anthropic as backend_mod
    else:
        from . import openai as backend_mod

    return await backend_mod.run_agent(
        prompt,
        model=model or backend_mod.DEFAULT_MODEL,
        max_tokens=max_tokens or backend_mod.DEFAULT_MAX_TOKENS,
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
    """Unified CLI for all agent backends."""

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--backend",
        choices=["openai", "google", "anthropic"],
        default="openai",
        help="Backend provider",
    )
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.backend == "google":
        from . import google as backend_mod
    elif pre_args.backend == "anthropic":
        from . import anthropic as backend_mod
    else:
        from . import openai as backend_mod

    parser = argparse.ArgumentParser(description="Sunstone Agent CLI")
    parser.add_argument(
        "task_file",
        nargs="?",
        help="Path to .txt file with the request, '-' for stdin or omit for interactive",
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "google", "anthropic"],
        default=pre_args.backend,
        help="Backend provider",
    )
    parser.add_argument(
        "--model",
        default=getattr(backend_mod, "DEFAULT_MODEL"),
        help="Model to use",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=getattr(backend_mod, "DEFAULT_MAX_TOKENS"),
        help="Maximum tokens for the final response",
    )
    parser.add_argument(
        "-q", "--query", help="Direct prompt text for single query mode"
    )
    parser.add_argument(
        "-o", "--out", help="File path to write the final result or error to"
    )
    parser.add_argument(
        "-p", "--persona", default="default", help="Persona instructions to load"
    )

    args = setup_cli(parser)
    out_path = args.out

    app_logger = backend_mod.setup_logging(args.verbose)
    event_writer = JSONEventWriter(out_path)

    def emit_event(data: Event) -> None:
        if "ts" not in data:
            data["ts"] = int(time.time() * 1000)
        event_writer.emit(data)

    if args.backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            from agents import set_default_openai_key

            set_default_openai_key(api_key)

    if args.query:
        user_prompt = args.query
    elif args.task_file is None:
        user_prompt = None
    elif args.task_file == "-":
        user_prompt = sys.stdin.read()
    else:
        if not os.path.isfile(args.task_file):
            parser.error(f"Task file not found: {args.task_file}")
        app_logger.info("Loading task file %s", args.task_file)
        user_prompt = Path(args.task_file).read_text(encoding="utf-8")

    try:
        if user_prompt is None:
            app_logger.info("Starting interactive mode with model %s", args.model)
            try:
                while True:
                    try:
                        loop = asyncio.get_event_loop()
                        prompt = await loop.run_in_executor(
                            None, lambda: input("chat> ")
                        )
                        if not prompt:
                            continue
                        await run_agent(
                            prompt,
                            backend=args.backend,
                            model=args.model,
                            max_tokens=args.max_tokens,
                            on_event=emit_event,
                            persona=args.persona,
                        )
                    except EOFError:
                        break
            except KeyboardInterrupt:
                pass
        else:
            app_logger.debug("Task contents: %s", user_prompt)
            app_logger.info("Running agent with model %s", args.model)
            await run_agent(
                user_prompt,
                backend=args.backend,
                model=args.model,
                max_tokens=args.max_tokens,
                on_event=emit_event,
                persona=args.persona,
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
