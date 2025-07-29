from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypedDict, Union

from think.utils import setup_cli


class ToolStartEvent(TypedDict):
    """Event emitted when a tool starts."""

    event: Literal["tool_start"]
    ts: int
    tool: str
    args: Optional[dict[str, Any]]


class ToolEndEvent(TypedDict):
    """Event emitted when a tool finishes."""

    event: Literal["tool_end"]
    ts: int
    tool: str
    result: Any


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


class ErrorEvent(TypedDict):
    """Event emitted when an error occurs."""

    event: Literal["error"]
    ts: int
    error: str


Event = Union[ToolStartEvent, ToolEndEvent, StartEvent, FinishEvent, ErrorEvent]


class JSONEventWriter:
    """Write JSONL events to stdout and an optional file."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.file = None
        if path:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                self.file = open(path, "a", encoding="utf-8")
            except Exception as exc:  # pragma: no cover - display only
                logging.error("Failed to open %s: %s", path, exc)

    def emit(self, data: Event) -> None:
        line = json.dumps(data, ensure_ascii=False)
        print(line)
        if self.file:
            try:
                self.file.write(line + "\n")
                self.file.flush()
            except Exception as exc:  # pragma: no cover - display only
                logging.error("Failed to write event to %s: %s", self.path, exc)

    def close(self) -> None:
        if self.file:
            try:
                self.file.close()
            except Exception:
                pass


class JournalEventWriter(JSONEventWriter):
    """Write JSONL events to ``<journal>/agents/<epoch>.jsonl``."""

    def __init__(self) -> None:
        journal = os.getenv("JOURNAL_PATH")
        path = None
        if journal:
            try:
                ts = int(time.time() * 1000)
                base = Path(journal) / "agents"
                base.mkdir(parents=True, exist_ok=True)
                path = str(base / f"{ts}.jsonl")
            except Exception as exc:  # pragma: no cover - optional
                logging.error("Failed to init journal log: %s", exc)
        super().__init__(path)

    def emit(self, data: Event) -> None:
        if self.file:
            try:
                self.file.write(json.dumps(data, ensure_ascii=False) + "\n")
                self.file.flush()
            except Exception as exc:  # pragma: no cover - display only
                logging.error("Failed to write journal event to %s: %s", self.path, exc)


_global_journal_writer: Optional[JournalEventWriter] | None = None


def _journal_emit(event: Event) -> None:
    """Write ``event`` to the journal log, creating it lazily."""
    global _global_journal_writer
    event_type = event.get("event")
    if event_type == "start":
        if _global_journal_writer is not None:
            _global_journal_writer.close()
        _global_journal_writer = JournalEventWriter()
    elif _global_journal_writer is None:
        _global_journal_writer = JournalEventWriter()

    if _global_journal_writer:
        _global_journal_writer.emit(event)
        if event_type in {"finish", "error"}:
            _global_journal_writer.close()
            _global_journal_writer = None


def _close_journal_writer() -> None:
    """Close the global journal log if open."""
    global _global_journal_writer
    if _global_journal_writer is not None:
        _global_journal_writer.close()
        _global_journal_writer = None


class JSONEventCallback:
    """Emit JSON events via a callback."""

    def __init__(self, callback: Optional[Callable[[Event], None]] = None) -> None:
        self.callback = callback

    def emit(self, data: Event) -> None:
        if "ts" not in data:
            data = {**data, "ts": int(time.time() * 1000)}
        if self.callback:
            self.callback(data)
        _journal_emit(data)

    def close(self) -> None:
        _close_journal_writer()


class BaseAgentSession(ABC):
    """Abstract base class for LLM agent sessions."""

    @abstractmethod
    async def __aenter__(self) -> "BaseAgentSession":
        """Enter the session context."""

    @abstractmethod
    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Exit the session context."""

    @property
    @abstractmethod
    def history(self) -> list[dict[str, str]]:
        """Return the chat history as ``role``/``content`` pairs."""

    @abstractmethod
    def add_history(self, role: str, text: str) -> None:
        """Queue a prior message for the next run."""

    @abstractmethod
    async def run(self, prompt: str) -> str:
        """Run ``prompt`` through the agent and return the response text."""


__all__ = [
    "ToolStartEvent",
    "ToolEndEvent",
    "StartEvent",
    "FinishEvent",
    "ErrorEvent",
    "Event",
    "JSONEventWriter",
    "JournalEventWriter",
    "JSONEventCallback",
    "BaseAgentSession",
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
        async with backend_mod.AgentSession(
            model=args.model,
            max_tokens=args.max_tokens,
            on_event=emit_event,
            persona=args.persona,
        ) as agent_session:
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
                            await agent_session.run(prompt)
                        except EOFError:
                            break
                except KeyboardInterrupt:
                    pass
            else:
                app_logger.debug("Task contents: %s", user_prompt)
                app_logger.info("Running agent with model %s", args.model)
                await agent_session.run(user_prompt)
    except Exception as exc:  # pragma: no cover - unexpected
        err = {"event": "error", "error": str(exc)}
        emit_event(err)
        _journal_emit(err)
        raise
    finally:
        event_writer.close()
        _close_journal_writer()


def main() -> None:
    """Entry point wrapper."""

    asyncio.run(main_async())
