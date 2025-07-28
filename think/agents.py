from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypedDict, Union


class ToolStartEvent(TypedDict):
    """Event emitted when a tool starts."""

    event: Literal["tool_start"]
    tool: str
    args: Optional[dict[str, Any]]


class ToolEndEvent(TypedDict):
    """Event emitted when a tool finishes."""

    event: Literal["tool_end"]
    tool: str
    result: Any


class StartEvent(TypedDict):
    """Event emitted when an agent run begins."""

    event: Literal["start"]
    prompt: str


class FinishEvent(TypedDict):
    """Event emitted when an agent run finishes successfully."""

    event: Literal["finish"]
    result: str


class ErrorEvent(TypedDict):
    """Event emitted when an error occurs."""

    event: Literal["error"]
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


class JSONEventCallback:
    """Emit JSON events via a callback."""

    def __init__(self, callback: Optional[Callable[[Event], None]] = None) -> None:
        self.callback = callback

    def emit(self, data: Event) -> None:
        if self.callback:
            self.callback(data)


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
