# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Calendar event storage utilities shared across think/app components.

Calendar events are stored as JSONL files with one JSON object per line.
Line number (1-indexed) serves as the stable event ID since events are
never removed, only cancelled.
"""

from __future__ import annotations

import fcntl
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from think.utils import get_journal, now_ms

__all__ = [
    "CalendarEvent",
    "EventDay",
    "CalendarEventError",
    "CalendarEventEmptyTitleError",
    "calendar_file_path",
    "validate_line_number",
]

TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class CalendarEventError(Exception):
    """Base exception for calendar event operations."""


class CalendarEventEmptyTitleError(CalendarEventError):
    """Raised when attempting to create or update with an empty event title."""

    def __init__(self) -> None:
        super().__init__("event title cannot be empty")


@dataclass(slots=True)
class CalendarEvent:
    """Structured representation of a calendar event entry."""

    index: int
    title: str
    start: str
    end: str | None
    summary: str | None
    participants: list[str] | None
    cancelled: bool
    cancelled_reason: str | None = None
    moved_to: str | None = None
    created_at: int | None = None
    updated_at: int | None = None

    def as_dict(self) -> dict[str, object]:
        """Return the item as a JSON-serializable dictionary."""
        data: dict[str, object] = {
            "index": self.index,
            "title": self.title,
            "start": self.start,
            "end": self.end,
            "summary": self.summary,
            "participants": self.participants,
            "cancelled": self.cancelled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.cancelled_reason is not None:
            data["cancelled_reason"] = self.cancelled_reason
        if self.moved_to is not None:
            data["moved_to"] = self.moved_to
        return data

    def to_jsonl(self) -> dict[str, Any]:
        """Return the event as a sparse JSONL-compatible dictionary for storage."""
        data: dict[str, Any] = {"title": self.title, "start": self.start}
        if self.end is not None:
            data["end"] = self.end
        if self.summary is not None:
            data["summary"] = self.summary
        if self.participants is not None:
            data["participants"] = self.participants
        if self.cancelled:
            data["cancelled"] = True
        if self.cancelled_reason is not None:
            data["cancelled_reason"] = self.cancelled_reason
        if self.moved_to is not None:
            data["moved_to"] = self.moved_to
        if self.created_at is not None:
            data["created_at"] = self.created_at
        if self.updated_at is not None:
            data["updated_at"] = self.updated_at
        return data

    @classmethod
    def from_jsonl(cls, data: dict[str, Any], index: int) -> "CalendarEvent":
        """Create a CalendarEvent from a JSONL dictionary."""
        participants = data.get("participants")
        if not isinstance(participants, list):
            participants = None

        summary = data.get("summary")
        if summary is not None:
            summary = str(summary)

        end = data.get("end")
        if end is not None:
            end = str(end)

        return cls(
            index=index,
            title=str(data.get("title", "")),
            start=str(data.get("start", "")),
            end=end,
            summary=summary,
            participants=participants,
            cancelled=bool(data.get("cancelled", False)),
            cancelled_reason=data.get("cancelled_reason"),
            moved_to=data.get("moved_to"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    def display_line(self) -> str:
        """Return human-readable display format for this event."""
        if self.end:
            line = f"{self.start}-{self.end} {self.title}"
        else:
            line = f"{self.start} {self.title}"

        if self.cancelled:
            return f"~~{line}~~"

        return line


@dataclass(slots=True)
class EventDay:
    """In-memory representation of a day's calendar events for a facet."""

    day: str
    facet: str
    path: Path
    items: list[CalendarEvent]
    exists: bool

    def _validated_title(self, title: str) -> str:
        """Validate and clean event title."""
        cleaned = title.strip()
        if not cleaned:
            raise CalendarEventEmptyTitleError()
        return cleaned

    def _get_item(self, line_number: int) -> tuple[int, CalendarEvent]:
        """Get item by line number, returning (index, item)."""
        validate_line_number(line_number, len(self.items))
        index = line_number - 1
        return index, self.items[index]

    @classmethod
    def load(cls, day: str, facet: str) -> "EventDay":
        """Load event entries for ``day`` and ``facet``."""
        path = calendar_file_path(day, facet)
        exists = path.is_file()
        items: list[CalendarEvent] = []

        if exists:
            try:
                text = path.read_text(encoding="utf-8")
                item_index = 0
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    item_index += 1
                    try:
                        data = json.loads(line)
                        items.append(CalendarEvent.from_jsonl(data, item_index))
                    except json.JSONDecodeError:
                        logging.debug(
                            "Skipping malformed JSONL line %d in %s", item_index, path
                        )
                        continue
            except OSError as exc:
                logging.debug("Failed reading calendar events from %s: %s", path, exc)
                exists = False

        return cls(day=day, facet=facet, path=path, items=items, exists=exists)

    @classmethod
    def locked_modify(
        cls,
        day: str,
        facet: str,
        modify_fn: Any,
        max_retries: int = 3,
    ) -> Any:
        """Perform a locked load-modify-save on a day of calendar events."""
        path = calendar_file_path(day, facet)
        lock_path = path.parent / f"{path.name}.lock"

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(lock_path, "w") as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    try:
                        day_events = cls.load(day, facet)
                        return modify_fn(day_events)
                    finally:
                        fcntl.flock(lock_file, fcntl.LOCK_UN)
            except (IndexError, CalendarEventError, FileNotFoundError):
                raise
            except OSError as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(0.05, 0.3) * (attempt + 1))

        raise last_error  # type: ignore[misc]

    def save(self) -> None:
        """Persist the day back to disk, creating parent directories if needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for item in self.items:
            lines.append(json.dumps(item.to_jsonl(), ensure_ascii=False))

        content = "\n".join(lines)
        if lines:
            content += "\n"
        self.path.write_text(content, encoding="utf-8")
        self.exists = True

    def display(self) -> str:
        """Return event list formatted for display with line numbers."""
        if not self.items:
            return "0: (no events)"

        lines = [f"{item.index}: {item.display_line()}" for item in self.items]
        return "\n".join(lines)

    def append_event(
        self,
        title: str,
        start: str,
        end: str | None = None,
        summary: str | None = None,
        participants: list[str] | None = None,
        created_at: int | None = None,
    ) -> CalendarEvent:
        """Append a new event entry."""
        clean_title = self._validated_title(title)
        validate_time(start)
        if end is not None:
            validate_time(end)
            if end < start:
                raise ValueError("end time must be greater than or equal to start time")

        ts = created_at if created_at is not None else now_ms()
        item = CalendarEvent(
            index=len(self.items) + 1,
            title=clean_title,
            start=start,
            end=end,
            summary=summary,
            participants=participants,
            cancelled=False,
            created_at=ts,
            updated_at=ts,
        )

        self.items.append(item)
        self.save()
        return item

    def cancel_event(
        self,
        line_number: int,
        cancelled_reason: str | None = None,
        moved_to: str | None = None,
    ) -> CalendarEvent:
        """Cancel an event entry (soft delete)."""
        _, item = self._get_item(line_number)
        item.cancelled = True
        if cancelled_reason is not None:
            item.cancelled_reason = cancelled_reason
        if moved_to is not None:
            item.moved_to = moved_to
        item.updated_at = now_ms()
        self.save()
        return item

    def update_event(self, line_number: int, **kwargs: Any) -> CalendarEvent:
        """Update selected fields on an event entry."""
        _, item = self._get_item(line_number)

        new_title = kwargs.get("title", None)
        new_start = kwargs.get("start", None)
        new_end = kwargs.get("end", None)
        new_summary = kwargs.get("summary", None)
        new_participants = kwargs.get("participants", None)

        if new_title is not None:
            item.title = self._validated_title(new_title)

        effective_start = item.start
        effective_end = item.end

        if new_start is not None:
            validate_time(new_start)
            effective_start = new_start

        if new_end is not None:
            validate_time(new_end)
            effective_end = new_end

        if effective_end is not None and effective_end < effective_start:
            raise ValueError("end time must be greater than or equal to start time")

        if new_start is not None:
            item.start = new_start
        if new_end is not None:
            item.end = new_end
        if new_summary is not None:
            item.summary = new_summary
        if new_participants is not None:
            item.participants = new_participants

        item.updated_at = now_ms()
        self.save()
        return item


def calendar_file_path(day: str, facet: str) -> Path:
    """Return the absolute path to ``facets/{facet}/calendar/{day}.jsonl``."""
    return Path(get_journal()) / "facets" / facet / "calendar" / f"{day}.jsonl"


def validate_line_number(line_number: int, max_line: int) -> None:
    """Ensure ``line_number`` is within ``[1, max_line]`` inclusive."""
    if line_number < 1 or line_number > max_line:
        raise IndexError(f"line number {line_number} is out of range (1..{max_line})")


def validate_time(value: str) -> None:
    """Validate HH:MM time format."""
    if not TIME_RE.fullmatch(value):
        raise ValueError(f"invalid time format '{value}', expected HH:MM")
