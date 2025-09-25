"""Todo checklist utilities shared across think components."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv

from think.domains import get_domains

__all__ = [
    "TodoChecklist",
    "TodoItem",
    "TodoError",
    "TodoLineNumberError",
    "TodoGuardMismatchError",
    "TodoEmptyTextError",
    "TodoDomainError",
    "get_todos",
    "todo_file_path",
    "format_numbered",
    "parse_entry",
    "validate_line_number",
    "parse_item",
    "parse_items",
]

TODO_ENTRY_RE = re.compile(r"^- \[( |x|X)\]\s?(.*)$")
LEADING_MARKUP_RE = re.compile(r"\*\*([^*]+)\*\*:\s*(.*)")
DOMAIN_RE = re.compile(r"\s#([a-zA-Z][\w-]*)\b")
TIME_RE = re.compile(r"\((\d{1,2}:[0-5]\d)\)\s*$")


class TodoError(Exception):
    """Base exception for todo checklist operations."""


class TodoLineNumberError(TodoError):
    """Raised when an unexpected line number is supplied."""

    def __init__(self, expected: int, received: int) -> None:
        super().__init__(f"line number {received} must match the next available line")
        self.expected = expected
        self.received = received


class TodoGuardMismatchError(TodoError):
    """Raised when guard text does not match the current todo line."""

    def __init__(self, expected: str) -> None:
        super().__init__("guard text does not match current todo")
        self.expected = expected


class TodoEmptyTextError(TodoError):
    """Raised when attempting to add an empty todo entry."""

    def __init__(self) -> None:
        super().__init__("todo text cannot be empty")


class TodoDomainError(TodoError):
    """Raised when an invalid domain is specified in a todo entry."""

    def __init__(self, invalid_domain: str, valid_domains: list[str]) -> None:
        super().__init__(f"Unknown domain: {invalid_domain}")
        self.invalid_domain = invalid_domain
        self.valid_domains = valid_domains


@dataclass(slots=True)
class TodoChecklist:
    """In-memory representation of a day's todo checklist."""

    day: str
    path: Path
    entries: list[str]
    exists: bool

    def _validated_body(self, text: str) -> str:
        body = text.strip()
        if not body:
            raise TodoEmptyTextError()
        return body

    def _entry_components(
        self, line_number: int, guard: str
    ) -> tuple[int, str, bool, str]:
        validate_line_number(line_number, len(self.entries))

        index = line_number - 1
        entry = self.entries[index]
        completed, body = parse_entry(entry)
        if guard and guard != entry:
            raise TodoGuardMismatchError(entry)

        return index, entry, completed, body

    @classmethod
    def load(cls, day: str, *, ensure_day: bool = False) -> "TodoChecklist":
        """Load checklist entries for ``day``.

        Args:
            day: Journal day in ``YYYYMMDD`` format.
            ensure_day: When ``True`` the day directory is created if missing.

        Raises:
            FileNotFoundError: If the day directory does not exist and ``ensure_day``
                is ``False``.
        """

        path = todo_file_path(day)
        day_dir = path.parents[1]

        if ensure_day:
            journal = os.getenv("JOURNAL_PATH")
            if not journal:
                raise RuntimeError("JOURNAL_PATH not set")
            day_dir.mkdir(parents=True, exist_ok=True)
        elif not day_dir.is_dir():
            raise FileNotFoundError(f"day folder '{day}' not found")

        exists = path.is_file()
        if not exists:
            path.parent.mkdir(parents=True, exist_ok=True)
            entries: list[str] = []
        else:
            text = path.read_text(encoding="utf-8")
            entries = [line.rstrip("\n") for line in text.splitlines() if line.strip()]

        return cls(day=day, path=path, entries=entries, exists=exists)

    def save(self) -> None:
        """Persist the checklist back to disk."""

        content = "\n".join(self.entries)
        if self.entries:
            content += "\n"
        self.path.write_text(content, encoding="utf-8")

    def numbered(self) -> str:
        """Return checklist entries formatted with ``1:`` numbering."""

        return format_numbered(self.entries)

    def add_entry(self, line_number: int, text: str) -> None:
        """Append a new unchecked todo entry."""

        expected = len(self.entries) + 1
        if line_number != expected:
            raise TodoLineNumberError(expected, line_number)

        self.append_entry(text)

    def append_entry(self, text: str) -> None:
        """Append a new unchecked todo entry without line validation."""

        body = self._validated_body(text)
        entry_line = f"- [ ] {body}"

        # Validate that the entry can be parsed correctly
        item = parse_item(entry_line, len(self.entries) + 1)
        if item is None:
            raise TodoError(f"Failed to create valid todo entry from: {body}")

        # Validate domain if specified
        if item.domain:
            try:
                domains = get_domains()
                valid_domains = list(domains.keys())
                if item.domain not in valid_domains:
                    raise TodoDomainError(item.domain, valid_domains)
            except RuntimeError:  # JOURNAL_PATH not set
                pass  # Skip domain validation if journal path not available

        self.entries.append(entry_line)
        self.save()

    def remove_entry(self, line_number: int, guard: str) -> None:
        """Remove a todo entry after validating guard text."""

        index, _, _, _ = self._entry_components(line_number, guard)

        del self.entries[index]
        self.save()

    def mark_done(self, line_number: int, guard: str) -> None:
        """Mark a todo entry complete."""

        index, _, _, body = self._entry_components(line_number, guard)

        self.entries[index] = f"- [x] {body}"
        self.save()

    def mark_undone(self, line_number: int, guard: str) -> None:
        """Mark a todo entry incomplete."""

        index, _, _, body = self._entry_components(line_number, guard)

        self.entries[index] = f"- [ ] {body}"
        self.save()

    def update_entry_text(self, line_number: int, guard: str, text: str) -> None:
        """Replace the body text of an entry while keeping its completion state."""

        index, _, completed, _ = self._entry_components(line_number, guard)
        body = self._validated_body(text)

        checkbox = "[x]" if completed else "[ ]"
        self.entries[index] = f"- {checkbox} {body}"
        self.save()


@dataclass(slots=True)
class TodoItem:
    """Structured representation of a todo entry."""

    index: int
    raw: str
    text: str
    domain: str | None
    time: str | None
    completed: bool
    cancelled: bool

    def as_dict(self) -> dict[str, object]:
        """Return the item as a JSON-serializable dictionary."""

        return {
            "index": self.index,
            "raw": self.raw,
            "text": self.text,
            "domain": self.domain,
            "time": self.time,
            "completed": self.completed,
            "cancelled": self.cancelled,
        }


def todo_file_path(day: str) -> Path:
    """Return the absolute path to ``todos/today.md`` for ``day``."""

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH", "journal")
    return Path(journal) / day / "todos" / "today.md"


def format_numbered(entries: Sequence[str]) -> str:
    """Return ``entries`` formatted with ``1:`` style numbering."""

    if not entries:
        return "0: (no todos)"

    return "\n".join(f"{idx}: {line}" for idx, line in enumerate(entries, start=1))


def parse_entry(entry: str) -> tuple[bool, str]:
    """Parse a markdown todo entry and return completion flag and body text."""

    match = TODO_ENTRY_RE.match(entry)
    if not match:
        raise ValueError("entry is not a markdown checklist item")

    completed = match.group(1).lower() == "x"
    text = match.group(2)
    return completed, text


def validate_line_number(line_number: int, max_line: int) -> None:
    """Ensure ``line_number`` is within ``[1, max_line]`` inclusive."""

    if line_number < 1 or line_number > max_line:
        raise IndexError(f"line number {line_number} is out of range (1..{max_line})")


def parse_item(line: str, index: int) -> TodoItem | None:
    """Parse a single todo line into a structured :class:`TodoItem` object.

    Args:
        line: The raw todo line text to parse.
        index: The 1-based index for this item.

    Returns:
        A TodoItem object if the line is valid, None otherwise.
    """

    stripped = line.strip()
    if not stripped or not stripped.startswith("- ["):
        return None

    match_state = TODO_ENTRY_RE.match(stripped)
    if not match_state:
        return None

    completed = match_state.group(1).lower() == "x"
    remainder = match_state.group(2).strip()

    cancelled = False
    cleaned = remainder
    if cleaned.startswith("~~"):
        close_idx = cleaned.find("~~", 2)
        if close_idx > 0:
            cancelled = True
            inner = cleaned[2:close_idx].strip()
            tail = cleaned[close_idx + 2 :].strip()
            cleaned = inner + (f" {tail}" if tail else "")

    description = cleaned

    markup_match = LEADING_MARKUP_RE.match(cleaned)
    if markup_match:
        description = markup_match.group(2).strip()

    domain: str | None = None
    domain_match = DOMAIN_RE.search(description)
    if domain_match:
        domain = domain_match.group(1)
        before = description[: domain_match.start()].rstrip()
        after = description[domain_match.end() :].strip()
        description = (before + (f" {after}" if after else "")).strip()

    time: str | None = None
    time_match = TIME_RE.search(description)
    if time_match:
        hour_str = time_match.group(1).split(":", 1)[0]
        try:
            if 0 <= int(hour_str) <= 23:
                time = time_match.group(1)
                description = description[: time_match.start()].rstrip()
        except ValueError:
            pass

    return TodoItem(
        index=index,
        raw=stripped,
        text=description,
        domain=domain,
        time=time,
        completed=completed,
        cancelled=cancelled,
    )


def parse_items(entries: Iterable[str]) -> list[TodoItem]:
    """Parse todo ``entries`` into structured :class:`TodoItem` objects."""

    items: list[TodoItem] = []
    index = 0

    for raw in entries:
        item = parse_item(raw, index + 1)
        if item:
            index += 1
            item.index = index  # Update with actual sequential index
            items.append(item)

    return items


def get_todos(day: str, *, ensure_day: bool = True) -> list[dict[str, Any]] | None:
    """Load todos for ``day`` returning checklist metadata dictionaries.

    Args:
        day: Journal day in ``YYYYMMDD`` format.
        ensure_day: When ``True`` the day directory is created when missing.

    Returns:
        List of parsed todo entries or ``None`` when the checklist does not
        exist or cannot be read.
    """

    try:
        checklist = TodoChecklist.load(day, ensure_day=ensure_day)
    except OSError as exc:  # pragma: no cover - filesystem failure
        logging.debug("Failed reading todos for %s: %s", day, exc)
        return None

    if not checklist.exists:
        return None

    return [item.as_dict() for item in parse_items(checklist.entries)]
