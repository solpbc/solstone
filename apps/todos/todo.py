"""Todo checklist utilities shared across think components."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv

from think.facets import get_facets

__all__ = [
    "TodoChecklist",
    "TodoItem",
    "TodoError",
    "TodoLineNumberError",
    "TodoGuardMismatchError",
    "TodoEmptyTextError",
    "get_todos",
    "todo_file_path",
    "format_numbered",
    "parse_entry",
    "validate_line_number",
    "parse_item",
    "parse_items",
    "upcoming",
    "get_facets_with_todos",
]

TODO_ENTRY_RE = re.compile(r"^- \[( |x|X)\]\s?(.*)$")
LEADING_MARKUP_RE = re.compile(r"\*\*([^*]+)\*\*:\s*(.*)")
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


@dataclass(slots=True)
class TodoChecklist:
    """In-memory representation of a day's todo checklist."""

    day: str
    facet: str
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
    def load(cls, day: str, facet: str) -> "TodoChecklist":
        """Load checklist entries for ``day`` and ``facet``.

        Args:
            day: Journal day in ``YYYYMMDD`` format.
            facet: Facet name (e.g., "personal", "work").

        Returns:
            TodoChecklist instance with entries loaded from disk, or empty if file doesn't exist.
        """

        path = todo_file_path(day, facet)

        exists = path.is_file()
        if not exists:
            entries: list[str] = []
        else:
            text = path.read_text(encoding="utf-8")
            entries = [line.rstrip("\n") for line in text.splitlines() if line.strip()]

        return cls(day=day, facet=facet, path=path, entries=entries, exists=exists)

    def save(self) -> None:
        """Persist the checklist back to disk, creating parent directories if needed."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
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
    time: str | None
    completed: bool
    cancelled: bool

    def as_dict(self) -> dict[str, object]:
        """Return the item as a JSON-serializable dictionary."""

        return {
            "index": self.index,
            "raw": self.raw,
            "text": self.text,
            "time": self.time,
            "completed": self.completed,
            "cancelled": self.cancelled,
        }


def todo_file_path(day: str, facet: str) -> Path:
    """Return the absolute path to ``facets/{facet}/todos/{day}.md``.

    Args:
        day: Journal day in ``YYYYMMDD`` format.
        facet: Facet name (e.g., "personal", "work").

    Returns:
        Path to the facet-scoped todo file for the specified day.
    """

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH", "journal")
    return Path(journal) / "facets" / facet / "todos" / f"{day}.md"


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


def get_todos(day: str, facet: str) -> list[dict[str, Any]] | None:
    """Load todos for ``day`` and ``facet`` returning checklist metadata dictionaries.

    Args:
        day: Journal day in ``YYYYMMDD`` format.
        facet: Facet name (e.g., "personal", "work").

    Returns:
        List of parsed todo entries or ``None`` when the checklist does not
        exist or cannot be read.
    """

    try:
        checklist = TodoChecklist.load(day, facet)
    except OSError as exc:  # pragma: no cover - filesystem failure
        logging.debug("Failed reading todos for %s/%s: %s", facet, day, exc)
        return None

    if not checklist.exists:
        return None

    return [item.as_dict() for item in parse_items(checklist.entries)]


def upcoming(
    limit: int = 20, facet: str | None = None, *, today: str | None = None
) -> str:
    """Return a markdown summary of upcoming todo entries.

    Args:
        limit: Maximum number of todo items to include.
        facet: Optional facet filter. When ``None`` aggregates todos from all facets.
        today: Optional ``YYYYMMDD`` override for the current day, useful for testing.

    Returns:
        Markdown with sections per facet and day. Format is ``### {Facet Title}: YYYYMMDD``
        followed by raw todo checklist lines. When no upcoming items exist the string
        ``"No upcoming todos."`` is returned.
    """

    if limit <= 0:
        return "No upcoming todos."

    journal = os.getenv("JOURNAL_PATH", "journal")
    root = Path(journal)
    if not root.is_dir():
        return "No upcoming todos."

    today_str = today if today is not None else datetime.now().strftime("%Y%m%d")

    # Determine which facets to scan
    facets_dir = root / "facets"
    if not facets_dir.is_dir():
        return "No upcoming todos."

    try:
        if facet is not None:
            # Single facet mode
            facet_paths = [(facet, facets_dir / facet)]
        else:
            # All facets mode
            facet_paths = [(d.name, d) for d in facets_dir.iterdir() if d.is_dir()]
    except OSError:  # pragma: no cover - filesystem failure
        return "No upcoming todos."

    if not facet_paths:
        return "No upcoming todos."

    # Collect all todos across facets
    all_todos: list[tuple[str, str, list[str]]] = []  # (facet, day, lines)

    for facet_name, facet_path in facet_paths:
        todos_dir = facet_path / "todos"
        if not todos_dir.is_dir():
            continue

        try:
            todo_files = sorted(
                f.stem
                for f in todos_dir.iterdir()
                if f.is_file()
                and f.suffix == ".md"
                and len(f.stem) == 8
                and f.stem.isdigit()
                and f.stem > today_str
            )
        except OSError:  # pragma: no cover - filesystem failure
            continue

        for day in todo_files:
            todo_path = todos_dir / f"{day}.md"
            try:
                lines = [
                    line.rstrip("\n")
                    for line in todo_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except OSError:  # pragma: no cover - filesystem failure
                continue

            items = parse_items(lines)
            if items:
                all_todos.append((facet_name, day, [item.raw for item in items]))

    if not all_todos:
        return "No upcoming todos."

    # Sort by day, then facet
    all_todos.sort(key=lambda x: (x[1], x[0]))

    # Build output sections
    remaining = limit
    sections: list[str] = []

    for facet_name, day, lines in all_todos:
        # Get facet title for better display
        try:
            facets = get_facets()
            facet_title = facets.get(facet_name, {}).get("title", facet_name.title())
        except (RuntimeError, KeyError):
            facet_title = facet_name.title()

        day_lines: list[str] = []
        for line in lines:
            day_lines.append(line)
            remaining -= 1
            if remaining == 0:
                break

        if day_lines:
            section = "\n".join([f"### {facet_title}: {day}"] + day_lines)
            sections.append(section)

        if remaining == 0:
            break

    if not sections:
        return "No upcoming todos."

    return "\n\n".join(sections)


def get_facets_with_todos(day: str) -> list[str]:
    """Return a list of facet names that have todos for the given day.

    Args:
        day: Journal day in ``YYYYMMDD`` format.

    Returns:
        List of facet names that have todo files for the specified day.
        Returns empty list if no facets have todos or if journal path is invalid.
    """

    journal = os.getenv("JOURNAL_PATH", "journal")
    root = Path(journal)
    facets_dir = root / "facets"

    if not facets_dir.is_dir():
        return []

    facets_with_todos: list[str] = []

    try:
        for facet_dir in facets_dir.iterdir():
            if not facet_dir.is_dir():
                continue

            todo_path = facet_dir / "todos" / f"{day}.md"
            if todo_path.is_file():
                facets_with_todos.append(facet_dir.name)
    except OSError:  # pragma: no cover - filesystem failure
        return []

    return sorted(facets_with_todos)
