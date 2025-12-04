"""Todo checklist utilities shared across think components.

Todos are stored as JSONL files with one JSON object per line. Line number (1-indexed)
serves as the stable todo ID since todos are never removed, only cancelled.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from think.facets import get_facets

__all__ = [
    "TodoChecklist",
    "TodoItem",
    "TodoError",
    "TodoLineNumberError",
    "TodoEmptyTextError",
    "get_todos",
    "todo_file_path",
    "validate_line_number",
    "upcoming",
    "get_facets_with_todos",
]

# Regex for extracting time annotation from text
TIME_RE = re.compile(r"\((\d{1,2}:[0-5]\d)\)\s*$")


def _now_ms() -> int:
    """Return current time as epoch milliseconds."""
    return int(time.time() * 1000)


class TodoError(Exception):
    """Base exception for todo checklist operations."""


class TodoLineNumberError(TodoError):
    """Raised when an unexpected line number is supplied."""

    def __init__(self, expected: int, received: int) -> None:
        super().__init__(f"line number {received} must match the next available line")
        self.expected = expected
        self.received = received


class TodoEmptyTextError(TodoError):
    """Raised when attempting to add an empty todo entry."""

    def __init__(self) -> None:
        super().__init__("todo text cannot be empty")


@dataclass(slots=True)
class TodoItem:
    """Structured representation of a todo entry."""

    index: int
    text: str
    time: str | None
    completed: bool
    cancelled: bool
    created_at: int | None = None
    updated_at: int | None = None

    def as_dict(self) -> dict[str, object]:
        """Return the item as a JSON-serializable dictionary."""
        return {
            "index": self.index,
            "text": self.text,
            "time": self.time,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_jsonl(self) -> dict[str, Any]:
        """Return the item as a JSONL-compatible dictionary for storage."""
        data: dict[str, Any] = {"text": self.text}
        if self.time:
            data["time"] = self.time
        if self.completed:
            data["completed"] = True
        if self.cancelled:
            data["cancelled"] = True
        if self.created_at is not None:
            data["created_at"] = self.created_at
        if self.updated_at is not None:
            data["updated_at"] = self.updated_at
        return data

    @classmethod
    def from_jsonl(cls, data: dict[str, Any], index: int) -> "TodoItem":
        """Create a TodoItem from a JSONL dictionary."""
        return cls(
            index=index,
            text=data.get("text", ""),
            time=data.get("time"),
            completed=data.get("completed", False),
            cancelled=data.get("cancelled", False),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    def display_line(self) -> str:
        """Return human-readable display format for this todo."""
        if self.cancelled:
            checkbox = "[cancelled]"
        elif self.completed:
            checkbox = "[x]"
        else:
            checkbox = "[ ]"

        text = self.text
        if self.time:
            text = f"{text} ({self.time})"

        if self.cancelled:
            return f"~~{checkbox} {text}~~"

        return f"{checkbox} {text}"


@dataclass(slots=True)
class TodoChecklist:
    """In-memory representation of a day's todo checklist."""

    day: str
    facet: str
    path: Path
    items: list[TodoItem]
    exists: bool

    def _validated_text(self, text: str) -> str:
        """Validate and clean todo text."""
        body = text.strip()
        if not body:
            raise TodoEmptyTextError()
        return body

    def _get_item(self, line_number: int) -> tuple[int, TodoItem]:
        """Get item by line number, returning (index, item)."""
        validate_line_number(line_number, len(self.items))
        index = line_number - 1
        return index, self.items[index]

    @classmethod
    def load(cls, day: str, facet: str) -> "TodoChecklist":
        """Load checklist entries for ``day`` and ``facet``.

        Args:
            day: Journal day in ``YYYYMMDD`` format.
            facet: Facet name (e.g., "personal", "work").

        Returns:
            TodoChecklist instance with items loaded from disk, or empty if file doesn't exist.
        """
        path = todo_file_path(day, facet)
        exists = path.is_file()
        items: list[TodoItem] = []

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
                        items.append(TodoItem.from_jsonl(data, item_index))
                    except json.JSONDecodeError:
                        logging.debug(
                            "Skipping malformed JSONL line %d in %s", item_index, path
                        )
                        continue
            except OSError as exc:
                logging.debug("Failed reading todos from %s: %s", path, exc)
                exists = False

        return cls(day=day, facet=facet, path=path, items=items, exists=exists)

    def save(self) -> None:
        """Persist the checklist back to disk, creating parent directories if needed."""
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
        """Return checklist formatted for display with line numbers.

        All items are included. Cancelled items are shown with ~~strikethrough~~.
        """
        if not self.items:
            return "0: (no todos)"

        lines = [f"{item.index}: {item.display_line()}" for item in self.items]
        return "\n".join(lines)

    def add_entry(
        self, line_number: int, text: str, time: str | None = None
    ) -> TodoItem:
        """Append a new unchecked todo entry.

        Args:
            line_number: Expected next line value; must be ``current_count + 1``.
            text: Body of the todo item.
            time: Optional scheduled time in "HH:MM" format.

        Returns:
            The newly created TodoItem.
        """
        expected = len(self.items) + 1
        if line_number != expected:
            raise TodoLineNumberError(expected, line_number)

        return self.append_entry(text, time)

    def append_entry(
        self, text: str, time: str | None = None, *, created_at: int | None = None
    ) -> TodoItem:
        """Append a new unchecked todo entry without line validation.

        Args:
            text: Body of the todo item.
            time: Optional scheduled time in "HH:MM" format. If not provided,
                  will be extracted from text if present as (HH:MM) suffix.
            created_at: Optional creation timestamp to preserve (e.g., when moving todos).
                        If not provided, uses current time.

        Returns:
            The newly created TodoItem.
        """
        body = self._validated_text(text)

        # Extract time from text if not explicitly provided
        if time is None:
            time_match = TIME_RE.search(body)
            if time_match:
                hour_str = time_match.group(1).split(":", 1)[0]
                try:
                    if 0 <= int(hour_str) <= 23:
                        time = time_match.group(1)
                        body = body[: time_match.start()].rstrip()
                except ValueError:
                    pass

        now = _now_ms()
        item = TodoItem(
            index=len(self.items) + 1,
            text=body,
            time=time,
            completed=False,
            cancelled=False,
            created_at=created_at if created_at is not None else now,
            updated_at=now,
        )

        self.items.append(item)
        self.save()
        return item

    def cancel_entry(self, line_number: int) -> TodoItem:
        """Cancel a todo entry (soft delete).

        Args:
            line_number: 1-based index of the entry to cancel.

        Returns:
            The cancelled TodoItem.
        """
        _, item = self._get_item(line_number)

        item.cancelled = True
        item.updated_at = _now_ms()
        self.save()
        return item

    def mark_done(self, line_number: int) -> TodoItem:
        """Mark a todo entry complete.

        Args:
            line_number: 1-based index of the entry to mark as done.

        Returns:
            The updated TodoItem.
        """
        _, item = self._get_item(line_number)

        item.completed = True
        item.updated_at = _now_ms()
        self.save()
        return item

    def mark_undone(self, line_number: int) -> TodoItem:
        """Mark a todo entry incomplete.

        Args:
            line_number: 1-based index of the entry to mark as not done.

        Returns:
            The updated TodoItem.
        """
        _, item = self._get_item(line_number)

        item.completed = False
        item.updated_at = _now_ms()
        self.save()
        return item

    def update_entry_text(self, line_number: int, text: str) -> TodoItem:
        """Replace the body text of an entry while keeping its completion state.

        Args:
            line_number: 1-based index of the entry to update.
            text: New text for the todo item.

        Returns:
            The updated TodoItem.
        """
        _, item = self._get_item(line_number)
        body = self._validated_text(text)

        # Extract time from new text
        time: str | None = None
        time_match = TIME_RE.search(body)
        if time_match:
            hour_str = time_match.group(1).split(":", 1)[0]
            try:
                if 0 <= int(hour_str) <= 23:
                    time = time_match.group(1)
                    body = body[: time_match.start()].rstrip()
            except ValueError:
                pass

        item.text = body
        item.time = time
        item.updated_at = _now_ms()
        self.save()
        return item

    def get_item(self, line_number: int) -> TodoItem:
        """Get a todo item by line number.

        Args:
            line_number: 1-based index of the entry.

        Returns:
            The TodoItem at that line number.
        """
        _, item = self._get_item(line_number)
        return item


def todo_file_path(day: str, facet: str) -> Path:
    """Return the absolute path to ``facets/{facet}/todos/{day}.jsonl``.

    Args:
        day: Journal day in ``YYYYMMDD`` format.
        facet: Facet name (e.g., "personal", "work").

    Returns:
        Path to the facet-scoped todo file for the specified day.
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH", "journal")
    return Path(journal) / "facets" / facet / "todos" / f"{day}.jsonl"


def validate_line_number(line_number: int, max_line: int) -> None:
    """Ensure ``line_number`` is within ``[1, max_line]`` inclusive."""
    if line_number < 1 or line_number > max_line:
        raise IndexError(f"line number {line_number} is out of range (1..{max_line})")


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

    return [item.as_dict() for item in checklist.items]


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
        followed by todo display lines. When no upcoming items exist the string
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
    all_todos: list[tuple[str, str, list[str]]] = []  # (facet, day, display_lines)

    for facet_name, facet_path in facet_paths:
        todos_dir = facet_path / "todos"
        if not todos_dir.is_dir():
            continue

        try:
            todo_files = sorted(
                f.stem
                for f in todos_dir.iterdir()
                if f.is_file()
                and f.suffix == ".jsonl"
                and len(f.stem) == 8
                and f.stem.isdigit()
                and f.stem > today_str
            )
        except OSError:  # pragma: no cover - filesystem failure
            continue

        for day_str in todo_files:
            checklist = TodoChecklist.load(day_str, facet_name)
            # Filter to non-cancelled items for display
            display_lines = [
                item.display_line() for item in checklist.items if not item.cancelled
            ]
            if display_lines:
                all_todos.append((facet_name, day_str, display_lines))

    if not all_todos:
        return "No upcoming todos."

    # Sort by day, then facet
    all_todos.sort(key=lambda x: (x[1], x[0]))

    # Build output sections
    remaining = limit
    sections: list[str] = []

    for facet_name, day_str, lines in all_todos:
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
            section = "\n".join([f"### {facet_title}: {day_str}"] + day_lines)
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

            todo_path = facet_dir / "todos" / f"{day}.jsonl"
            if todo_path.is_file():
                facets_with_todos.append(facet_dir.name)
    except OSError:  # pragma: no cover - filesystem failure
        return []

    return sorted(facets_with_todos)


def format_todos(
    entries: list[dict],
    context: dict | None = None,
) -> tuple[list[dict], dict]:
    """Format todo JSONL entries to markdown chunks.

    This is the formatter function used by the formatters registry.

    Args:
        entries: Raw JSONL entries (one todo item per line)
        context: Optional context with:
            - file_path: Path to JSONL file (for extracting facet name and day)

    Returns:
        Tuple of (chunks, meta) where:
            - chunks: List of {"timestamp": int, "markdown": str} dicts, one per item
            - meta: Dict with optional "header" and "error" keys
    """
    ctx = context or {}
    file_path = ctx.get("file_path")
    meta: dict[str, Any] = {}
    chunks: list[dict[str, Any]] = []
    skipped_count = 0

    # Extract facet name and day from path
    facet_name = "unknown"
    day_str: str | None = None
    file_mtime_ms = 0

    if file_path:
        file_path = Path(file_path)

        # Get file modification time as fallback timestamp (in milliseconds)
        try:
            file_mtime_ms = int(file_path.stat().st_mtime * 1000)
        except (OSError, ValueError):
            pass

        # Extract facet name from path
        # Pattern: facets/{facet}/todos/{day}.jsonl
        path_str = str(file_path)
        facet_match = re.search(r"facets/([^/]+)/todos", path_str)
        if facet_match:
            facet_name = facet_match.group(1)

        # Extract day from filename
        if file_path.stem.isdigit() and len(file_path.stem) == 8:
            day_str = file_path.stem

    # Build header
    if day_str:
        formatted_day = f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}"
        header_title = f"# Todos: {facet_name} ({formatted_day})"
    else:
        header_title = f"# Todos: {facet_name}"

    item_count = len(entries)
    meta["header"] = f"{header_title}\n\n{item_count} items"

    # Format each todo item as a chunk
    for i, entry in enumerate(entries):
        # Skip entries without text field
        if "text" not in entry:
            skipped_count += 1
            continue

        # Create TodoItem using existing from_jsonl
        item = TodoItem.from_jsonl(entry, i + 1)

        # Determine timestamp: updated_at -> created_at -> file mtime
        ts = item.updated_at or item.created_at or file_mtime_ms

        # Format as list item using existing display_line
        markdown = f"* {item.display_line()}"

        chunks.append({"timestamp": ts, "markdown": markdown})

    # Report skipped entries
    if skipped_count > 0:
        error_msg = f"Skipped {skipped_count} entries missing 'text' field"
        if file_path:
            error_msg += f" in {file_path}"
        meta["error"] = error_msg
        logging.info(error_msg)

    return chunks, meta
