"""Tests for the simplified todos/today.md checklist parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from think.todo import get_todos, parse_item, parse_items


@pytest.fixture
def journal_root(tmp_path):
    path = tmp_path / "journal"
    path.mkdir()
    return path


def _write_todos(root: Path, day: str, lines: list[str]) -> Path:
    day_dir = root / day / "todos"
    day_dir.mkdir(parents=True)
    todo_path = day_dir / "today.md"
    todo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return todo_path


def test_get_todos_returns_none_when_missing(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    assert get_todos("20240101") is None


def test_get_todos_parses_basic_fields(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    _write_todos(
        journal_root,
        "20240102",
        [
            "- [ ] **Review**: Merge analytics PR #think (10:30)",
            "- [x] **Meeting**: Project sync",  # no domain/time
            "- [ ] Write retrospective notes",
        ],
    )

    todos = get_todos("20240102")
    assert todos is not None
    assert len(todos) == 3

    first = todos[0]
    assert first["index"] == 1
    assert first["domain"] == "think"
    assert first["time"] == "10:30"
    assert first["completed"] is False
    assert first["text"] == "Merge analytics PR"

    second = todos[1]
    assert second["completed"] is True
    assert second["domain"] is None
    assert second["time"] is None
    assert second["text"] == "Project sync"

    third = todos[2]
    assert third["text"] == "Write retrospective notes"
    assert third["index"] == 3


def test_get_todos_handles_strikethrough_and_spacing(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    _write_todos(
        journal_root,
        "20240103",
        [
            "- [ ] ~~**Task**: Optional experiment if time allows~~",
            "  - [ ]  **Fix**: Address bug #backend  (14:45)  ",
            "not a todo line",
            "- [x] **Research**: Draft report #think",
        ],
    )

    todos = get_todos("20240103")
    assert todos is not None
    assert len(todos) == 3

    cancelled = todos[0]
    assert cancelled["cancelled"] is True
    assert cancelled["text"] == "Optional experiment if time allows"

    second = todos[1]
    assert second["domain"] == "backend"
    assert second["time"] == "14:45"
    assert second["index"] == 2

    third = todos[2]
    assert third["completed"] is True
    assert third["text"] == "Draft report"


def test_get_todos_ignores_blank_lines(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    _write_todos(
        journal_root,
        "20240104",
        [
            "",
            "- [ ] First",
            "# comment line",
            "- [ ] Second",
        ],
    )

    todos = get_todos("20240104")
    assert [item["index"] for item in todos] == [1, 2]


def test_parse_item_valid_todo():
    item = parse_item("- [ ] Simple todo", 1)
    assert item is not None
    assert item.index == 1
    assert item.text == "Simple todo"
    assert item.completed is False
    assert item.cancelled is False
    assert item.domain is None
    assert item.time is None


def test_parse_item_completed_todo():
    item = parse_item("- [x] Completed task", 2)
    assert item is not None
    assert item.completed is True
    assert item.text == "Completed task"


def test_parse_item_with_domain():
    item = parse_item("- [ ] Review PR #backend", 3)
    assert item is not None
    assert item.domain == "backend"
    assert item.text == "Review PR"


def test_parse_item_with_time():
    item = parse_item("- [ ] Meeting at (14:30)", 4)
    assert item is not None
    assert item.time == "14:30"
    assert item.text == "Meeting at"


def test_parse_item_cancelled():
    item = parse_item("- [ ] ~~Cancelled task~~", 5)
    assert item is not None
    assert item.cancelled is True
    assert item.text == "Cancelled task"


def test_parse_item_with_markup():
    item = parse_item("- [ ] **Task**: Do something", 6)
    assert item is not None
    assert item.text == "Do something"


def test_parse_item_complex():
    item = parse_item("- [x] **Review**: Merge PR #think (10:30)", 7)
    assert item is not None
    assert item.completed is True
    assert item.domain == "think"
    assert item.time == "10:30"
    assert item.text == "Merge PR"


def test_parse_item_invalid_lines():
    assert parse_item("", 1) is None
    assert parse_item("Not a todo", 2) is None
    assert parse_item("# Comment", 3) is None
    assert parse_item("   ", 4) is None


def test_parse_items_maintains_sequential_index():
    lines = [
        "- [ ] First",
        "",
        "Not a todo",
        "- [x] Second",
        "- [ ] Third",
    ]
    items = parse_items(lines)
    assert len(items) == 3
    assert items[0].index == 1
    assert items[0].text == "First"
    assert items[1].index == 2
    assert items[1].text == "Second"
    assert items[2].index == 3
    assert items[2].text == "Third"


def test_append_entry_validates_parsing(monkeypatch, journal_root):
    from think.todo import TodoChecklist

    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))

    # Create a day directory
    day_dir = journal_root / "20240105" / "todos"
    day_dir.mkdir(parents=True)

    # Create domains with valid domain files
    domains_dir = journal_root / "domains"
    domains_dir.mkdir(parents=True)
    for domain in ["work", "personal", "hobby"]:
        domain_path = domains_dir / domain
        domain_path.mkdir(parents=True)
        domain_json = domain_path / "domain.json"
        domain_json.write_text(f'{{"title": "{domain.title()}"}}', encoding="utf-8")

    checklist = TodoChecklist.load("20240105")

    # Test normal entry works
    checklist.append_entry("Test task #work (10:30)")
    assert len(checklist.entries) == 1

    # Verify it parses correctly
    items = parse_items(checklist.entries)
    assert len(items) == 1
    assert items[0].text == "Test task"
    assert items[0].domain == "work"
    assert items[0].time == "10:30"


def test_append_entry_validates_domain(monkeypatch, journal_root):
    from think.todo import TodoChecklist, TodoDomainError

    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))

    # Create a day directory
    day_dir = journal_root / "20240106" / "todos"
    day_dir.mkdir(parents=True)

    # Create domains with valid domain files
    domains_dir = journal_root / "domains"
    domains_dir.mkdir(parents=True)
    for domain in ["work", "personal", "hobby"]:
        domain_path = domains_dir / domain
        domain_path.mkdir(parents=True)
        domain_json = domain_path / "domain.json"
        domain_json.write_text(f'{{"title": "{domain.title()}"}}', encoding="utf-8")

    checklist = TodoChecklist.load("20240106")

    # Test invalid domain raises error
    with pytest.raises(TodoDomainError) as exc_info:
        checklist.append_entry("Test task #invalid (10:30)")

    assert exc_info.value.invalid_domain == "invalid"
    assert set(exc_info.value.valid_domains) == {"work", "personal", "hobby"}
