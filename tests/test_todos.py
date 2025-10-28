"""Tests for the domain-scoped todo checklist system."""

from __future__ import annotations

from pathlib import Path

import pytest

from think.todo import (
    get_domains_with_todos,
    get_todos,
    parse_item,
    parse_items,
    upcoming,
)


@pytest.fixture
def journal_root(tmp_path):
    path = tmp_path / "journal"
    path.mkdir()
    return path


def _write_todos(root: Path, domain: str, day: str, lines: list[str]) -> Path:
    """Write todos to domains/{domain}/todos/{day}.md"""
    todos_dir = root / "domains" / domain / "todos"
    todos_dir.mkdir(parents=True, exist_ok=True)
    todo_path = todos_dir / f"{day}.md"
    todo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return todo_path


def test_get_todos_returns_none_when_missing(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    assert get_todos("20240101", "personal") is None


def test_get_todos_parses_basic_fields(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    _write_todos(
        journal_root,
        "personal",
        "20240102",
        [
            "- [ ] **Review**: Merge analytics PR (10:30)",
            "- [x] **Meeting**: Project sync",  # no time
            "- [ ] Write retrospective notes",
        ],
    )

    todos = get_todos("20240102", "personal")
    assert todos is not None
    assert len(todos) == 3

    first = todos[0]
    assert first["index"] == 1
    assert first["time"] == "10:30"
    assert first["completed"] is False
    assert first["text"] == "Merge analytics PR"
    assert "domain" not in first  # domain field removed

    second = todos[1]
    assert second["completed"] is True
    assert second["time"] is None
    assert second["text"] == "Project sync"

    third = todos[2]
    assert third["text"] == "Write retrospective notes"
    assert third["index"] == 3


def test_get_todos_handles_strikethrough_and_spacing(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    _write_todos(
        journal_root,
        "work",
        "20240103",
        [
            "- [ ] ~~**Task**: Optional experiment if time allows~~",
            "  - [ ]  **Fix**: Address bug  (14:45)  ",
            "not a todo line",
            "- [x] **Research**: Draft report",
        ],
    )

    todos = get_todos("20240103", "work")
    assert todos is not None
    assert len(todos) == 3

    cancelled = todos[0]
    assert cancelled["cancelled"] is True
    assert cancelled["text"] == "Optional experiment if time allows"

    second = todos[1]
    assert second["time"] == "14:45"
    assert second["index"] == 2

    third = todos[2]
    assert third["completed"] is True
    assert third["text"] == "Draft report"


def test_get_todos_ignores_blank_lines(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    _write_todos(
        journal_root,
        "personal",
        "20240104",
        [
            "",
            "- [ ] First",
            "# comment line",
            "- [ ] Second",
        ],
    )

    todos = get_todos("20240104", "personal")
    assert [item["index"] for item in todos] == [1, 2]


def test_parse_item_valid_todo():
    item = parse_item("- [ ] Simple todo", 1)
    assert item is not None
    assert item.index == 1
    assert item.text == "Simple todo"
    assert item.completed is False
    assert item.cancelled is False
    assert item.time is None


def test_parse_item_completed_todo():
    item = parse_item("- [x] Completed task", 2)
    assert item is not None
    assert item.completed is True
    assert item.text == "Completed task"


def test_parse_item_with_time():
    item = parse_item("- [ ] Meeting at (14:30)", 3)
    assert item is not None
    assert item.time == "14:30"
    assert item.text == "Meeting at"


def test_parse_item_cancelled():
    item = parse_item("- [ ] ~~Cancelled task~~", 4)
    assert item is not None
    assert item.cancelled is True
    assert item.text == "Cancelled task"


def test_parse_item_with_markup():
    item = parse_item("- [ ] **Task**: Do something", 5)
    assert item is not None
    assert item.text == "Do something"


def test_parse_item_complex():
    item = parse_item("- [x] **Review**: Merge PR (10:30)", 6)
    assert item is not None
    assert item.completed is True
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


def test_upcoming_groups_future_days(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    # Create domain structure
    (journal_root / "domains" / "personal").mkdir(parents=True)
    (journal_root / "domains" / "personal" / "domain.json").write_text(
        '{"title": "Personal"}', encoding="utf-8"
    )

    _write_todos(
        journal_root,
        "personal",
        "20240105",
        [
            "- [ ] First future task",
            "- [x] Completed future task",
        ],
    )
    _write_todos(
        journal_root,
        "personal",
        "20240106",
        [
            "- [ ] Another future task",
        ],
    )
    _write_todos(
        journal_root,
        "personal",
        "20240103",
        [
            "- [ ] Past task",
        ],
    )

    result = upcoming(today="20240104")

    expected = (
        "### Personal: 20240105\n"
        "- [ ] First future task\n"
        "- [x] Completed future task\n\n"
        "### Personal: 20240106\n"
        "- [ ] Another future task"
    )

    assert result == expected


def test_upcoming_respects_limit(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    # Create domain structure
    (journal_root / "domains" / "work").mkdir(parents=True)
    (journal_root / "domains" / "work" / "domain.json").write_text(
        '{"title": "Work"}', encoding="utf-8"
    )

    _write_todos(
        journal_root,
        "work",
        "20240105",
        [
            "- [ ] Task one",
            "- [ ] Task two",
            "- [ ] Task three",
        ],
    )

    result = upcoming(limit=2, today="20240104")

    expected = "### Work: 20240105\n" "- [ ] Task one\n" "- [ ] Task two"

    assert result == expected


def test_upcoming_when_no_future_todos(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    (journal_root / "domains" / "personal").mkdir(parents=True)

    _write_todos(
        journal_root,
        "personal",
        "20240102",
        [
            "- [ ] Existing task",
        ],
    )

    result = upcoming(today="20240102")

    assert result == "No upcoming todos."


def test_upcoming_filters_by_domain(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    # Create multiple domains
    for domain_name in ["personal", "work"]:
        domain_dir = journal_root / "domains" / domain_name
        domain_dir.mkdir(parents=True)
        (domain_dir / "domain.json").write_text(
            f'{{"title": "{domain_name.title()}"}}', encoding="utf-8"
        )

    _write_todos(journal_root, "personal", "20240105", ["- [ ] Personal task"])
    _write_todos(journal_root, "work", "20240105", ["- [ ] Work task"])

    # Test filtering by domain
    result = upcoming(domain="personal", today="20240104")
    assert "Personal: 20240105" in result
    assert "Personal task" in result
    assert "Work task" not in result


def test_upcoming_aggregates_all_domains(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))
    # Create multiple domains
    for domain_name in ["personal", "work"]:
        domain_dir = journal_root / "domains" / domain_name
        domain_dir.mkdir(parents=True)
        (domain_dir / "domain.json").write_text(
            f'{{"title": "{domain_name.title()}"}}', encoding="utf-8"
        )

    _write_todos(journal_root, "personal", "20240105", ["- [ ] Personal task"])
    _write_todos(journal_root, "work", "20240105", ["- [ ] Work task"])

    # Test aggregation (domain=None)
    result = upcoming(domain=None, today="20240104")
    assert "Personal: 20240105" in result
    assert "Work: 20240105" in result
    assert "Personal task" in result
    assert "Work task" in result


def test_append_entry_validates_parsing(monkeypatch, journal_root):
    from think.todo import TodoChecklist

    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))

    # Create domain directory
    domains_dir = journal_root / "domains" / "work"
    domains_dir.mkdir(parents=True)

    checklist = TodoChecklist.load("20240105", "work")

    # Test normal entry works
    checklist.append_entry("Test task (10:30)")
    assert len(checklist.entries) == 1

    # Verify it parses correctly
    items = parse_items(checklist.entries)
    assert len(items) == 1
    assert items[0].text == "Test task"
    assert items[0].time == "10:30"


def test_get_domains_with_todos(monkeypatch, journal_root):
    monkeypatch.setenv("JOURNAL_PATH", str(journal_root))

    # Create todos in multiple domains
    _write_todos(journal_root, "personal", "20240105", ["- [ ] Personal task"])
    _write_todos(journal_root, "work", "20240105", ["- [ ] Work task"])
    _write_todos(journal_root, "hobby", "20240106", ["- [ ] Hobby task"])

    # Test getting domains for a specific day
    domains_20240105 = get_domains_with_todos("20240105")
    assert sorted(domains_20240105) == ["personal", "work"]

    domains_20240106 = get_domains_with_todos("20240106")
    assert domains_20240106 == ["hobby"]

    domains_20240107 = get_domains_with_todos("20240107")
    assert domains_20240107 == []
