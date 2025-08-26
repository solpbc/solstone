"""Tests for TODO.md parsing functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from think.utils import get_todos


@pytest.fixture
def todo_content():
    """Sample TODO.md content for testing."""
    return """# Today
- [x] **Meeting**: Weekly standup with team (09:30)
- [ ] **Task**: Implement audio capture improvements #hear (10:15)
- [ ] **Review**: Check PR comments on indexer module #think (11:45)
- [ ] ~~**Task**: Update documentation for API~~ #dream (14:20)
- [ ] **Fix**: Debug transcription timeout issue (15:00)

# Future
- [ ] **Goal**: Design new agent persona system #think 01/25/2025
- [ ] **Research**: Investigate alternative OCR libraries #see 01/26/2025
- [ ] **Task**: Refactor domain summary generation 02/01/2025
- [ ] ~~**Meeting**: Cancelled planning session~~ #personal 01/27/2025
"""


@pytest.fixture
def mock_journal(tmp_path, todo_content):
    """Create a mock journal with TODO.md file."""
    day_dir = tmp_path / "20250124"
    day_dir.mkdir()
    todo_file = day_dir / "TODO.md"
    todo_file.write_text(todo_content)
    return tmp_path


def test_get_todos_parses_today_section(mock_journal):
    """Test parsing of Today section items."""
    with patch.dict("os.environ", {"JOURNAL_PATH": str(mock_journal)}):
        todos = get_todos("20250124")

        assert todos is not None
        assert "today" in todos
        assert len(todos["today"]) == 5

        # Check completed meeting
        meeting = todos["today"][0]
        assert meeting["type"] == "Meeting"
        assert meeting["description"] == "Weekly standup with team"
        assert meeting["completed"] is True
        assert meeting["cancelled"] is False
        assert meeting["time"] == "09:30"
        assert "domain" not in meeting

        # Check task with domain
        task = todos["today"][1]
        assert task["type"] == "Task"
        assert task["description"] == "Implement audio capture improvements"
        assert task["completed"] is False
        assert task["cancelled"] is False
        assert task["time"] == "10:15"
        assert task["domain"] == "hear"

        # Check cancelled task
        cancelled = todos["today"][3]
        assert cancelled["type"] == "Task"
        assert cancelled["description"] == "Update documentation for API"
        assert cancelled["completed"] is False
        assert cancelled["cancelled"] is True
        assert cancelled["time"] == "14:20"
        assert cancelled["domain"] == "dream"


def test_get_todos_parses_future_section(mock_journal):
    """Test parsing of Future section items (now sorted by date)."""
    with patch.dict("os.environ", {"JOURNAL_PATH": str(mock_journal)}):
        todos = get_todos("20250124")

        assert todos is not None
        assert "future" in todos
        assert len(todos["future"]) == 4

        # Items should be sorted by date: 01/25, 01/26, 01/27, 02/01

        # First item (01/25/2025) - Goal
        goal = todos["future"][0]
        assert goal["type"] == "Goal"
        assert goal["description"] == "Design new agent persona system"
        assert goal["completed"] is False
        assert goal["cancelled"] is False
        assert goal["date"] == "01/25/2025"
        assert goal["domain"] == "think"

        # Second item (01/26/2025) - Research task
        research = todos["future"][1]
        assert research["type"] == "Research"
        assert research["description"] == "Investigate alternative OCR libraries"
        assert research["date"] == "01/26/2025"
        assert research["domain"] == "see"

        # Third item (01/27/2025) - Cancelled meeting
        cancelled_meeting = todos["future"][2]
        assert cancelled_meeting["type"] == "Meeting"
        assert cancelled_meeting["description"] == "Cancelled planning session"
        assert cancelled_meeting["cancelled"] is True
        assert cancelled_meeting["date"] == "01/27/2025"
        assert cancelled_meeting["domain"] == "personal"

        # Fourth item (02/01/2025) - Task
        task = todos["future"][3]
        assert task["type"] == "Task"
        assert task["description"] == "Refactor domain summary generation"
        assert task["date"] == "02/01/2025"


def test_get_todos_returns_none_for_missing_file(mock_journal):
    """Test that get_todos returns None when TODO.md doesn't exist."""
    with patch.dict("os.environ", {"JOURNAL_PATH": str(mock_journal)}):
        todos = get_todos("20250125")  # Different day with no TODO.md
        assert todos is None


def test_get_todos_handles_empty_sections(tmp_path):
    """Test handling of empty Today/Future sections."""
    day_dir = tmp_path / "20250124"
    day_dir.mkdir()
    todo_file = day_dir / "TODO.md"
    todo_file.write_text("# Today\n\n# Future\n")

    with patch.dict("os.environ", {"JOURNAL_PATH": str(tmp_path)}):
        todos = get_todos("20250124")

        assert todos is not None
        assert todos["today"] == []
        assert todos["future"] == []


def test_get_todos_handles_malformed_lines(tmp_path):
    """Test that malformed lines are skipped gracefully."""
    day_dir = tmp_path / "20250124"
    day_dir.mkdir()
    todo_file = day_dir / "TODO.md"
    todo_file.write_text(
        """# Today
- [ ] **Task**: Valid task with type (10:00)
- This is not a valid todo line
- [ ] **Task** Missing colon description (12:00)
- [ ] Valid task without type
"""
    )

    with patch.dict("os.environ", {"JOURNAL_PATH": str(tmp_path)}):
        todos = get_todos("20250124")

        assert todos is not None
        # Now with optional types, more lines are valid
        assert len(todos["today"]) == 3

        # First has type
        assert todos["today"][0]["type"] == "Task"
        assert todos["today"][0]["description"] == "Valid task with type"
        assert todos["today"][0]["time"] == "10:00"

        # Second line without proper type format is now valid (no type)
        assert todos["today"][1]["type"] is None
        assert todos["today"][1]["description"] == "**Task** Missing colon description"
        assert todos["today"][1]["time"] == "12:00"

        # Third has no type
        assert todos["today"][2]["type"] is None
        assert todos["today"][2]["description"] == "Valid task without type"
        assert todos["today"][2]["time"] is None


def test_get_todos_json_serializable(mock_journal):
    """Test that the returned structure is JSON serializable."""
    with patch.dict("os.environ", {"JOURNAL_PATH": str(mock_journal)}):
        todos = get_todos("20250124")

        # Should not raise an exception
        json_str = json.dumps(todos)
        assert json_str is not None

        # Verify round-trip
        parsed = json.loads(json_str)
        assert parsed["today"][0]["type"] == "Meeting"
        assert parsed["future"][0]["type"] == "Goal"


def test_get_todos_handles_various_time_formats(tmp_path):
    """Test that various time formats are handled correctly."""
    day_dir = tmp_path / "20250124"
    day_dir.mkdir()
    todo_file = day_dir / "TODO.md"
    todo_file.write_text(
        """# Today
- [ ] **Task**: Morning task (09:01)
- [ ] **Meeting**: Noon meeting (12:00)
- [ ] **Review**: Afternoon review (14:33)
- [ ] **Task**: Late night work (23:59)
- [ ] **Fix**: Midnight task (00:00)
- [ ] **Task**: Invalid time (25:00)
- [ ] **Task**: No time at all

# Future
"""
    )

    with patch.dict("os.environ", {"JOURNAL_PATH": str(tmp_path)}):
        todos = get_todos("20250124")

        assert todos is not None
        assert len(todos["today"]) == 7

        # Valid times should be preserved
        assert todos["today"][0]["time"] == "09:01"
        assert todos["today"][1]["time"] == "12:00"
        assert todos["today"][2]["time"] == "14:33"
        assert todos["today"][3]["time"] == "23:59"
        assert todos["today"][4]["time"] == "00:00"

        # Invalid time (25:00) should not be parsed as time
        assert todos["today"][5]["time"] is None
        assert "25:00" in todos["today"][5]["description"]

        # No time should result in None
        assert todos["today"][6]["time"] is None


def test_get_todos_optional_type_prefix(tmp_path):
    """Test that todos work with optional type prefixes."""
    day_dir = tmp_path / "20250124"
    day_dir.mkdir()
    todo_file = day_dir / "TODO.md"
    todo_file.write_text(
        """# Today
- [ ] **Meeting**: Weekly standup with team (09:30)
- [ ] Fix the broken test in utils.py (10:00)
- [x] Update documentation (11:00)
- [ ] **Review**: Code review for feature branch #review (14:00)
- [ ] Prepare presentation slides #meeting (15:30)

# Future
- [ ] **Goal**: Complete Q1 objectives 03/31/2025
- [ ] Plan team retreat 02/15/2025
- [ ] **Research**: Investigate new framework #research 02/01/2025
- [ ] Migrate database to new schema 02/10/2025
"""
    )

    with patch.dict("os.environ", {"JOURNAL_PATH": str(tmp_path)}):
        todos = get_todos("20250124")

        assert todos is not None
        assert "today" in todos
        assert "future" in todos

        # Check today's todos
        today_items = todos["today"]
        assert len(today_items) == 5

        # Items with type prefix
        assert today_items[0]["type"] == "Meeting"
        assert today_items[0]["description"] == "Weekly standup with team"
        assert today_items[0]["time"] == "09:30"

        assert today_items[3]["type"] == "Review"
        assert today_items[3]["description"] == "Code review for feature branch"
        assert today_items[3]["domain"] == "review"

        # Items without type prefix (should have None as type)
        assert today_items[1]["type"] is None
        assert today_items[1]["description"] == "Fix the broken test in utils.py"
        assert today_items[1]["time"] == "10:00"
        assert today_items[1]["completed"] is False

        assert today_items[2]["type"] is None
        assert today_items[2]["description"] == "Update documentation"
        assert today_items[2]["time"] == "11:00"
        assert today_items[2]["completed"] is True

        assert today_items[4]["type"] is None
        assert today_items[4]["description"] == "Prepare presentation slides"
        assert today_items[4]["time"] == "15:30"
        assert today_items[4]["domain"] == "meeting"

        # Check future todos (now sorted by date)
        future_items = todos["future"]
        assert len(future_items) == 4

        # Items should be sorted by date: 02/01, 02/10, 02/15, 03/31

        # First item (02/01/2025) - Research with type prefix
        assert future_items[0]["type"] == "Research"
        assert future_items[0]["description"] == "Investigate new framework"
        assert future_items[0]["date"] == "02/01/2025"
        assert future_items[0]["domain"] == "research"

        # Second item (02/10/2025) - No type prefix
        assert future_items[1]["type"] is None
        assert future_items[1]["description"] == "Migrate database to new schema"
        assert future_items[1]["date"] == "02/10/2025"

        # Third item (02/15/2025) - No type prefix
        assert future_items[2]["type"] is None
        assert future_items[2]["description"] == "Plan team retreat"
        assert future_items[2]["date"] == "02/15/2025"

        # Fourth item (03/31/2025) - Goal with type prefix
        assert future_items[3]["type"] == "Goal"
        assert future_items[3]["description"] == "Complete Q1 objectives"
        assert future_items[3]["date"] == "03/31/2025"


def test_timestamp_preserves_domain_tags(tmp_path):
    """Test that timestamps work correctly with domain tags."""
    day_dir = tmp_path / "20250124"
    day_dir.mkdir()
    todo_file = day_dir / "TODO.md"
    todo_file.write_text(
        """# Today
- [ ] **Task**: Task with domain #hear (10:30)
- [ ] **Meeting**: Meeting without time #dream
- [ ] **Review**: Review with both #think (14:45)

# Future
"""
    )

    with patch.dict("os.environ", {"JOURNAL_PATH": str(tmp_path)}):
        todos = get_todos("20250124")

        assert todos is not None
        assert len(todos["today"]) == 3

        # Domain and time should both be extracted
        assert todos["today"][0]["domain"] == "hear"
        assert todos["today"][0]["time"] == "10:30"
        assert todos["today"][0]["description"] == "Task with domain"

        # Domain without time
        assert todos["today"][1]["domain"] == "dream"
        assert todos["today"][1]["time"] is None
        assert todos["today"][1]["description"] == "Meeting without time"

        # Both domain and time
        assert todos["today"][2]["domain"] == "think"
        assert todos["today"][2]["time"] == "14:45"
        assert todos["today"][2]["description"] == "Review with both"
