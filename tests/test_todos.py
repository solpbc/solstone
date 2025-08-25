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
    """Test parsing of Future section items."""
    with patch.dict("os.environ", {"JOURNAL_PATH": str(mock_journal)}):
        todos = get_todos("20250124")
        
        assert todos is not None
        assert "future" in todos
        assert len(todos["future"]) == 4
        
        # Check goal with domain
        goal = todos["future"][0]
        assert goal["type"] == "Goal"
        assert goal["description"] == "Design new agent persona system"
        assert goal["completed"] is False
        assert goal["cancelled"] is False
        assert goal["date"] == "01/25/2025"
        assert goal["domain"] == "think"
        
        # Check research task
        research = todos["future"][1]
        assert research["type"] == "Research"
        assert research["description"] == "Investigate alternative OCR libraries"
        assert research["date"] == "01/26/2025"
        assert research["domain"] == "see"
        
        # Check cancelled meeting
        cancelled_meeting = todos["future"][3]
        assert cancelled_meeting["type"] == "Meeting"
        assert cancelled_meeting["description"] == "Cancelled planning session"
        assert cancelled_meeting["cancelled"] is True
        assert cancelled_meeting["date"] == "01/27/2025"
        assert cancelled_meeting["domain"] == "personal"


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
    todo_file.write_text("""# Today
- [ ] **Task**: Valid task (10:00)
- This is not a valid todo line
- [x] Missing bold type: description (11:00)
- [ ] **Task** Missing colon description (12:00)
- [ ] **Task**: Valid task without time
""")
    
    with patch.dict("os.environ", {"JOURNAL_PATH": str(tmp_path)}):
        todos = get_todos("20250124")
        
        assert todos is not None
        assert len(todos["today"]) == 2  # Only valid lines parsed
        assert todos["today"][0]["description"] == "Valid task"
        assert todos["today"][1]["description"] == "Valid task without time"
        assert todos["today"][1]["time"] is None


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