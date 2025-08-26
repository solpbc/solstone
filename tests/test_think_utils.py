"""Tests for think.utils module."""

import os
import tempfile
from pathlib import Path

import pytest

from think.utils import get_todos, load_entity_names


def test_load_entity_names_with_valid_file():
    """Test loading entity names from a valid entities.md file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "entities.md"
        entities_path.write_text(
            """
* Person: John Smith - A software engineer at Google
* Company: Acme Corp - Technology company based in SF
* Project: Project X - Secret internal project
* Tool: Hammer - For hitting things
* Person: Jane Doe - Product manager at Meta
* Company: Widget Inc - Manufacturing company
"""
        )

        result = load_entity_names(tmpdir)
        assert (
            result == "John Smith, Acme Corp, Project X, Hammer, Jane Doe, Widget Inc"
        )

        # Check that names are extracted without duplicates
        names = result.split(", ")
        assert len(names) == 6
        assert "John Smith" in names
        assert "Acme Corp" in names
        assert "Project X" in names
        assert "Hammer" in names
        assert "Jane Doe" in names
        assert "Widget Inc" in names


def test_load_entity_names_missing_file_not_required():
    """Test that missing file returns None when not required."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_entity_names(tmpdir, required=False)
        assert result is None


def test_load_entity_names_missing_file_required():
    """Test that missing file raises FileNotFoundError when required."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError) as exc_info:
            load_entity_names(tmpdir, required=True)
        assert "entities.md" in str(exc_info.value)


def test_load_entity_names_empty_file():
    """Test that empty file returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "entities.md"
        entities_path.write_text("")

        result = load_entity_names(tmpdir)
        assert result is None


def test_load_entity_names_no_valid_entries():
    """Test file with no parseable entity lines returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "entities.md"
        entities_path.write_text(
            """
# Header comment
Some random text
Not a valid entity line
"""
        )

        result = load_entity_names(tmpdir)
        assert result is None


def test_load_entity_names_with_duplicates():
    """Test that duplicate names are filtered out."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "entities.md"
        entities_path.write_text(
            """
* Person: John Smith - Engineer
* Company: Acme Corp - Tech company
* Person: John Smith - Also an engineer
* Company: Acme Corp - Still a tech company
"""
        )

        result = load_entity_names(tmpdir)
        assert result == "John Smith, Acme Corp"

        names = result.split(", ")
        assert len(names) == 2


def test_load_entity_names_handles_special_characters():
    """Test that names with special characters are handled correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "entities.md"
        entities_path.write_text(
            """
* Person: Jean-Pierre O'Malley - Engineer
* Company: AT&T - Telecom company
* Project: C++ Compiler - Development tool
* Tool: Node.js - JavaScript runtime
"""
        )

        result = load_entity_names(tmpdir)
        assert "Jean-Pierre O'Malley" in result
        assert "AT&T" in result
        assert "C++ Compiler" in result
        assert "Node.js" in result


def test_load_entity_names_with_env_var(monkeypatch):
    """Test loading using JOURNAL_PATH environment variable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "entities.md"
        entities_path.write_text(
            """
* Person: Test User - A test person
"""
        )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)

        # Should use env var when journal_path is None
        result = load_entity_names(None)
        assert result == "Test User"


def test_load_entity_names_missing_env_var(monkeypatch):
    """Test that missing JOURNAL_PATH raises ValueError when needed."""
    # Ensure JOURNAL_PATH is not set, even after load_dotenv
    monkeypatch.delenv("JOURNAL_PATH", raising=False)
    # Mock load_dotenv to prevent it from loading a .env file
    monkeypatch.setattr("think.utils.load_dotenv", lambda: None)

    with pytest.raises(ValueError) as exc_info:
        load_entity_names(None)
    assert "JOURNAL_PATH not set" in str(exc_info.value)


def test_get_todos_with_line_numbers():
    """Test that get_todos correctly tracks line numbers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["JOURNAL_PATH"] = tmpdir

        # Create a test day directory with TODO.md
        day_dir = Path(tmpdir) / "20250126"
        day_dir.mkdir()

        todo_content = """# Today
- [ ] **Task**: First task (10:30)
- [x] **Meeting**: Team standup (11:00)
- [ ] ~~Cancelled task~~ (12:00)

# Future
- [ ] **Review**: Code review 01/28/2025
- [ ] Plan next sprint 01/30/2025
- [x] **Done**: Completed future task 02/01/2025
"""

        (day_dir / "TODO.md").write_text(todo_content)

        todos = get_todos("20250126")

        assert todos is not None
        assert "today" in todos
        assert "future" in todos

        # Check today section
        assert len(todos["today"]) == 3

        # First task - line 2
        assert todos["today"][0]["line_number"] == 2
        assert todos["today"][0]["type"] == "Task"
        assert todos["today"][0]["description"] == "First task"
        assert todos["today"][0]["time"] == "10:30"
        assert todos["today"][0]["completed"] is False
        assert todos["today"][0]["cancelled"] is False

        # Second task - line 3
        assert todos["today"][1]["line_number"] == 3
        assert todos["today"][1]["type"] == "Meeting"
        assert todos["today"][1]["description"] == "Team standup"
        assert todos["today"][1]["time"] == "11:00"
        assert todos["today"][1]["completed"] is True
        assert todos["today"][1]["cancelled"] is False

        # Third task - line 4 (cancelled)
        assert todos["today"][2]["line_number"] == 4
        assert todos["today"][2]["description"] == "Cancelled task"
        assert todos["today"][2]["time"] == "12:00"
        assert todos["today"][2]["completed"] is False
        assert todos["today"][2]["cancelled"] is True

        # Check future section
        assert len(todos["future"]) == 3

        # First future task - line 7
        assert todos["future"][0]["line_number"] == 7
        assert todos["future"][0]["type"] == "Review"
        assert todos["future"][0]["description"] == "Code review"
        assert todos["future"][0]["date"] == "01/28/2025"
        assert todos["future"][0]["completed"] is False

        # Second future task - line 8
        assert todos["future"][1]["line_number"] == 8
        assert todos["future"][1]["type"] is None
        assert todos["future"][1]["description"] == "Plan next sprint"
        assert todos["future"][1]["date"] == "01/30/2025"
        assert todos["future"][1]["completed"] is False

        # Third future task - line 9 (completed)
        assert todos["future"][2]["line_number"] == 9
        assert todos["future"][2]["type"] == "Done"
        assert todos["future"][2]["description"] == "Completed future task"
        assert todos["future"][2]["date"] == "02/01/2025"
        assert todos["future"][2]["completed"] is True


def test_get_todos_with_domains():
    """Test that get_todos correctly parses domain tags."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["JOURNAL_PATH"] = tmpdir

        # Create a test day directory with TODO.md
        day_dir = Path(tmpdir) / "20250126"
        day_dir.mkdir()

        todo_content = """# Today
- [ ] Fix bug in authentication #backend (14:00)
- [ ] **Design**: Update landing page #frontend (15:30)

# Future
- [ ] Deploy to production #devops 02/15/2025
- [ ] **Meeting**: Sprint planning #team 02/20/2025
"""

        (day_dir / "TODO.md").write_text(todo_content)

        todos = get_todos("20250126")

        # Check today section with domains
        assert todos["today"][0]["description"] == "Fix bug in authentication"
        assert todos["today"][0]["domain"] == "backend"
        assert todos["today"][0]["time"] == "14:00"
        assert todos["today"][0]["line_number"] == 2

        assert todos["today"][1]["type"] == "Design"
        assert todos["today"][1]["description"] == "Update landing page"
        assert todos["today"][1]["domain"] == "frontend"
        assert todos["today"][1]["time"] == "15:30"
        assert todos["today"][1]["line_number"] == 3

        # Check future section with domains
        assert todos["future"][0]["description"] == "Deploy to production"
        assert todos["future"][0]["domain"] == "devops"
        assert todos["future"][0]["date"] == "02/15/2025"
        assert todos["future"][0]["line_number"] == 6

        assert todos["future"][1]["type"] == "Meeting"
        assert todos["future"][1]["description"] == "Sprint planning"
        assert todos["future"][1]["domain"] == "team"
        assert todos["future"][1]["date"] == "02/20/2025"
        assert todos["future"][1]["line_number"] == 7


def test_get_todos_empty_sections():
    """Test get_todos with empty sections."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["JOURNAL_PATH"] = tmpdir

        # Create a test day directory with TODO.md
        day_dir = Path(tmpdir) / "20250126"
        day_dir.mkdir()

        todo_content = """# Today

# Future

"""

        (day_dir / "TODO.md").write_text(todo_content)

        todos = get_todos("20250126")

        assert todos is not None
        assert len(todos["today"]) == 0
        assert len(todos["future"]) == 0


def test_get_todos_missing_file():
    """Test get_todos when TODO.md doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["JOURNAL_PATH"] = tmpdir

        # Create a test day directory WITHOUT TODO.md
        day_dir = Path(tmpdir) / "20250126"
        day_dir.mkdir()

        todos = get_todos("20250126")
        assert todos is None


def test_get_todos_preserves_line_numbers_with_blank_lines():
    """Test that line numbers are preserved even with blank lines and comments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["JOURNAL_PATH"] = tmpdir

        # Create a test day directory with TODO.md
        day_dir = Path(tmpdir) / "20250126"
        day_dir.mkdir()

        todo_content = """# Today

- [ ] First task after blank (10:00)

- [ ] Second task after blank (11:00)
Some random text that should be ignored

# Future

- [ ] Future task 03/01/2025
"""

        (day_dir / "TODO.md").write_text(todo_content)

        todos = get_todos("20250126")

        # Line numbers should match actual lines in file
        assert todos["today"][0]["line_number"] == 3  # "First task after blank"
        assert todos["today"][1]["line_number"] == 5  # "Second task after blank"
        assert todos["future"][0]["line_number"] == 10  # "Future task"
