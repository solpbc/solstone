"""Tests for think.utils module."""

import tempfile
from pathlib import Path

import pytest

from think.utils import load_entity_names


def test_load_entity_names_with_valid_file():
    """Test loading entity names from a valid entities.md file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "entities.md"
        entities_path.write_text("""
* Person: John Smith - A software engineer at Google
* Company: Acme Corp - Technology company based in SF
* Project: Project X - Secret internal project
* Tool: Hammer - For hitting things
* Person: Jane Doe - Product manager at Meta
* Company: Widget Inc - Manufacturing company
""")
        
        result = load_entity_names(tmpdir)
        assert result == "John Smith, Acme Corp, Project X, Hammer, Jane Doe, Widget Inc"
        
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
        entities_path.write_text("""
# Header comment
Some random text
Not a valid entity line
""")
        
        result = load_entity_names(tmpdir)
        assert result is None


def test_load_entity_names_with_duplicates():
    """Test that duplicate names are filtered out."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "entities.md"
        entities_path.write_text("""
* Person: John Smith - Engineer
* Company: Acme Corp - Tech company
* Person: John Smith - Also an engineer
* Company: Acme Corp - Still a tech company
""")
        
        result = load_entity_names(tmpdir)
        assert result == "John Smith, Acme Corp"
        
        names = result.split(", ")
        assert len(names) == 2


def test_load_entity_names_handles_special_characters():
    """Test that names with special characters are handled correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "entities.md"
        entities_path.write_text("""
* Person: Jean-Pierre O'Malley - Engineer
* Company: AT&T - Telecom company
* Project: C++ Compiler - Development tool
* Tool: Node.js - JavaScript runtime
""")
        
        result = load_entity_names(tmpdir)
        assert "Jean-Pierre O'Malley" in result
        assert "AT&T" in result
        assert "C++ Compiler" in result
        assert "Node.js" in result


def test_load_entity_names_with_env_var(monkeypatch):
    """Test loading using JOURNAL_PATH environment variable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "entities.md"
        entities_path.write_text("""
* Person: Test User - A test person
""")
        
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