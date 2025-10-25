"""Tests for think.utils module."""

import os
import tempfile
from pathlib import Path

import pytest

from think.entities import load_entity_names


def test_load_entity_names_with_valid_file(monkeypatch):
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

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names()
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


def test_load_entity_names_missing_file(monkeypatch):
    """Test that missing file returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names()
        assert result is None


def test_load_entity_names_empty_file(monkeypatch):
    """Test that empty file returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "entities.md"
        entities_path.write_text("")

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names()
        assert result is None


def test_load_entity_names_no_valid_entries(monkeypatch):
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

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names()
        assert result is None


def test_load_entity_names_with_duplicates(monkeypatch):
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

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names()
        assert result == "John Smith, Acme Corp"

        names = result.split(", ")
        assert len(names) == 2


def test_load_entity_names_handles_special_characters(monkeypatch):
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

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names()
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

        # Should use env var
        result = load_entity_names()
        assert result == "Test User"


def test_load_entity_names_missing_env_var(monkeypatch):
    """Test that missing JOURNAL_PATH returns None."""
    # Ensure JOURNAL_PATH is not set, even after load_dotenv
    monkeypatch.delenv("JOURNAL_PATH", raising=False)
    # Mock load_dotenv to prevent it from loading a .env file
    monkeypatch.setattr("think.entities.load_dotenv", lambda: None)

    result = load_entity_names()
    assert result is None


def test_load_entity_names_spoken_mode(monkeypatch):
    """Test spoken mode returns shortened forms with uniform processing for all types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "entities.md"
        entities_path.write_text(
            """
* Person: Jeremie Miller (Jer) - Software engineer
* Person: Jane Elizabeth Doe - Product manager
* Company: Acme Corporation (ACME) - Tech company
* Company: Widget Inc - Manufacturing company
* Company: Google - Search engine
* Project: Sunstone Project (SUN) - AI journaling
* Project: Project X - Secret project
* Tool: Hammer - For hitting things
* Tool: Docker - Container runtime
"""
        )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names(spoken=True)

        # Should return a list, not a string
        assert isinstance(result, list)

        # Person: "Jeremie Miller (Jer)" -> ["Jeremie", "Jer"]
        assert "Jeremie" in result
        assert "Jer" in result

        # Person: "Jane Elizabeth Doe" -> ["Jane"]
        assert "Jane" in result
        # Should not include middle/last names
        assert "Elizabeth" not in result
        assert "Doe" not in result

        # Company: "Acme Corporation (ACME)" -> ["Acme", "ACME"] (uniform processing)
        assert "Acme" in result  # First word
        assert "ACME" in result  # From parens

        # Company: "Widget Inc" (multi-word) -> ["Widget"]
        assert "Widget" in result

        # Company: "Google" (single word) -> ["Google"]
        assert "Google" in result

        # Project: "Sunstone Project (SUN)" -> ["Sunstone", "SUN"] (uniform processing)
        assert "Sunstone" in result  # First word
        assert "SUN" in result  # From parens

        # Project: "Project X" (no parens) -> ["Project"] (first word only)
        assert "Project" in result

        # Tools are now included (uniform processing for all types)
        assert "Hammer" in result
        assert "Docker" in result


def test_load_entity_names_spoken_mode_with_tools(monkeypatch):
    """Test spoken mode includes tools with uniform processing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "entities.md"
        entities_path.write_text(
            """
* Tool: Hammer - For hitting things
* Tool: Docker - Container runtime
"""
        )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names(spoken=True)
        # Tools are now included (uniform processing)
        assert isinstance(result, list)
        assert "Hammer" in result
        assert "Docker" in result


def test_load_entity_names_spoken_mode_duplicates(monkeypatch):
    """Test spoken mode filters out duplicate shortened forms."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "entities.md"
        entities_path.write_text(
            """
* Person: John Smith - Engineer
* Person: John Doe - Manager
* Company: Acme Corp - Tech
* Company: Acme Industries - Manufacturing
"""
        )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names(spoken=True)

        # Should have only one "John" and one "Acme" even though there are two of each
        assert result.count("John") == 1
        assert result.count("Acme") == 1


def test_load_entity_names_uniform_processing(monkeypatch):
    """Test that uniform processing works correctly for all entity types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "entities.md"
        entities_path.write_text(
            """
* Person: Ryan Reed (R2) - Software developer
* Company: Federal Aviation Administration (FAA) - Government agency
* Project: Backend API (API) - Core service
* Tool: pytest - Testing framework
* Location: New York City (NYC) - Metropolitan area
"""
        )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names(spoken=True)

        assert isinstance(result, list)

        # "Ryan Reed (R2)" -> ["Ryan", "R2"]
        assert "Ryan" in result
        assert "R2" in result
        assert "Reed" not in result

        # "Federal Aviation Administration (FAA)" -> ["Federal", "FAA"]
        assert "Federal" in result
        assert "FAA" in result
        assert "Aviation" not in result
        assert "Administration" not in result

        # "Backend API (API)" -> ["Backend", "API"]
        assert "Backend" in result
        assert "API" in result

        # "pytest" -> ["pytest"]
        assert "pytest" in result

        # "New York City (NYC)" -> ["New", "NYC"]
        assert "New" in result
        assert "NYC" in result
        assert "York" not in result
        assert "City" not in result
