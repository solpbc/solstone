# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.utils module."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from think.entities import load_entity_names
from think.utils import segment_key, setup_cli


def write_entities_jsonl(path: Path, entities: list[tuple[str, str, str]] | list[dict]):
    """Helper to write entities in JSONL format for tests.

    Args:
        path: Path to entities.jsonl file to write
        entities: Either list of (type, name, desc) tuples or list of entity dicts
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in entities:
            if isinstance(item, dict):
                entity = item
            else:
                etype, name, desc = item
                entity = {"type": etype, "name": name, "description": desc}
            f.write(json.dumps(entity, ensure_ascii=False) + "\n")


def test_load_entity_names_with_valid_file(monkeypatch):
    """Test loading entity names from a valid entities.jsonl file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "facets" / "test" / "entities.jsonl"
        write_entities_jsonl(
            entities_path,
            [
                ("Person", "John Smith", "A software engineer at Google"),
                ("Company", "Acme Corp", "Technology company based in SF"),
                ("Project", "Project X", "Secret internal project"),
                ("Tool", "Hammer", "For hitting things"),
                ("Person", "Jane Doe", "Product manager at Meta"),
                ("Company", "Widget Inc", "Manufacturing company"),
            ],
        )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names()
        assert (
            result == "John Smith; Acme Corp; Project X; Hammer; Jane Doe; Widget Inc"
        )

        # Check that names are extracted without duplicates
        names = result.split("; ")
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
        entities_path = Path(tmpdir) / "facets" / "test" / "entities.jsonl"
        entities_path.parent.mkdir(parents=True, exist_ok=True)
        entities_path.write_text("")

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names()
        assert result is None


def test_load_entity_names_no_valid_entries(monkeypatch):
    """Test file with no parseable entity lines returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "facets" / "test" / "entities.jsonl"
        entities_path.parent.mkdir(parents=True, exist_ok=True)
        # Write malformed JSON
        entities_path.write_text(
            """
# Header comment
Some random text
Not valid JSON
"""
        )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names()
        assert result is None


def test_load_entity_names_with_duplicates(monkeypatch):
    """Test that duplicate names are filtered out."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "facets" / "test" / "entities.jsonl"
        write_entities_jsonl(
            entities_path,
            [
                ("Person", "John Smith", "Engineer"),
                ("Company", "Acme Corp", "Tech company"),
                ("Person", "John Smith", "Also an engineer"),
                ("Company", "Acme Corp", "Still a tech company"),
            ],
        )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names()
        assert result == "John Smith; Acme Corp"

        names = result.split("; ")
        assert len(names) == 2


def test_load_entity_names_handles_special_characters(monkeypatch):
    """Test that names with special characters are handled correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "facets" / "test" / "entities.jsonl"
        write_entities_jsonl(
            entities_path,
            [
                ("Person", "Jean-Pierre O'Malley", "Engineer"),
                ("Company", "AT&T", "Telecom company"),
                ("Project", "C++ Compiler", "Development tool"),
                ("Tool", "Node.js", "JavaScript runtime"),
            ],
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
        entities_path = Path(tmpdir) / "facets" / "test" / "entities.jsonl"
        write_entities_jsonl(
            entities_path,
            [("Person", "Test User", "A test person")],
        )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)

        # Should use env var
        result = load_entity_names()
        assert result == "Test User"


def test_load_entity_names_uses_default_path_when_env_var_unset(monkeypatch):
    """Test that missing JOURNAL_PATH falls back to platform default."""
    # Ensure JOURNAL_PATH is not set
    monkeypatch.delenv("JOURNAL_PATH", raising=False)
    # Mock load_dotenv to prevent it from loading a .env file
    monkeypatch.setattr("think.utils.load_dotenv", lambda: None)

    # With get_journal() fallback, this will use the platform default path
    # The result depends on whether entities exist there
    result = load_entity_names()
    # Result should be either None (no entities) or a string (entities exist)
    assert result is None or isinstance(result, str)


def test_load_entity_names_spoken_mode(monkeypatch):
    """Test spoken mode returns shortened forms with uniform processing for all types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "facets" / "test" / "entities.jsonl"
        write_entities_jsonl(
            entities_path,
            [
                ("Person", "Jeremie Miller (Jer)", "Software engineer"),
                ("Person", "Jane Elizabeth Doe", "Product manager"),
                ("Company", "Acme Corporation (ACME)", "Tech company"),
                ("Company", "Widget Inc", "Manufacturing company"),
                ("Company", "Google", "Search engine"),
                ("Project", "solstone Project (SUN)", "AI journaling"),
                ("Project", "Project X", "Secret project"),
                ("Tool", "Hammer", "For hitting things"),
                ("Tool", "Docker", "Container runtime"),
            ],
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

        # Project: "solstone Project (SUN)" -> ["solstone", "SUN"] (uniform processing)
        assert "solstone" in result  # First word
        assert "SUN" in result  # From parens

        # Project: "Project X" (no parens) -> ["Project"] (first word only)
        assert "Project" in result

        # Tools are now included (uniform processing for all types)
        assert "Hammer" in result
        assert "Docker" in result


def test_load_entity_names_spoken_mode_with_tools(monkeypatch):
    """Test spoken mode includes tools with uniform processing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "facets" / "test" / "entities.jsonl"
        write_entities_jsonl(
            entities_path,
            [
                ("Tool", "Hammer", "For hitting things"),
                ("Tool", "Docker", "Container runtime"),
            ],
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
        entities_path = Path(tmpdir) / "facets" / "test" / "entities.jsonl"
        write_entities_jsonl(
            entities_path,
            [
                ("Person", "John Smith", "Engineer"),
                ("Person", "John Doe", "Manager"),
                ("Company", "Acme Corp", "Tech"),
                ("Company", "Acme Industries", "Manufacturing"),
            ],
        )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names(spoken=True)

        # Should have only one "John" and one "Acme" even though there are two of each
        assert result.count("John") == 1
        assert result.count("Acme") == 1


def test_load_entity_names_uniform_processing(monkeypatch):
    """Test that uniform processing works correctly for all entity types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "facets" / "test" / "entities.jsonl"
        write_entities_jsonl(
            entities_path,
            [
                ("Person", "Ryan Reed (R2)", "Software developer"),
                (
                    "Company",
                    "Federal Aviation Administration (FAA)",
                    "Government agency",
                ),
                ("Project", "Backend API (API)", "Core service"),
                ("Tool", "pytest", "Testing framework"),
                ("Location", "New York City (NYC)", "Metropolitan area"),
            ],
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


def test_load_entity_names_with_aka_field(monkeypatch):
    """Test that aka field values are included in spoken mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "facets" / "test" / "entities.jsonl"
        entities_path.parent.mkdir(parents=True, exist_ok=True)

        # Write entities with aka fields using manual JSON
        with open(entities_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "Person",
                        "name": "Alice Johnson",
                        "description": "Lead engineer",
                        "aka": ["Ali", "AJ"],
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "type": "Company",
                        "name": "PostgreSQL",
                        "description": "Database system",
                        "aka": ["Postgres", "PG"],
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "type": "Tool",
                        "name": "Docker Container (Docker)",
                        "description": "Container runtime",
                        "aka": ["Dock"],
                    }
                )
                + "\n"
            )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names(spoken=True)

        assert isinstance(result, list)

        # Main name: "Alice Johnson" -> ["Alice"]
        assert "Alice" in result
        # aka entries: ["Ali", "AJ"]
        assert "Ali" in result
        assert "AJ" in result

        # Main name: "PostgreSQL" -> ["PostgreSQL"]
        assert "PostgreSQL" in result
        # aka entries: ["Postgres", "PG"]
        assert "Postgres" in result
        assert "PG" in result

        # Main name: "Docker Container (Docker)" -> ["Docker", "Docker"]
        # aka entries: ["Dock"]
        assert "Docker" in result
        assert "Dock" in result
        # Should be deduplicated - only one "Docker"
        assert result.count("Docker") == 1


def test_load_entity_names_aka_with_parens(monkeypatch):
    """Test that aka entries with parentheses are processed correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "facets" / "test" / "entities.jsonl"
        entities_path.parent.mkdir(parents=True, exist_ok=True)

        with open(entities_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "Person",
                        "name": "Robert Smith",
                        "description": "Manager",
                        "aka": ["Bob Smith (Bobby)", "Rob"],
                    }
                )
                + "\n"
            )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names(spoken=True)

        assert isinstance(result, list)

        # Main name: "Robert Smith" -> ["Robert"]
        assert "Robert" in result

        # aka entry: "Bob Smith (Bobby)" -> ["Bob", "Bobby"]
        assert "Bob" in result
        assert "Bobby" in result

        # aka entry: "Rob" -> ["Rob"]
        assert "Rob" in result


def test_load_entity_names_aka_deduplication(monkeypatch):
    """Test that aka values are deduplicated with main names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "facets" / "test" / "entities.jsonl"
        entities_path.parent.mkdir(parents=True, exist_ok=True)

        with open(entities_path, "w", encoding="utf-8") as f:
            # First entity has "John" in aka
            f.write(
                json.dumps(
                    {
                        "type": "Person",
                        "name": "Alice",
                        "description": "Person 1",
                        "aka": ["John"],
                    }
                )
                + "\n"
            )
            # Second entity has "John" as main name
            f.write(
                json.dumps(
                    {"type": "Person", "name": "John Smith", "description": "Person 2"}
                )
                + "\n"
            )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names(spoken=True)

        # Should have only one "John" even though it appears in aka and as main name
        assert result.count("John") == 1
        assert "Alice" in result


def test_load_entity_names_non_spoken_with_aka(monkeypatch):
    """Test non-spoken mode includes aka values in parentheses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entities_path = Path(tmpdir) / "facets" / "test" / "entities.jsonl"
        entities_path.parent.mkdir(parents=True, exist_ok=True)

        with open(entities_path, "w", encoding="utf-8") as f:
            # Entity with aka values
            f.write(
                json.dumps(
                    {
                        "type": "Person",
                        "name": "Alice Johnson",
                        "description": "Lead engineer",
                        "aka": ["Ali", "AJ"],
                    }
                )
                + "\n"
            )
            # Entity without aka
            f.write(
                json.dumps(
                    {
                        "type": "Company",
                        "name": "TechCorp",
                        "description": "Tech company",
                    }
                )
                + "\n"
            )
            # Entity with single aka
            f.write(
                json.dumps(
                    {
                        "type": "Tool",
                        "name": "PostgreSQL",
                        "description": "Database",
                        "aka": ["Postgres", "PG"],
                    }
                )
                + "\n"
            )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names(spoken=False)

        # Should be semicolon-delimited with aka in parentheses
        assert result == "Alice Johnson (Ali, AJ); TechCorp; PostgreSQL (Postgres, PG)"


def test_segment_key_hhmmss_with_duration():
    """Test segment_key with HHMMSS_LEN format."""
    assert segment_key("143022_300") == "143022_300"
    assert segment_key("095604_303") == "095604_303"
    assert segment_key("120000_3600") == "120000_3600"
    assert segment_key("000000_1") == "000000_1"


def test_segment_key_hhmmss_len_with_suffix():
    """Test segment_key with HHMMSS_LEN_suffix format."""
    assert segment_key("143022_300_audio") == "143022_300"
    assert segment_key("095604_303_screen") == "095604_303"
    assert segment_key("120000_3600_recording") == "120000_3600"
    assert segment_key("000000_1_mic_sys") == "000000_1"


def test_segment_key_with_file_extension():
    """Test segment_key with various file extensions."""
    assert segment_key("143022_300_audio.flac") == "143022_300"
    assert segment_key("095604_303_screen.webm") == "095604_303"
    assert segment_key("143022_300.jsonl") == "143022_300"


def test_segment_key_in_path():
    """Test segment_key extraction from full paths."""
    assert segment_key("/journal/20250109/143022_300/audio.jsonl") == "143022_300"
    assert segment_key("/home/user/20250110/095604_303_screen.webm") == "095604_303"
    assert segment_key("20250110/143022_300_audio.flac") == "143022_300"


def test_segment_key_invalid_formats():
    """Test segment_key with invalid formats returns None."""
    assert segment_key("invalid") is None
    assert segment_key("12345") is None  # Too short
    assert segment_key("1234567") is None  # Too long
    assert segment_key("abcdef") is None  # Not digits
    assert segment_key("14:30:22") is None  # Wrong separator
    assert segment_key("") is None
    assert segment_key("_143022") is None
    # Legacy formats without duration now return None
    assert segment_key("143022") is None
    assert segment_key("143022_audio") is None
    assert segment_key("143022_screen") is None


def test_segment_key_edge_cases():
    """Test segment_key with edge cases."""
    # Multiple underscores in suffix
    assert segment_key("143022_300_mic_sys_audio") == "143022_300"
    # Segment key with non-word boundary prefix (should not match)
    assert segment_key("prefix_143022_300_suffix") is None
    # Segment key with space/path separator (word boundary - should match)
    assert segment_key("prefix/143022_300/suffix") == "143022_300"
    assert segment_key("prefix 143022_300 suffix") == "143022_300"
    # Multiple potential matches (should match first)
    assert segment_key("143022_300 and 150000_600") == "143022_300"


class TestSetupCliConfigEnv:
    """Tests for config env injection via setup_cli()."""

    @pytest.fixture
    def cli_env(self, monkeypatch, tmp_path):
        """Set up a journal with config and mock sys.argv for setup_cli tests.

        Returns a helper function to write config and run setup_cli.
        """
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["test"])

        def write_config_and_run(config: dict | None = None):
            """Write config to journal and run setup_cli."""
            if config is not None:
                config_dir = tmp_path / "config"
                config_dir.mkdir(exist_ok=True)
                config_file = config_dir / "journal.json"
                config_file.write_text(json.dumps(config))

            parser = argparse.ArgumentParser()
            setup_cli(parser)

        return write_config_and_run

    def test_config_env_injected_into_os_environ(self, monkeypatch, cli_env):
        """Test that config env values are injected into os.environ."""
        monkeypatch.delenv("TEST_API_KEY", raising=False)
        monkeypatch.delenv("ANOTHER_VAR", raising=False)

        cli_env(
            {
                "identity": {"name": "Test"},
                "env": {
                    "TEST_API_KEY": "from_config",
                    "ANOTHER_VAR": "also_from_config",
                },
            }
        )

        assert os.environ.get("TEST_API_KEY") == "from_config"
        assert os.environ.get("ANOTHER_VAR") == "also_from_config"

    def test_shell_env_takes_precedence_over_config(self, monkeypatch, cli_env):
        """Test that shell/dotenv values are not overridden by config."""
        monkeypatch.setenv("EXISTING_VAR", "from_shell")

        cli_env(
            {
                "identity": {"name": "Test"},
                "env": {"EXISTING_VAR": "from_config"},
            }
        )

        assert os.environ.get("EXISTING_VAR") == "from_shell"

    def test_empty_shell_env_allows_config_override(self, monkeypatch, cli_env):
        """Test that empty shell env values are overridden by config."""
        monkeypatch.setenv("EMPTY_VAR", "")

        cli_env(
            {
                "identity": {"name": "Test"},
                "env": {"EMPTY_VAR": "from_config"},
            }
        )

        assert os.environ.get("EMPTY_VAR") == "from_config"

    def test_missing_env_section_is_safe(self, cli_env):
        """Test that missing env section in config doesn't cause errors."""
        cli_env({"identity": {"name": "Test"}})

    def test_missing_config_file_is_safe(self, cli_env):
        """Test that missing config file doesn't cause errors."""
        cli_env(None)  # No config file

    def test_config_env_converts_non_string_values(self, monkeypatch, cli_env):
        """Test that non-string config values are converted to strings."""
        monkeypatch.delenv("INT_VAR", raising=False)
        monkeypatch.delenv("BOOL_VAR", raising=False)

        cli_env(
            {
                "identity": {"name": "Test"},
                "env": {
                    "INT_VAR": 42,
                    "BOOL_VAR": True,
                },
            }
        )

        assert os.environ.get("INT_VAR") == "42"
        assert os.environ.get("BOOL_VAR") == "True"
