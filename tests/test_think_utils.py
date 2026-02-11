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


def setup_entities_new_structure(
    journal_path: Path,
    facet: str,
    entities: list[tuple[str, str, str]] | list[dict],
):
    """Helper to set up entities using the new structure for tests.

    Creates both journal-level entity files and facet relationship files.

    Args:
        journal_path: Path to journal root
        facet: Facet name (e.g., "test")
        entities: Either list of (type, name, desc) tuples or list of entity dicts
    """
    from slugify import slugify

    for item in entities:
        if isinstance(item, dict):
            etype = item.get("type", "")
            name = item.get("name", "")
            desc = item.get("description", "")
            aka = item.get("aka", [])
        else:
            etype, name, desc = item
            aka = []

        entity_id = slugify(name, separator="_")
        if not entity_id:
            continue

        # Create journal-level entity
        journal_entity_dir = journal_path / "entities" / entity_id
        journal_entity_dir.mkdir(parents=True, exist_ok=True)
        journal_entity = {"id": entity_id, "name": name, "type": etype}
        if aka:
            journal_entity["aka"] = aka
        with open(journal_entity_dir / "entity.json", "w", encoding="utf-8") as f:
            json.dump(journal_entity, f)

        # Create facet relationship
        facet_entity_dir = journal_path / "facets" / facet / "entities" / entity_id
        facet_entity_dir.mkdir(parents=True, exist_ok=True)
        relationship = {"entity_id": entity_id, "description": desc}
        with open(facet_entity_dir / "entity.json", "w", encoding="utf-8") as f:
            json.dump(relationship, f)


def test_load_entity_names_with_valid_file(monkeypatch):
    """Test loading entity names from entities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_entities_new_structure(
            Path(tmpdir),
            "test",
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


def test_load_entity_names_empty_facet(monkeypatch):
    """Test that empty facet returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create facet directory but no entities
        facet_dir = Path(tmpdir) / "facets" / "test"
        facet_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names()
        assert result is None


def test_load_entity_names_no_valid_entries(monkeypatch):
    """Test empty entities directory returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create entities directory but no entity subdirectories
        entities_dir = Path(tmpdir) / "facets" / "test" / "entities"
        entities_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names()
        assert result is None


def test_load_entity_names_with_duplicates(monkeypatch):
    """Test that duplicate names are filtered out (by entity id)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # With new structure, same entity_id means same entity
        # Can't have true duplicates - just test two entities
        setup_entities_new_structure(
            Path(tmpdir),
            "test",
            [
                ("Person", "John Smith", "Engineer"),
                ("Company", "Acme Corp", "Tech company"),
            ],
        )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names()

        names = result.split("; ")
        assert len(names) == 2
        assert "John Smith" in names
        assert "Acme Corp" in names


def test_load_entity_names_handles_special_characters(monkeypatch):
    """Test that names with special characters are handled correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_entities_new_structure(
            Path(tmpdir),
            "test",
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
        setup_entities_new_structure(
            Path(tmpdir),
            "test",
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
        setup_entities_new_structure(
            Path(tmpdir),
            "test",
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
        setup_entities_new_structure(
            Path(tmpdir),
            "test",
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
        setup_entities_new_structure(
            Path(tmpdir),
            "test",
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
        setup_entities_new_structure(
            Path(tmpdir),
            "test",
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

        # "Ryan Reed (R2)" -> ["Ryan", "R2"] (digits allowed if has letter)
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
        setup_entities_new_structure(
            Path(tmpdir),
            "test",
            [
                {
                    "type": "Person",
                    "name": "Alice Johnson",
                    "description": "Lead engineer",
                    "aka": ["Ali", "AJ"],
                },
                {
                    "type": "Company",
                    "name": "PostgreSQL",
                    "description": "Database system",
                    "aka": ["Postgres", "PG"],
                },
                {
                    "type": "Tool",
                    "name": "Docker Container (Docker)",
                    "description": "Container runtime",
                    "aka": ["Dock"],
                },
            ],
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
        setup_entities_new_structure(
            Path(tmpdir),
            "test",
            [
                {
                    "type": "Person",
                    "name": "Robert Smith",
                    "description": "Manager",
                    "aka": ["Bob Smith (Bobby)", "Rob"],
                },
            ],
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
        setup_entities_new_structure(
            Path(tmpdir),
            "test",
            [
                # First entity has "John" in aka
                {
                    "type": "Person",
                    "name": "Alice",
                    "description": "Person 1",
                    "aka": ["John"],
                },
                # Second entity has "John" as main name
                {"type": "Person", "name": "John Smith", "description": "Person 2"},
            ],
        )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names(spoken=True)

        # Should have only one "John" even though it appears in aka and as main name
        assert result.count("John") == 1
        assert "Alice" in result


def test_load_entity_names_non_spoken_with_aka(monkeypatch):
    """Test non-spoken mode includes aka values in parentheses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_entities_new_structure(
            Path(tmpdir),
            "test",
            [
                # Entity with aka values
                {
                    "type": "Person",
                    "name": "Alice Johnson",
                    "description": "Lead engineer",
                    "aka": ["Ali", "AJ"],
                },
                # Entity without aka
                {
                    "type": "Company",
                    "name": "TechCorp",
                    "description": "Tech company",
                },
                # Entity with multiple aka
                {
                    "type": "Tool",
                    "name": "PostgreSQL",
                    "description": "Database",
                    "aka": ["Postgres", "PG"],
                },
            ],
        )

        monkeypatch.setenv("JOURNAL_PATH", tmpdir)
        result = load_entity_names(spoken=False)

        # Check all entities are present with their aka
        assert "Alice Johnson (Ali, AJ)" in result
        assert "TechCorp" in result
        assert "PostgreSQL (Postgres, PG)" in result


class TestTruncatedEcho:
    """Tests for truncated_echo output helper."""

    def test_under_limit_passes_through(self, capsys):
        """Text under the limit is printed without truncation."""
        from think.utils import truncated_echo

        truncated_echo("hello world", max_bytes=1024)
        captured = capsys.readouterr()
        assert captured.out == "hello world\n"
        assert captured.err == ""

    def test_over_limit_truncates_and_warns(self, capsys):
        """Text over the limit is truncated with stderr warning."""
        from think.utils import truncated_echo

        text = "a" * 200
        truncated_echo(text, max_bytes=50)
        captured = capsys.readouterr()
        # stdout should have exactly 50 bytes of content + newline
        assert captured.out == "a" * 50 + "\n"
        assert "truncated" in captured.err
        assert "200" in captured.err
        assert "50" in captured.err

    def test_zero_means_unlimited(self, capsys):
        """max_bytes=0 disables truncation."""
        from think.utils import truncated_echo

        text = "b" * 100_000
        truncated_echo(text, max_bytes=0)
        captured = capsys.readouterr()
        assert captured.out == text + "\n"
        assert captured.err == ""

    def test_utf8_boundary_safe(self, capsys):
        """Truncation at a multibyte UTF-8 boundary drops partial chars."""
        from think.utils import truncated_echo

        # Each emoji is 4 bytes in UTF-8
        text = "\U0001f600" * 10  # 40 bytes total
        truncated_echo(text, max_bytes=6)  # mid-second emoji
        captured = capsys.readouterr()
        # Should get only the first complete emoji (4 bytes) since bytes 5-6
        # form an incomplete character that gets dropped by errors="ignore"
        assert captured.out == "\U0001f600\n"
        assert "truncated" in captured.err

    def test_exact_limit_no_truncation(self, capsys):
        """Text exactly at the byte limit is not truncated."""
        from think.utils import truncated_echo

        text = "x" * 100
        truncated_echo(text, max_bytes=100)
        captured = capsys.readouterr()
        assert captured.out == text + "\n"
        assert captured.err == ""


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


class TestPortDiscovery:
    """Tests for service port discovery utilities."""

    def test_find_available_port_returns_valid_port(self):
        """Test that find_available_port returns a valid port number."""
        from think.utils import find_available_port

        port = find_available_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535  # User-space port range

    def test_find_available_port_different_each_call(self):
        """Test that multiple calls can return different ports."""
        from think.utils import find_available_port

        # Get multiple ports - they may or may not be unique, but should all be valid
        ports = [find_available_port() for _ in range(3)]
        for port in ports:
            assert isinstance(port, int)
            assert 1024 <= port <= 65535

    def test_write_and_read_service_port(self, monkeypatch, tmp_path):
        """Test writing and reading a service port file."""
        from think.utils import read_service_port, write_service_port

        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        # Write port
        write_service_port("test_service", 12345)

        # Read port back
        port = read_service_port("test_service")
        assert port == 12345

        # Verify file exists in correct location
        port_file = tmp_path / "health" / "test_service.port"
        assert port_file.exists()
        assert port_file.read_text() == "12345"

    def test_read_service_port_missing_file(self, monkeypatch, tmp_path):
        """Test that reading missing port file returns None."""
        from think.utils import read_service_port

        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        port = read_service_port("nonexistent")
        assert port is None

    def test_read_service_port_invalid_content(self, monkeypatch, tmp_path):
        """Test that reading invalid port file content returns None."""
        from think.utils import read_service_port

        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        # Create port file with invalid content
        health_dir = tmp_path / "health"
        health_dir.mkdir()
        port_file = health_dir / "bad_service.port"
        port_file.write_text("not a number")

        port = read_service_port("bad_service")
        assert port is None

    def test_write_service_port_creates_health_dir(self, monkeypatch, tmp_path):
        """Test that write_service_port creates health directory if needed."""
        from think.utils import write_service_port

        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        # Health dir doesn't exist yet
        health_dir = tmp_path / "health"
        assert not health_dir.exists()

        write_service_port("new_service", 9999)

        # Now it should exist
        assert health_dir.exists()
        assert (health_dir / "new_service.port").read_text() == "9999"
