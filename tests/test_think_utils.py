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
from think.utils import (
    _merge_instructions_config,
    compose_instructions,
    segment_key,
    setup_cli,
)


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


class TestMergeInstructionsConfig:
    """Tests for _merge_instructions_config helper."""

    def test_returns_defaults_when_no_overrides(self):
        """Test that defaults are returned when overrides is None."""
        defaults = {"system": "journal", "facets": "short"}
        result = _merge_instructions_config(defaults, None)
        assert result == defaults
        # Should be a copy, not the same object
        assert result is not defaults

    def test_returns_defaults_when_empty_overrides(self):
        """Test that defaults are returned when overrides is empty dict."""
        defaults = {"system": "journal", "facets": "short"}
        result = _merge_instructions_config(defaults, {})
        assert result == defaults

    def test_overrides_system_key(self):
        """Test that system key can be overridden."""
        defaults = {"system": "journal", "facets": "short"}
        overrides = {"system": "custom_prompt"}
        result = _merge_instructions_config(defaults, overrides)
        assert result["system"] == "custom_prompt"
        assert result["facets"] == "short"

    def test_overrides_facets_key(self):
        """Test that facets key can be overridden."""
        defaults = {"system": "journal", "facets": "short"}
        overrides = {"facets": "detailed"}
        result = _merge_instructions_config(defaults, overrides)
        assert result["system"] == "journal"
        assert result["facets"] == "detailed"

    def test_merges_sources_dict(self):
        """Test that sources dict is merged, not replaced."""
        defaults = {
            "system": "journal",
            "sources": {"audio": True, "screen": True, "agents": False},
        }
        overrides = {"sources": {"screen": False}}
        result = _merge_instructions_config(defaults, overrides)
        assert result["sources"]["audio"] is True  # Preserved from defaults
        assert result["sources"]["screen"] is False  # Overridden
        assert result["sources"]["agents"] is False  # Preserved from defaults

    def test_ignores_unknown_keys(self):
        """Test that unknown keys in overrides are ignored."""
        defaults = {"system": "journal", "facets": "short"}
        overrides = {"unknown_key": "value", "another": 123}
        result = _merge_instructions_config(defaults, overrides)
        assert "unknown_key" not in result
        assert "another" not in result


class TestComposeInstructions:
    """Tests for compose_instructions function."""

    def test_default_system_instruction_is_journal(self, monkeypatch, tmp_path):
        """Test that default system instruction loads from journal.md."""
        # Create journal.md in think/ directory
        think_dir = tmp_path / "think"
        think_dir.mkdir()
        journal_txt = think_dir / "journal.md"
        journal_txt.write_text("Test system instruction content")

        # Mock the think module's parent to use our temp dir
        import think.utils

        original_file = think.utils.__file__
        monkeypatch.setattr(think.utils, "__file__", str(think_dir / "utils.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions()

        # Restore
        monkeypatch.setattr(think.utils, "__file__", original_file)

        assert "system_instruction" in result
        assert result["system_prompt_name"] == "journal"

    def test_custom_system_instruction(self, monkeypatch, tmp_path):
        """Test that custom system prompt can be loaded."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()
        custom_txt = think_dir / "custom.md"
        custom_txt.write_text("Custom system instruction")

        import think.utils

        monkeypatch.setattr(think.utils, "__file__", str(think_dir / "utils.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions(
            config_overrides={"system": "custom"},
        )

        assert result["system_prompt_name"] == "custom"
        assert "Custom system instruction" in result["system_instruction"]

    def test_user_instruction_loaded_when_provided(self, monkeypatch, tmp_path):
        """Test that user instruction is loaded when user_prompt is provided."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()
        journal_txt = think_dir / "journal.md"
        journal_txt.write_text("System instruction")
        user_txt = think_dir / "default.md"
        user_txt.write_text("User instruction content")

        import think.utils

        monkeypatch.setattr(think.utils, "__file__", str(think_dir / "utils.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions(user_prompt="default")

        assert result["user_instruction"] == "User instruction content"

    def test_user_instruction_none_when_not_provided(self, monkeypatch, tmp_path):
        """Test that user instruction is None when user_prompt is not provided."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()
        journal_txt = think_dir / "journal.md"
        journal_txt.write_text("System instruction")

        import think.utils

        monkeypatch.setattr(think.utils, "__file__", str(think_dir / "utils.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions()

        assert result["user_instruction"] is None

    def test_facets_none_excludes_facets_from_context(self, monkeypatch, tmp_path):
        """Test that facets='none' excludes facet info from extra_context."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()
        journal_txt = think_dir / "journal.md"
        journal_txt.write_text("System instruction")

        import think.utils

        monkeypatch.setattr(think.utils, "__file__", str(think_dir / "utils.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions(
            include_datetime=False,
            config_overrides={"facets": "none"},
        )

        # With no datetime and no facets, extra_context should be empty/None
        assert result["extra_context"] is None or result["extra_context"] == ""

    def test_include_datetime_false_excludes_time(self, monkeypatch, tmp_path):
        """Test that include_datetime=False excludes time from context."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()
        journal_txt = think_dir / "journal.md"
        journal_txt.write_text("System instruction")

        import think.utils

        monkeypatch.setattr(think.utils, "__file__", str(think_dir / "utils.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions(
            include_datetime=False,
            config_overrides={"facets": "none"},
        )

        extra = result.get("extra_context") or ""
        assert "Current Date and Time" not in extra

    def test_include_datetime_true_includes_time(self, monkeypatch, tmp_path):
        """Test that include_datetime=True includes time in context."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()
        journal_txt = think_dir / "journal.md"
        journal_txt.write_text("System instruction")

        import think.utils

        monkeypatch.setattr(think.utils, "__file__", str(think_dir / "utils.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions(
            include_datetime=True,
            config_overrides={"facets": "none"},
        )

        assert "Current Date and Time" in result["extra_context"]

    def test_sources_returned_from_defaults(self, monkeypatch, tmp_path):
        """Test that sources config is returned with defaults."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()
        journal_txt = think_dir / "journal.md"
        journal_txt.write_text("System instruction")

        import think.utils

        monkeypatch.setattr(think.utils, "__file__", str(think_dir / "utils.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions()

        assert "sources" in result
        assert result["sources"]["audio"] is True
        assert result["sources"]["screen"] is True
        assert result["sources"]["agents"] is False

    def test_sources_can_be_overridden(self, monkeypatch, tmp_path):
        """Test that sources config can be overridden."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()
        journal_txt = think_dir / "journal.md"
        journal_txt.write_text("System instruction")

        import think.utils

        monkeypatch.setattr(think.utils, "__file__", str(think_dir / "utils.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions(
            config_overrides={
                "sources": {"audio": False, "agents": True},
            },
        )

        assert result["sources"]["audio"] is False
        assert result["sources"]["screen"] is True  # Default preserved
        assert result["sources"]["agents"] is True  # Overridden


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


# =============================================================================
# source_is_enabled / source_is_required / get_agent_filter tests
# =============================================================================


def test_source_is_enabled_bool():
    """Test source_is_enabled with bool values."""
    from think.utils import source_is_enabled

    assert source_is_enabled(True) is True
    assert source_is_enabled(False) is False


def test_source_is_enabled_required_string():
    """Test source_is_enabled with 'required' string."""
    from think.utils import source_is_enabled

    assert source_is_enabled("required") is True


def test_source_is_enabled_dict():
    """Test source_is_enabled with dict values for agents source."""
    from think.utils import source_is_enabled

    # Dict with at least one True value -> enabled
    assert source_is_enabled({"entities": True, "meetings": False}) is True

    # Dict with at least one "required" value -> enabled
    assert source_is_enabled({"entities": "required", "meetings": False}) is True

    # Dict with all False values -> disabled
    assert source_is_enabled({"entities": False, "meetings": False}) is False

    # Empty dict -> disabled
    assert source_is_enabled({}) is False


def test_source_is_required_bool():
    """Test source_is_required with bool values."""
    from think.utils import source_is_required

    assert source_is_required(True) is False
    assert source_is_required(False) is False


def test_source_is_required_string():
    """Test source_is_required with 'required' string."""
    from think.utils import source_is_required

    assert source_is_required("required") is True


def test_source_is_required_dict():
    """Test source_is_required with dict values."""
    from think.utils import source_is_required

    # Dict with at least one "required" value -> required
    assert source_is_required({"entities": "required", "meetings": False}) is True

    # Dict with no "required" values -> not required
    assert source_is_required({"entities": True, "meetings": False}) is False

    # Empty dict -> not required
    assert source_is_required({}) is False


def test_get_agent_filter_bool():
    """Test get_agent_filter with bool values."""
    from think.utils import get_agent_filter

    # True -> None (all agents)
    assert get_agent_filter(True) is None

    # False -> empty dict (no agents)
    assert get_agent_filter(False) == {}


def test_get_agent_filter_required_string():
    """Test get_agent_filter with 'required' string."""
    from think.utils import get_agent_filter

    # "required" -> None (all agents, required)
    assert get_agent_filter("required") is None


def test_get_agent_filter_dict():
    """Test get_agent_filter with dict values."""
    from think.utils import get_agent_filter

    # Dict -> returned as-is for filtering
    filter_dict = {"entities": True, "meetings": "required", "flow": False}
    assert get_agent_filter(filter_dict) == filter_dict

    # Empty dict -> empty dict (no agents)
    assert get_agent_filter({}) == {}
