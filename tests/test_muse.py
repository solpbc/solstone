# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.muse module.

Tests for muse prompt loading, configuration, and instruction composition.
"""

import pytest

from think.muse import (
    _merge_instructions_config,
    compose_instructions,
    get_agent_filter,
    source_is_enabled,
    source_is_required,
)

# =============================================================================
# _merge_instructions_config tests
# =============================================================================


def test_merge_instructions_config_empty_overrides():
    """Test that empty overrides returns defaults copy."""
    defaults = {"system": "journal", "facets": True, "sources": {"audio": False}}
    result = _merge_instructions_config(defaults, None)
    assert result == defaults
    assert result is not defaults  # Should be a copy


def test_merge_instructions_config_with_overrides():
    """Test that overrides are merged correctly."""
    defaults = {"system": "journal", "facets": True, "sources": {"audio": False}}
    overrides = {"system": "custom", "facets": False}
    result = _merge_instructions_config(defaults, overrides)
    assert result["system"] == "custom"
    assert result["facets"] is False
    assert result["sources"] == {"audio": False}  # Preserved


def test_merge_instructions_config_sources_merge():
    """Test that sources dict is merged, not replaced."""
    defaults = {"system": None, "sources": {"audio": False, "screen": False}}
    overrides = {"sources": {"audio": True}}
    result = _merge_instructions_config(defaults, overrides)
    assert result["sources"]["audio"] is True  # Overridden
    assert result["sources"]["screen"] is False  # Preserved from defaults


def test_merge_instructions_config_ignores_unknown_keys():
    """Test that unknown keys in overrides are ignored."""
    defaults = {"system": "journal", "facets": True}
    overrides = {"unknown_key": "value", "another": 123}
    result = _merge_instructions_config(defaults, overrides)
    assert "unknown_key" not in result
    assert "another" not in result


def test_merge_instructions_config_facets_override():
    """Test that facets key can be overridden with different values."""
    defaults = {"system": "journal", "facets": True}
    overrides = {"facets": "full"}
    result = _merge_instructions_config(defaults, overrides)
    assert result["system"] == "journal"
    assert result["facets"] == "full"


# =============================================================================
# compose_instructions tests
# =============================================================================


class TestComposeInstructions:
    """Tests for compose_instructions function."""

    def test_default_system_instruction_is_none(self, monkeypatch, tmp_path):
        """Test that default system instruction is empty (agents must opt-in)."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()

        import think.prompts

        monkeypatch.setattr(think.prompts, "__file__", str(think_dir / "prompts.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions()

        assert "system_instruction" in result
        assert result["system_instruction"] == ""
        assert result["system_prompt_name"] == ""

    def test_custom_system_instruction(self, monkeypatch, tmp_path):
        """Test that custom system prompt can be loaded."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()
        custom_txt = think_dir / "custom.md"
        custom_txt.write_text("Custom system instruction")

        import think.prompts

        monkeypatch.setattr(think.prompts, "__file__", str(think_dir / "prompts.py"))
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

        import think.muse
        import think.prompts

        # Monkeypatch both modules since compose_instructions uses muse.__file__ for
        # default user_prompt_dir, and load_prompt uses prompts.__file__ for defaults
        monkeypatch.setattr(think.prompts, "__file__", str(think_dir / "prompts.py"))
        monkeypatch.setattr(think.muse, "__file__", str(think_dir / "muse.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions(user_prompt="default")

        assert result["user_instruction"] == "User instruction content"

    def test_user_instruction_none_when_not_provided(self, monkeypatch, tmp_path):
        """Test that user instruction is None when user_prompt is not provided."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()
        journal_txt = think_dir / "journal.md"
        journal_txt.write_text("System instruction")

        import think.prompts

        monkeypatch.setattr(think.prompts, "__file__", str(think_dir / "prompts.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions()

        assert result["user_instruction"] is None

    def test_facets_none_excludes_facets_from_context(self, monkeypatch, tmp_path):
        """Test that facets='none' excludes facet info from extra_context."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()
        journal_txt = think_dir / "journal.md"
        journal_txt.write_text("System instruction")

        import think.prompts

        monkeypatch.setattr(think.prompts, "__file__", str(think_dir / "prompts.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions(
            config_overrides={"facets": False, "now": False, "day": False},
        )

        # With no datetime and no facets, extra_context should be empty/None
        assert result["extra_context"] is None or result["extra_context"] == ""

    def test_now_false_excludes_time(self, monkeypatch, tmp_path):
        """Test that now=False excludes current datetime from context."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()
        journal_txt = think_dir / "journal.md"
        journal_txt.write_text("System instruction")

        import think.prompts

        monkeypatch.setattr(think.prompts, "__file__", str(think_dir / "prompts.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions(
            config_overrides={"facets": False, "now": False},
        )

        extra = result.get("extra_context") or ""
        assert "Current Date and Time" not in extra

    def test_now_true_includes_time(self, monkeypatch, tmp_path):
        """Test that now=True includes current datetime in context."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()
        journal_txt = think_dir / "journal.md"
        journal_txt.write_text("System instruction")

        import think.prompts

        monkeypatch.setattr(think.prompts, "__file__", str(think_dir / "prompts.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions(
            config_overrides={"facets": False, "now": True},
        )

        assert "Current Date and Time" in result["extra_context"]

    def test_day_true_includes_analysis_day(self, monkeypatch, tmp_path):
        """Test that day=True includes analysis day in context."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()
        journal_txt = think_dir / "journal.md"
        journal_txt.write_text("System instruction")

        import think.prompts

        monkeypatch.setattr(think.prompts, "__file__", str(think_dir / "prompts.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions(
            analysis_day="20250115",
            config_overrides={"facets": False, "day": True},
        )

        extra = result.get("extra_context") or ""
        assert "Analysis Day" in extra
        assert "20250115" in extra

    def test_sources_returned_from_defaults(self, monkeypatch, tmp_path):
        """Test that sources config is returned with defaults (all false)."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()

        import think.prompts

        monkeypatch.setattr(think.prompts, "__file__", str(think_dir / "prompts.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions()

        assert "sources" in result
        assert result["sources"]["audio"] is False
        assert result["sources"]["screen"] is False
        assert result["sources"]["agents"] is False

    def test_sources_can_be_overridden(self, monkeypatch, tmp_path):
        """Test that sources config can be overridden."""
        think_dir = tmp_path / "think"
        think_dir.mkdir()

        import think.prompts

        monkeypatch.setattr(think.prompts, "__file__", str(think_dir / "prompts.py"))
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        result = compose_instructions(
            config_overrides={
                "sources": {"audio": True, "agents": True},
            },
        )

        assert result["sources"]["audio"] is True  # Overridden
        assert result["sources"]["screen"] is False  # Default preserved
        assert result["sources"]["agents"] is True  # Overridden


# =============================================================================
# source_is_enabled / source_is_required / get_agent_filter tests
# =============================================================================


def test_source_is_enabled_bool():
    """Test source_is_enabled with bool values."""
    assert source_is_enabled(True) is True
    assert source_is_enabled(False) is False


def test_source_is_enabled_required_string():
    """Test source_is_enabled with 'required' string."""
    assert source_is_enabled("required") is True


def test_source_is_enabled_dict():
    """Test source_is_enabled with dict values for agents source."""
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
    assert source_is_required(True) is False
    assert source_is_required(False) is False


def test_source_is_required_string():
    """Test source_is_required with 'required' string."""
    assert source_is_required("required") is True


def test_source_is_required_dict():
    """Test source_is_required with dict values."""
    # Dict with at least one "required" value -> required
    assert source_is_required({"entities": "required", "meetings": False}) is True

    # Dict with no "required" values -> not required
    assert source_is_required({"entities": True, "meetings": False}) is False

    # Empty dict -> not required
    assert source_is_required({}) is False


def test_get_agent_filter_bool():
    """Test get_agent_filter with bool values."""
    # True -> None (all agents)
    assert get_agent_filter(True) is None

    # False -> empty dict (no agents)
    assert get_agent_filter(False) == {}


def test_get_agent_filter_required_string():
    """Test get_agent_filter with 'required' string."""
    # "required" -> None (all agents, required)
    assert get_agent_filter("required") is None


def test_get_agent_filter_dict():
    """Test get_agent_filter with dict values."""
    # Dict -> returned as-is for filtering
    filter_dict = {"entities": True, "meetings": "required", "flow": False}
    assert get_agent_filter(filter_dict) == filter_dict

    # Empty dict -> empty dict (no agents)
    assert get_agent_filter({}) == {}
