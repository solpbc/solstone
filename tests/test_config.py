# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for journal configuration utilities."""

import json
import os

import pytest

from think.utils import get_config


@pytest.fixture
def config_journal(tmp_path):
    """Create a temporary journal with config."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    config_data = {
        "identity": {
            "name": "Test User",
            "preferred": "Tester",
            "bio": "a software engineer and tester",
            "pronouns": {
                "subject": "they",
                "object": "them",
                "possessive": "their",
                "reflexive": "themselves",
            },
            "aliases": ["test", "tester"],
            "email_addresses": ["test@example.com"],
            "timezone": "America/New_York",
        }
    }

    config_file = config_dir / "journal.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)
        f.write("\n")

    return tmp_path


def test_get_config_default_structure(tmp_path, monkeypatch):
    """Test get_config returns default structure when file doesn't exist."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    config = get_config()

    assert "identity" in config
    assert config["identity"]["name"] == ""
    assert config["identity"]["preferred"] == ""
    assert config["identity"]["pronouns"] == {
        "subject": "",
        "object": "",
        "possessive": "",
        "reflexive": "",
    }
    assert config["identity"]["aliases"] == []
    assert config["identity"]["email_addresses"] == []
    assert config["identity"]["timezone"] == ""
    assert config["identity"]["bio"] == ""


def test_get_config_loads_existing(config_journal, monkeypatch):
    """Test get_config loads existing configuration."""
    monkeypatch.setenv("JOURNAL_PATH", str(config_journal))

    config = get_config()

    assert config["identity"]["name"] == "Test User"
    assert config["identity"]["preferred"] == "Tester"
    assert config["identity"]["pronouns"] == {
        "subject": "they",
        "object": "them",
        "possessive": "their",
        "reflexive": "themselves",
    }
    assert config["identity"]["aliases"] == ["test", "tester"]
    assert config["identity"]["email_addresses"] == ["test@example.com"]
    assert config["identity"]["timezone"] == "America/New_York"
    assert config["identity"]["bio"] == "a software engineer and tester"


def test_get_config_fills_missing_fields(tmp_path, monkeypatch):
    """Test get_config fills in missing identity fields."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Create config with partial identity data
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    partial_config = {
        "identity": {
            "name": "Partial User",
            # Missing other fields
        }
    }

    config_file = config_dir / "journal.json"
    with open(config_file, "w") as f:
        json.dump(partial_config, f)

    config = get_config()

    # Check that missing fields are filled with defaults
    assert config["identity"]["name"] == "Partial User"
    assert config["identity"]["preferred"] == ""
    assert config["identity"]["pronouns"] == {
        "subject": "",
        "object": "",
        "possessive": "",
        "reflexive": "",
    }
    assert config["identity"]["aliases"] == []
    assert config["identity"]["email_addresses"] == []
    assert config["identity"]["timezone"] == ""
    assert config["identity"]["bio"] == ""


def test_get_config_uses_default_when_journal_path_empty(monkeypatch, tmp_path):
    """Test get_config uses platform default when JOURNAL_PATH is empty."""
    # Set to empty string - should fall back to platform default
    monkeypatch.setenv("JOURNAL_PATH", "")

    # get_config should work (will use platform default journal path)
    config = get_config()
    # Returns default structure with empty identity
    assert "identity" in config
    assert config["identity"]["name"] == ""


def test_get_config_handles_invalid_json(tmp_path, monkeypatch):
    """Test get_config returns defaults when JSON is invalid."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Create config with invalid JSON
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    config_file = config_dir / "journal.json"
    with open(config_file, "w") as f:
        f.write("{ invalid json }")

    # Should return default structure and log warning
    config = get_config()

    assert "identity" in config
    assert config["identity"]["name"] == ""
    assert config["identity"]["pronouns"] == {
        "subject": "",
        "object": "",
        "possessive": "",
        "reflexive": "",
    }
    assert config["identity"]["bio"] == ""


def test_get_config_with_fixtures():
    """Test get_config with fixtures/journal path."""
    # Set JOURNAL_PATH to fixtures
    os.environ["JOURNAL_PATH"] = "fixtures/journal"

    config = get_config()

    # Should return default structure since fixtures doesn't have config yet
    assert "identity" in config
    assert isinstance(config["identity"]["name"], str)
    assert isinstance(config["identity"]["preferred"], str)
    assert isinstance(config["identity"]["pronouns"], dict)
    assert "subject" in config["identity"]["pronouns"]
    assert "object" in config["identity"]["pronouns"]
    assert "possessive" in config["identity"]["pronouns"]
    assert "reflexive" in config["identity"]["pronouns"]
    assert isinstance(config["identity"]["aliases"], list)
    assert isinstance(config["identity"]["email_addresses"], list)
    assert isinstance(config["identity"]["timezone"], str)
    assert isinstance(config["identity"]["bio"], str)
