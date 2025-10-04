"""Tests for template variable substitution in load_prompt."""

import json
import os

import pytest

from think.utils import _flatten_identity_to_template_vars, load_prompt


@pytest.fixture
def mock_journal_with_config(tmp_path):
    """Create a temporary journal with config and entities."""
    # Create config directory and journal.json
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    config = {
        "identity": {
            "name": "Test User",
            "preferred": "Testy",
            "pronouns": {
                "subject": "they",
                "object": "them",
                "possessive": "their",
                "reflexive": "themselves",
            },
            "aliases": ["test", "tester"],
            "email_addresses": ["test@example.com"],
            "timezone": "America/Los_Angeles",
            "entity": "Test User (Testy)",
            "bio": "a curious software engineer",
        }
    }

    with open(config_dir / "journal.json", "w") as f:
        json.dump(config, f)

    # Create entities.md
    entities_content = """- Person: Test User (Testy) - a curious software engineer interested in AI
- Company: Acme Corp - example company for testing
"""
    (tmp_path / "entities.md").write_text(entities_content)

    # Set JOURNAL_PATH for the test
    old_journal = os.environ.get("JOURNAL_PATH")
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    yield tmp_path

    # Restore original JOURNAL_PATH
    if old_journal:
        os.environ["JOURNAL_PATH"] = old_journal
    else:
        os.environ.pop("JOURNAL_PATH", None)


@pytest.fixture
def mock_prompt_dir(tmp_path):
    """Create a temporary directory with test prompt files."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    # Create a prompt with template variables
    template_prompt = """Hello $name, also known as $preferred!

You use $pronouns_subject/$pronouns_object/$pronouns_possessive/$pronouns_reflexive pronouns.

Capitalized: $Pronouns_subject will do it $Pronouns_reflexive.

Entity: $entity_name described as: $entity_value

Bio: $bio
"""
    (prompts_dir / "test_template.txt").write_text(template_prompt)

    # Create a prompt without template variables
    plain_prompt = "This is a plain prompt without any variables."
    (prompts_dir / "plain.txt").write_text(plain_prompt)

    return prompts_dir


def test_flatten_identity_basic_fields():
    """Test flattening of basic identity fields."""
    identity = {"name": "Alice Smith", "preferred": "Alice", "timezone": "UTC"}

    result = _flatten_identity_to_template_vars(identity)

    assert result["name"] == "Alice Smith"
    assert result["Name"] == "Alice smith"  # Capitalized
    assert result["preferred"] == "Alice"
    assert result["Preferred"] == "Alice"
    assert result["timezone"] == "UTC"


def test_flatten_identity_nested_pronouns():
    """Test flattening of nested pronoun fields."""
    identity = {
        "pronouns": {
            "subject": "she",
            "object": "her",
            "possessive": "her",
            "reflexive": "herself",
        }
    }

    result = _flatten_identity_to_template_vars(identity)

    assert result["pronouns_subject"] == "she"
    assert result["Pronouns_subject"] == "She"
    assert result["pronouns_object"] == "her"
    assert result["pronouns_possessive"] == "her"
    assert result["pronouns_reflexive"] == "herself"
    assert result["Pronouns_reflexive"] == "Herself"


def test_flatten_identity_with_entity(mock_journal_with_config):
    """Test entity name and value extraction."""
    from think.utils import get_config

    config = get_config()
    identity = config["identity"]

    result = _flatten_identity_to_template_vars(identity)

    assert result["entity_name"] == "Test User (Testy)"
    assert result["entity_value"] == "a curious software engineer interested in AI"
    assert (
        result["Entity_name"] == "Test User (Testy)"
    )  # Not capitalized for entity names


def test_load_prompt_with_substitution(mock_journal_with_config, mock_prompt_dir):
    """Test that load_prompt performs template substitution."""
    result = load_prompt("test_template", base_dir=mock_prompt_dir)

    # Check that variables were substituted
    assert "Test User" in result.text
    assert "Testy" in result.text
    assert "they/them/their/themselves" in result.text
    assert "They will do it Themselves" in result.text
    assert "Test User (Testy)" in result.text
    assert "a curious software engineer interested in AI" in result.text
    assert "a curious software engineer" in result.text  # bio

    # Ensure no template variables remain
    assert "$name" not in result.text
    assert "$pronouns_subject" not in result.text
    assert "$entity_value" not in result.text


def test_load_prompt_without_substitution(mock_journal_with_config, mock_prompt_dir):
    """Test that prompts without variables work normally."""
    result = load_prompt("plain", base_dir=mock_prompt_dir)

    assert result.text == "This is a plain prompt without any variables."


def test_load_prompt_missing_config_graceful(tmp_path, mock_prompt_dir):
    """Test that load_prompt works even without config (safe_substitute)."""
    # Point to a journal without config
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    result = load_prompt("test_template", base_dir=mock_prompt_dir)

    # When config exists but has empty values, safe_substitute replaces with empty strings
    # Entity variables without entity set should remain as $var (not in template_vars dict)
    assert "$entity_name" in result.text
    assert "$entity_value" in result.text
    # The prompt should still load without errors
    assert result.path.exists()
