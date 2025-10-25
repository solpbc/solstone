"""Tests for entity detection and review agent configurations."""

import os

import pytest

from think.utils import get_agent


@pytest.fixture
def fixture_journal():
    """Set JOURNAL_PATH to fixtures/journal for testing."""
    os.environ["JOURNAL_PATH"] = "fixtures/journal"
    yield
    # No cleanup needed - just testing reads


def test_entities_agent_config(fixture_journal):
    """Test detection agent configuration loads correctly."""
    config = get_agent("entities")

    # Verify required fields
    assert config["persona"] == "entities"
    assert "instruction" in config
    assert len(config["instruction"]) > 0

    # Verify JSON metadata fields from entities.json
    assert config.get("title") == "Entity Detector"
    assert config.get("schedule") == "daily"
    assert config.get("priority") == 25
    assert config.get("tools") == "journal, entities"
    assert config.get("multi_domain") is True


def test_entities_review_agent_config(fixture_journal):
    """Test review agent configuration loads correctly."""
    config = get_agent("entities_review")

    # Verify required fields
    assert config["persona"] == "entities_review"
    assert "instruction" in config
    assert len(config["instruction"]) > 0

    # Verify JSON metadata fields from entities_review.json
    assert config.get("title") == "Entity Reviewer"
    assert config.get("schedule") == "daily"
    assert config.get("priority") == 26
    assert config.get("tools") == "journal, entities"
    assert config.get("multi_domain") is True


def test_entities_agent_instruction_content(fixture_journal):
    """Test detection agent instruction contains expected sections."""
    config = get_agent("entities")
    instruction = config["instruction"]

    # Check for key sections in the prompt
    assert "Core Mission" in instruction
    assert "entity_detect" in instruction
    assert "entity_list" in instruction
    assert "Knowledge Graphs" in instruction or "knowledge_graph" in instruction
    assert "day-specific context" in instruction.lower()


def test_entities_review_agent_instruction_content(fixture_journal):
    """Test review agent instruction contains expected sections."""
    config = get_agent("entities_review")
    instruction = config["instruction"]

    # Check for key sections in the prompt
    assert "Core Mission" in instruction
    assert "entity_attach" in instruction
    assert "entity_list" in instruction
    assert "3+" in instruction or "promotion" in instruction.lower()
    assert "description" in instruction.lower()


def test_agent_context_includes_entities_by_domain(fixture_journal):
    """Test that agent context includes entities grouped by domain."""
    config = get_agent("entities")

    # extra_context should contain domain summaries with entities
    extra_context = config.get("extra_context", "")
    assert "Available Domains" in extra_context

    # Should include domain names in backtick format
    assert "`test-domain`" in extra_context or "`full-featured`" in extra_context

    # Should include entities from fixture domains
    # fixtures/journal/domains/ contains various entities
    assert "Entities" in extra_context

    # Check for some known entities from the fixtures
    assert (
        "John Smith" in extra_context
        or "Jane Doe" in extra_context
        or "Acme Corp" in extra_context
    )


def test_agent_priority_ordering(fixture_journal):
    """Test that entity agents have correct priority ordering."""
    detection_config = get_agent("entities")
    review_config = get_agent("entities_review")

    detection_priority = detection_config.get("priority", 50)
    review_priority = review_config.get("priority", 50)

    # Review should run after detection
    assert review_priority > detection_priority
    assert detection_priority == 25
    assert review_priority == 26
