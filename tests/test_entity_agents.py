# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for entity detection and review agent configurations."""

import os

import pytest

from think.muse import get_agent


@pytest.fixture
def fixture_journal():
    """Set JOURNAL_PATH to fixtures/journal for testing."""
    os.environ["JOURNAL_PATH"] = "fixtures/journal"
    yield
    # No cleanup needed - just testing reads


def test_entities_agent_config(fixture_journal):
    """Test detection agent configuration loads correctly."""
    # Entity agents are in apps/entities/muse/ so use app-qualified name
    config = get_agent("entities:entities")

    # Verify required fields
    assert config["name"] == "entities:entities"
    assert "system_instruction" in config
    assert "user_instruction" in config
    assert len(config["system_instruction"]) > 0
    assert len(config["user_instruction"]) > 0

    # Verify JSON metadata fields from entities.json
    assert config.get("title") == "Entity Detector"
    assert config.get("schedule") == "daily"
    assert config.get("priority") == 55
    assert config.get("multi_facet") is True


def test_entities_review_agent_config(fixture_journal):
    """Test review agent configuration loads correctly."""
    # Entity agents are in apps/entities/muse/ so use app-qualified name
    config = get_agent("entities:entities_review")

    # Verify required fields
    assert config["name"] == "entities:entities_review"
    assert "system_instruction" in config
    assert "user_instruction" in config
    assert len(config["system_instruction"]) > 0
    assert len(config["user_instruction"]) > 0

    # Verify JSON metadata fields from entities_review.json
    assert config.get("title") == "Entity Reviewer"
    assert config.get("schedule") == "daily"
    assert config.get("priority") == 56
    assert config.get("multi_facet") is True


def test_entities_agent_instruction_content(fixture_journal):
    """Test detection agent instruction contains expected sections."""
    config = get_agent("entities:entities")
    prompt = config["user_instruction"]

    # Check for key sections in the agent prompt
    assert "Core Mission" in prompt
    assert "sol call entities detect" in prompt
    assert "sol call entities list" in prompt
    assert "Knowledge Graphs" in prompt or "knowledge_graph" in prompt
    assert "day-specific context" in prompt.lower()


def test_entities_review_agent_instruction_content(fixture_journal):
    """Test review agent instruction contains expected sections."""
    config = get_agent("entities:entities_review")
    prompt = config["user_instruction"]

    # Check for key sections in the agent prompt
    assert "Core Mission" in prompt
    assert "sol call entities attach" in prompt
    assert "sol call entities list" in prompt
    assert "3+" in prompt or "promotion" in prompt.lower()
    assert "description" in prompt.lower()


def test_agent_context_includes_entities_by_facet(fixture_journal):
    """Test that agent context includes entities grouped by facet."""
    config = get_agent("entities:entities")

    # extra_context should contain facet summaries with entities
    extra_context = config.get("extra_context", "")
    assert "Available Facets" in extra_context

    # Should include facet names in backtick format
    assert "`test-facet`" in extra_context or "`full-featured`" in extra_context

    # Should include entities from fixture facets
    # fixtures/journal/facets/ contains various entities
    assert "Entities" in extra_context

    # Check for some known entities from the fixtures
    assert (
        "John Smith" in extra_context
        or "Jane Doe" in extra_context
        or "Acme Corp" in extra_context
    )


def test_agent_context_with_facet_focus(fixture_journal):
    """Test that get_agent with facet parameter uses focused single-facet context."""
    config = get_agent("default", facet="full-featured")

    extra_context = config.get("extra_context", "")

    # Should have Facet Focus section instead of Available Facets
    assert "## Facet Focus" in extra_context
    assert "Available Facets" not in extra_context

    # Should include the focused facet's details
    assert "Full Featured Facet" in extra_context
    assert "A facet for testing all features" in extra_context

    # Should include entity details from the focused facet (inline format)
    assert "**Entities**:" in extra_context or "Entities:" in extra_context
    assert "Entity 1" in extra_context or "First test entity" in extra_context


def test_agent_priority_ordering(fixture_journal):
    """Test that entity agents have correct priority ordering."""
    detection_config = get_agent("entities:entities")
    review_config = get_agent("entities:entities_review")

    detection_priority = detection_config["priority"]
    review_priority = review_config["priority"]

    # Review should run after detection
    assert review_priority > detection_priority
    assert detection_priority == 55
    assert review_priority == 56
