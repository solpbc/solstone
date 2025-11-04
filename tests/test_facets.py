"""Tests for think.facets module."""

from pathlib import Path

import pytest

from think.facets import facet_summaries, facet_summary, get_facets

# Use the permanent fixtures in fixtures/journal/facets/
FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "journal"


def test_facet_summary_full(monkeypatch):
    """Test facet_summary with full metadata."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = facet_summary("full-featured")

    # Check title without emoji
    assert "# Full Featured Facet" in summary

    # Check description
    assert "**Description:** A facet for testing all features" in summary

    # Check color badge
    assert "![Color](#28a745)" in summary

    # Check entities section
    assert "## Entities" in summary
    assert "**Entity 1**: First test entity" in summary
    assert "**Entity 2**: Second test entity" in summary
    assert "**Entity 3**: Third test entity with description" in summary


def test_facet_summary_minimal(monkeypatch):
    """Test facet_summary with minimal metadata."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = facet_summary("minimal-facet")

    # Check title without emoji
    assert "# Minimal Facet" in summary

    # Should not have description, color, or entities
    assert "**Description:**" not in summary
    assert "![Color]" not in summary
    assert "## Entities" not in summary


def test_facet_summary_test_facet(monkeypatch):
    """Test facet_summary with the existing test-facet fixture."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = facet_summary("test-facet")

    # Check title without emoji
    assert "# Test Facet" in summary

    # Check description
    assert "**Description:** A test facet for validating functionality" in summary

    # Check color badge
    assert "![Color](#007bff)" in summary


def test_facet_summary_nonexistent(monkeypatch):
    """Test facet_summary with nonexistent facet."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    with pytest.raises(FileNotFoundError, match="Facet 'nonexistent' not found"):
        facet_summary("nonexistent")


def test_facet_summary_no_journal_path(monkeypatch):
    """Test facet_summary without JOURNAL_PATH set."""
    # Set to empty string to override any .env file
    monkeypatch.setenv("JOURNAL_PATH", "")

    with pytest.raises(RuntimeError, match="JOURNAL_PATH not set"):
        facet_summary("any-facet")


def test_facet_summary_missing_facet_json(monkeypatch):
    """Test facet_summary with missing facet.json."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    with pytest.raises(FileNotFoundError, match="facet.json not found"):
        facet_summary("broken-facet")


def test_facet_summary_empty_entities(monkeypatch):
    """Test facet_summary with empty entities file."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = facet_summary("empty-entities")

    # Should not include entities section if file is empty
    assert "## Entities" not in summary


def test_get_facets_with_entities(monkeypatch):
    """Test that get_facets() returns metadata and load_entity_names() works with facets."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    facets = get_facets()

    # Check test-facet exists
    assert "test-facet" in facets
    test_facet = facets["test-facet"]

    # Check basic metadata
    assert test_facet["title"] == "Test Facet"
    assert test_facet["emoji"] == "🧪"

    # Verify entities are NOT included in get_facets() anymore
    assert "entities" not in test_facet

    # Instead, verify entities can be loaded via load_entity_names()
    from think.entities import load_entity_names

    entity_names = load_entity_names(facet="test-facet")
    assert entity_names is not None

    # Check that specific entities are in the semicolon-delimited string
    assert "John Smith" in entity_names
    assert "Jane Doe" in entity_names
    assert "Bob Wilson" in entity_names
    assert "Acme Corp" in entity_names
    assert "Tech Solutions Inc" in entity_names
    assert "API Optimization" in entity_names
    assert "Dashboard Redesign" in entity_names
    assert "Visual Studio Code" in entity_names
    assert "Docker" in entity_names
    assert "PostgreSQL" in entity_names


def test_get_facets_empty_entities(monkeypatch):
    """Test get_facets() with facet that has no entities."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    facets = get_facets()

    # Check minimal-facet (should have no entities file)
    if "minimal-facet" in facets:
        minimal_facet = facets["minimal-facet"]
        # Entities are no longer included in get_facets()
        assert "entities" not in minimal_facet

        # Verify load_entity_names returns None for facets without entities.jsonl
        from think.entities import load_entity_names

        entity_names = load_entity_names(facet="minimal-facet")
        assert entity_names is None


def test_facet_summaries(monkeypatch):
    """Test facet_summaries() generates correct agent prompt format."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = facet_summaries()

    # Check header
    assert "## Available Facets" in summary

    # Check test-facet is included with backtick format
    assert "**Test Facet** (`test-facet`)" in summary
    assert "A test facet for validating functionality" in summary

    # Check entities are included with title prefix
    assert "  - **Test Facet Entities**:" in summary
    # Verify some specific entities are present
    assert "John Smith" in summary
    assert "Jane Doe" in summary
    assert "Acme Corp" in summary
    assert "API Optimization" in summary

    # Check other facets are included
    assert "(`full-featured`)" in summary
    assert "(`minimal-facet`)" in summary


def test_facet_summaries_no_facets(monkeypatch, tmp_path):
    """Test facet_summaries() when no facets exist."""
    empty_journal = tmp_path / "empty_journal"
    empty_journal.mkdir()
    monkeypatch.setenv("JOURNAL_PATH", str(empty_journal))

    summary = facet_summaries()
    assert summary == "No facets found."


def test_facet_summaries_no_journal_path(monkeypatch):
    """Test facet_summaries() without JOURNAL_PATH set."""
    monkeypatch.setenv("JOURNAL_PATH", "")

    with pytest.raises(RuntimeError, match="JOURNAL_PATH not set"):
        facet_summaries()


def test_facet_summaries_mixed_entities(monkeypatch):
    """Test facet_summaries() with facets having different entity configurations."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = facet_summaries()

    # Test facet should have entities (semicolon-delimited, not grouped by type)
    assert "**Test Facet** (`test-facet`)" in summary
    assert "  - **Test Facet Entities**:" in summary

    # Minimal facet should not have entity lists
    assert "**Minimal Facet** (`minimal-facet`)" in summary
    # Check that there's no entity list immediately after minimal-facet
    lines = summary.split("\n")
    for i, line in enumerate(lines):
        if "**Minimal Facet** (`minimal-facet`)" in line:
            # Next non-empty line should not be an entity list
            j = i + 1
            while j < len(lines) and lines[j].strip():
                # Should not have Entities line for minimal-facet
                if lines[j].strip().startswith("- **"):
                    # This means we've reached the next facet
                    break
                # If we're still in minimal-facet section, shouldn't have entities
                assert not lines[j].strip().startswith("- **Entities**:")
                j += 1
            break
