"""Tests for think.domains module."""

from pathlib import Path

import pytest

from think.domains import domain_summaries, domain_summary, get_domains

# Use the permanent fixtures in fixtures/journal/domains/
FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "journal"


def test_domain_summary_full(monkeypatch):
    """Test domain_summary with full metadata."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = domain_summary("full-featured")

    # Check title without emoji
    assert "# Full Featured Domain" in summary

    # Check description
    assert "**Description:** A domain for testing all features" in summary

    # Check color badge
    assert "![Color](#28a745)" in summary

    # Check entities section
    assert "## Entities" in summary
    assert "**Entity 1**: First test entity" in summary
    assert "**Entity 2**: Second test entity" in summary
    assert "**Entity 3**: Third test entity with description" in summary

    # Check matters section
    assert "## Matters" in summary
    assert "**Total:** 2 matter(s)" in summary

    # Check active matters
    assert "### Active (1)" in summary
    assert "🔴 **matter_1**: High Priority Task" in summary

    # Check completed matters
    assert "### Completed (1)" in summary
    assert "**matter_2**: Completed Task" in summary


def test_domain_summary_minimal(monkeypatch):
    """Test domain_summary with minimal metadata."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = domain_summary("minimal-domain")

    # Check title without emoji
    assert "# Minimal Domain" in summary

    # Should not have description, color, entities, or matters
    assert "**Description:**" not in summary
    assert "![Color]" not in summary
    assert "## Entities" not in summary
    assert "## Matters" not in summary


def test_domain_summary_test_domain(monkeypatch):
    """Test domain_summary with the existing test-domain fixture."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = domain_summary("test-domain")

    # Check title without emoji
    assert "# Test Domain" in summary

    # Check description
    assert (
        "**Description:** A test domain for validating matter functionality" in summary
    )

    # Check color badge
    assert "![Color](#007bff)" in summary

    # Check matters section
    assert "## Matters" in summary
    assert "**matter_1**: Test Matter" in summary


def test_domain_summary_nonexistent(monkeypatch):
    """Test domain_summary with nonexistent domain."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    with pytest.raises(FileNotFoundError, match="Domain 'nonexistent' not found"):
        domain_summary("nonexistent")


def test_domain_summary_no_journal_path(monkeypatch):
    """Test domain_summary without JOURNAL_PATH set."""
    # Set to empty string to override any .env file
    monkeypatch.setenv("JOURNAL_PATH", "")

    with pytest.raises(RuntimeError, match="JOURNAL_PATH not set"):
        domain_summary("any-domain")


def test_domain_summary_missing_domain_json(monkeypatch):
    """Test domain_summary with missing domain.json."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    with pytest.raises(FileNotFoundError, match="domain.json not found"):
        domain_summary("broken-domain")


def test_domain_summary_empty_entities(monkeypatch):
    """Test domain_summary with empty entities file."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = domain_summary("empty-entities")

    # Should not include entities section if file is empty
    assert "## Entities" not in summary


def test_domain_summary_matter_priorities(monkeypatch):
    """Test domain_summary with different matter priorities."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = domain_summary("priority-test")

    # Check priority markers
    assert "🔴 **matter_1**: High Priority" in summary
    assert "🟡 **matter_2**: Medium Priority" in summary
    # Normal priority has no marker
    assert "**matter_3**: Normal Priority" in summary
    assert "🔴 **matter_3**" not in summary
    assert "🟡 **matter_3**" not in summary


def test_get_domains_with_entities(monkeypatch):
    """Test that get_domains() returns metadata and load_entity_names() works with domains."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    domains = get_domains()

    # Check test-domain exists
    assert "test-domain" in domains
    test_domain = domains["test-domain"]

    # Check basic metadata
    assert test_domain["title"] == "Test Domain"
    assert test_domain["emoji"] == "🧪"

    # Verify entities are NOT included in get_domains() anymore
    assert "entities" not in test_domain

    # Instead, verify entities can be loaded via load_entity_names()
    from think.utils import load_entity_names

    entity_names = load_entity_names(journal_path=FIXTURES_PATH, domain="test-domain")
    assert entity_names is not None

    # Check that specific entities are in the comma-delimited string
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


def test_get_domains_empty_entities(monkeypatch):
    """Test get_domains() with domain that has no entities."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    domains = get_domains()

    # Check minimal-domain (should have no entities file)
    if "minimal-domain" in domains:
        minimal_domain = domains["minimal-domain"]
        # Entities are no longer included in get_domains()
        assert "entities" not in minimal_domain

        # Verify load_entity_names returns None for domains without entities.md
        from think.utils import load_entity_names

        entity_names = load_entity_names(
            journal_path=FIXTURES_PATH, domain="minimal-domain"
        )
        assert entity_names is None


def test_domain_summaries(monkeypatch):
    """Test domain_summaries() generates correct agent prompt format."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = domain_summaries()

    # Check header
    assert "## Available Domains" in summary

    # Check test-domain is included with hashtag format
    assert "**Test Domain** (#test-domain)" in summary
    assert "A test domain for validating matter functionality" in summary

    # Check entities are included as comma-delimited list (not grouped by type anymore)
    assert "  - **Entities**:" in summary
    # Verify some specific entities are present
    assert "John Smith" in summary
    assert "Jane Doe" in summary
    assert "Acme Corp" in summary
    assert "API Optimization" in summary

    # Check other domains are included
    assert "(#full-featured)" in summary
    assert "(#minimal-domain)" in summary


def test_domain_summaries_no_domains(monkeypatch, tmp_path):
    """Test domain_summaries() when no domains exist."""
    empty_journal = tmp_path / "empty_journal"
    empty_journal.mkdir()
    monkeypatch.setenv("JOURNAL_PATH", str(empty_journal))

    summary = domain_summaries()
    assert summary == "No domains found."


def test_domain_summaries_no_journal_path(monkeypatch):
    """Test domain_summaries() without JOURNAL_PATH set."""
    monkeypatch.setenv("JOURNAL_PATH", "")

    with pytest.raises(RuntimeError, match="JOURNAL_PATH not set"):
        domain_summaries()


def test_domain_summaries_mixed_entities(monkeypatch):
    """Test domain_summaries() with domains having different entity configurations."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = domain_summaries()

    # Test domain should have entities (comma-delimited, not grouped by type)
    assert "**Test Domain** (#test-domain)" in summary
    assert "  - **Entities**:" in summary

    # Minimal domain should not have entity lists
    assert "**Minimal Domain** (#minimal-domain)" in summary
    # Check that there's no entity list immediately after minimal-domain
    lines = summary.split("\n")
    for i, line in enumerate(lines):
        if "**Minimal Domain** (#minimal-domain)" in line:
            # Next non-empty line should not be an entity list
            j = i + 1
            while j < len(lines) and lines[j].strip():
                # Should not have Entities line for minimal-domain
                if lines[j].strip().startswith("- **"):
                    # This means we've reached the next domain
                    break
                # If we're still in minimal-domain section, shouldn't have entities
                assert not lines[j].strip().startswith("- **Entities**:")
                j += 1
            break
