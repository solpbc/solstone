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
    """Test that get_domains() includes parsed entities."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    domains = get_domains()

    # Check test-domain exists
    assert "test-domain" in domains
    test_domain = domains["test-domain"]

    # Check basic metadata
    assert test_domain["title"] == "Test Domain"
    assert test_domain["emoji"] == "🧪"

    # Check entities are parsed and included
    assert "entities" in test_domain
    entities = test_domain["entities"]

    # Should have entity types as keys
    assert "Person" in entities
    assert "Company" in entities
    assert "Project" in entities
    assert "Tool" in entities

    # Check specific entities
    assert "John Smith" in entities["Person"]
    assert "Jane Doe" in entities["Person"]
    assert "Bob Wilson" in entities["Person"]
    assert "Acme Corp" in entities["Company"]
    assert "Tech Solutions Inc" in entities["Company"]
    assert "API Optimization" in entities["Project"]
    assert "Dashboard Redesign" in entities["Project"]
    assert "Visual Studio Code" in entities["Tool"]
    assert "Docker" in entities["Tool"]
    assert "PostgreSQL" in entities["Tool"]


def test_get_domains_empty_entities(monkeypatch):
    """Test get_domains() with domain that has no entities."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    domains = get_domains()

    # Check minimal-domain (should have no entities file)
    if "minimal-domain" in domains:
        minimal_domain = domains["minimal-domain"]
        assert "entities" in minimal_domain
        assert minimal_domain["entities"] == {}


def test_domain_summaries(monkeypatch):
    """Test domain_summaries() generates correct agent prompt format."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = domain_summaries()

    # Check header
    assert "## Available Domains" in summary

    # Check test-domain is included with hashtag format
    assert "**Test Domain** (#test-domain)" in summary
    assert "A test domain for validating matter functionality" in summary

    # Check entities are formatted as sub-lists
    assert "  - **Person**: John Smith, Jane Doe, Bob Wilson" in summary
    assert "  - **Company**: Acme Corp, Tech Solutions Inc" in summary
    assert "  - **Project**: API Optimization, Dashboard Redesign" in summary
    assert "  - **Tool**: Visual Studio Code, Docker, PostgreSQL" in summary

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

    # Test domain should have entities
    assert "**Test Domain** (#test-domain)" in summary
    assert "  - **Person**:" in summary

    # Minimal domain should not have entity sub-lists
    assert "**Minimal Domain** (#minimal-domain)" in summary
    # Check that there's no entity list immediately after minimal-domain
    lines = summary.split("\n")
    for i, line in enumerate(lines):
        if "**Minimal Domain** (#minimal-domain)" in line:
            # Next non-empty line should not be an entity list
            j = i + 1
            while j < len(lines) and lines[j].strip():
                assert not lines[j].strip().startswith("- **Person**:")
                assert not lines[j].strip().startswith("- **Company**:")
                assert not lines[j].strip().startswith("- **Project**:")
                assert not lines[j].strip().startswith("- **Tool**:")
                j += 1
            break
