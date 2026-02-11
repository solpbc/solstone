# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.facets module."""

import json
from pathlib import Path

import pytest
from slugify import slugify

from think.facets import (
    _format_principal_role,
    _get_principal_display_name,
    facet_summaries,
    facet_summary,
    get_active_facets,
    get_facets,
)

# Use the permanent fixtures in tests/fixtures/journal/facets/
FIXTURES_PATH = Path(__file__).parent / "fixtures" / "journal"


def setup_entities_new_structure(
    journal_path: Path,
    facet: str,
    entities: list[dict],
):
    """Helper to set up entities using the new structure for tests.

    Creates both journal-level entity files and facet relationship files.

    Args:
        journal_path: Path to journal root
        facet: Facet name (e.g., "work")
        entities: List of entity dicts with type, name, description, etc.
    """
    for entity in entities:
        etype = entity.get("type", "")
        name = entity.get("name", "")
        desc = entity.get("description", "")
        is_principal = entity.get("is_principal", False)

        entity_id = slugify(name, separator="_")
        if not entity_id:
            continue

        # Create journal-level entity
        journal_entity_dir = journal_path / "entities" / entity_id
        journal_entity_dir.mkdir(parents=True, exist_ok=True)
        journal_entity = {"id": entity_id, "name": name, "type": etype}
        if is_principal:
            journal_entity["is_principal"] = True
        with open(journal_entity_dir / "entity.json", "w", encoding="utf-8") as f:
            json.dump(journal_entity, f)

        # Create facet relationship
        facet_entity_dir = journal_path / "facets" / facet / "entities" / entity_id
        facet_entity_dir.mkdir(parents=True, exist_ok=True)
        relationship = {"entity_id": entity_id, "description": desc}
        with open(facet_entity_dir / "entity.json", "w", encoding="utf-8") as f:
            json.dump(relationship, f)


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

    # Check activities section
    assert "## Activities" in summary
    assert "**Meetings** (high)" in summary
    assert "**Coding**" in summary
    assert "**Custom Activity**:" in summary
    assert "A custom test activity" in summary


def test_facet_summary_short_mode(monkeypatch):
    """Test facet_summary with detailed=False shows names only."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = facet_summary("full-featured", detailed=False)

    # Check title and description still present
    assert "# Full Featured Facet" in summary
    assert "**Description:** A facet for testing all features" in summary

    # Should NOT have detailed entities section
    assert "## Entities" not in summary
    # Should have inline entities list
    assert "**Entities**:" in summary

    # Should NOT have detailed activities section
    assert "## Activities" not in summary
    # Should have inline activities list
    assert "**Activities**:" in summary

    # Should NOT have activity descriptions
    assert "A custom test activity" not in summary


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


def test_facet_summary_uses_default_path_when_journal_path_empty(monkeypatch):
    """Test facet_summary uses platform default when JOURNAL_PATH is empty."""
    # Set to empty string - uses platform default (facet won't exist there)
    monkeypatch.setenv("JOURNAL_PATH", "")

    # Should raise FileNotFoundError because facet doesn't exist in default path
    with pytest.raises(FileNotFoundError, match="not found"):
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

        # Verify load_entity_names returns None for facets without entities
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

    # Check activities are included (short mode - names only)
    assert (
        "**Full Featured Facet Activities**: Meetings; Coding; Custom Activity"
        in summary
    )


def test_facet_summaries_excludes_muted(monkeypatch, tmp_path):
    """Test facet_summaries() excludes muted facets."""
    facets_dir = tmp_path / "facets"
    active_dir = facets_dir / "active"
    muted_dir = facets_dir / "muted_one"
    active_dir.mkdir(parents=True)
    muted_dir.mkdir(parents=True)

    (active_dir / "facet.json").write_text(
        json.dumps({"name": "active", "title": "Active Facet"}),
        encoding="utf-8",
    )
    (muted_dir / "facet.json").write_text(
        json.dumps({"name": "muted_one", "title": "Muted Facet", "muted": True}),
        encoding="utf-8",
    )

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    summary = facet_summaries()

    assert "(`active`)" in summary
    assert "(`muted_one`)" not in summary


def test_facet_summaries_no_facets(monkeypatch, tmp_path):
    """Test facet_summaries() when no facets exist."""
    empty_journal = tmp_path / "empty_journal"
    empty_journal.mkdir()
    monkeypatch.setenv("JOURNAL_PATH", str(empty_journal))

    summary = facet_summaries()
    assert summary == "No facets found."


def test_facet_summaries_uses_default_path_when_journal_path_empty(monkeypatch):
    """Test facet_summaries() uses platform default when JOURNAL_PATH is empty."""
    monkeypatch.setenv("JOURNAL_PATH", "")

    # Should return "No facets found" since default path has no facets
    summary = facet_summaries()
    assert summary == "No facets found."


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


def test_get_active_facets_from_segment_facets(monkeypatch, tmp_path):
    """Test get_active_facets() returns facets from segment facets.json files."""
    journal = tmp_path / "journal"
    day_dir = journal / "20240115"

    # Create segment with facets.json containing two facets (stream layout)
    seg1 = day_dir / "archon" / "100000_300" / "agents"
    seg1.mkdir(parents=True)
    (seg1 / "facets.json").write_text(
        json.dumps(
            [
                {"facet": "work", "activity": "Code review", "level": "high"},
                {"facet": "personal", "activity": "Email check", "level": "low"},
            ]
        )
    )

    # Create another segment with overlapping + new facet
    seg2 = day_dir / "archon" / "110000_300" / "agents"
    seg2.mkdir(parents=True)
    (seg2 / "facets.json").write_text(
        json.dumps(
            [
                {"facet": "work", "activity": "Meeting", "level": "high"},
                {"facet": "sunstone", "activity": "Dev work", "level": "medium"},
            ]
        )
    )

    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    active = get_active_facets("20240115")

    assert active == {"work", "personal", "sunstone"}


def test_get_active_facets_empty_segments(monkeypatch, tmp_path):
    """Test get_active_facets() with segments that have empty facets.json."""
    journal = tmp_path / "journal"
    day_dir = journal / "20240115"

    # Segment with empty facets array (stream layout)
    seg1 = day_dir / "archon" / "100000_300" / "agents"
    seg1.mkdir(parents=True)
    (seg1 / "facets.json").write_text("[]")

    # Segment with empty file
    seg2 = day_dir / "archon" / "110000_300" / "agents"
    seg2.mkdir(parents=True)
    (seg2 / "facets.json").write_text("")

    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    active = get_active_facets("20240115")

    assert active == set()


def test_get_active_facets_no_segments(monkeypatch, tmp_path):
    """Test get_active_facets() when day directory has no segments."""
    journal = tmp_path / "journal"
    (journal / "20240115").mkdir(parents=True)

    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    active = get_active_facets("20240115")

    assert active == set()


def test_get_active_facets_no_day_dir(monkeypatch, tmp_path):
    """Test get_active_facets() when day directory doesn't exist."""
    journal = tmp_path / "journal"
    journal.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    active = get_active_facets("20240115")

    assert active == set()


def test_get_active_facets_malformed_json(monkeypatch, tmp_path):
    """Test get_active_facets() skips malformed facets.json gracefully."""
    journal = tmp_path / "journal"
    day_dir = journal / "20240115"

    # Malformed JSON segment (stream layout)
    seg1 = day_dir / "archon" / "100000_300" / "agents"
    seg1.mkdir(parents=True)
    (seg1 / "facets.json").write_text("{ invalid json")

    # Valid segment
    seg2 = day_dir / "archon" / "110000_300" / "agents"
    seg2.mkdir(parents=True)
    (seg2 / "facets.json").write_text(
        json.dumps(
            [
                {"facet": "work", "activity": "Coding", "level": "high"},
            ]
        )
    )

    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    active = get_active_facets("20240115")

    assert active == {"work"}


# ============================================================================
# Principal role in facet summaries tests
# ============================================================================


def test_get_principal_display_name_preferred(tmp_path, monkeypatch):
    """Test _get_principal_display_name returns preferred name."""
    import json

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = {"identity": {"name": "Jeremy Miller", "preferred": "Jer"}}
    (config_dir / "journal.json").write_text(json.dumps(config))

    assert _get_principal_display_name() == "Jer"


def test_get_principal_display_name_fallback_to_name(tmp_path, monkeypatch):
    """Test _get_principal_display_name falls back to name when no preferred."""
    import json

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = {"identity": {"name": "Jeremy Miller", "preferred": ""}}
    (config_dir / "journal.json").write_text(json.dumps(config))

    assert _get_principal_display_name() == "Jeremy Miller"


def test_get_principal_display_name_none_when_empty(tmp_path, monkeypatch):
    """Test _get_principal_display_name returns None when identity empty."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    # No config file

    assert _get_principal_display_name() is None


def test_format_principal_role_with_principal(tmp_path, monkeypatch):
    """Test _format_principal_role extracts and formats principal."""
    import json

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = {"identity": {"name": "Jeremy", "preferred": "Jer"}}
    (config_dir / "journal.json").write_text(json.dumps(config))

    entities = [
        {"name": "Jeremy", "description": "Software engineer", "is_principal": True},
        {"name": "Bob", "description": "Friend"},
    ]

    role_line, filtered = _format_principal_role(entities)

    assert role_line == "**Jer's Role**: Software engineer"
    assert len(filtered) == 1
    assert filtered[0]["name"] == "Bob"


def test_format_principal_role_no_principal(tmp_path, monkeypatch):
    """Test _format_principal_role returns None when no principal."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    entities = [
        {"name": "Alice", "description": "Friend"},
        {"name": "Bob", "description": "Colleague"},
    ]

    role_line, filtered = _format_principal_role(entities)

    assert role_line is None
    assert filtered == entities


def test_format_principal_role_no_description(tmp_path, monkeypatch):
    """Test _format_principal_role returns None when principal has no description."""
    import json

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = {"identity": {"name": "Jeremy", "preferred": "Jer"}}
    (config_dir / "journal.json").write_text(json.dumps(config))

    entities = [
        {"name": "Jeremy", "description": "", "is_principal": True},
        {"name": "Bob", "description": "Friend"},
    ]

    role_line, filtered = _format_principal_role(entities)

    # No role line because description is empty
    assert role_line is None
    # But principal is still filtered out
    assert filtered == entities


def test_format_principal_role_no_identity(tmp_path, monkeypatch):
    """Test _format_principal_role returns None when no identity configured."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    # No config file

    entities = [
        {"name": "Jeremy", "description": "Engineer", "is_principal": True},
        {"name": "Bob", "description": "Friend"},
    ]

    role_line, filtered = _format_principal_role(entities)

    # No role line because no identity config
    assert role_line is None
    # Entities unchanged
    assert filtered == entities


def test_facet_summary_with_principal(tmp_path, monkeypatch):
    """Test facet_summary shows principal role and excludes from entities list."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Create identity config
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = {"identity": {"name": "Test User", "preferred": "Tester"}}
    (config_dir / "journal.json").write_text(json.dumps(config))

    # Create facet with principal entity using new structure
    facet_dir = tmp_path / "facets" / "work"
    facet_dir.mkdir(parents=True)
    (facet_dir / "facet.json").write_text(
        json.dumps({"title": "Work", "description": "Work stuff"})
    )
    setup_entities_new_structure(
        tmp_path,
        "work",
        [
            {
                "type": "Person",
                "name": "Test User",
                "description": "Lead developer",
                "is_principal": True,
            },
            {"type": "Person", "name": "Alice", "description": "Colleague"},
        ],
    )

    summary = facet_summary("work")

    # Should have principal role line
    assert "**Tester's Role**: Lead developer" in summary
    # Should have entities section with Alice but not Test User
    assert "## Entities" in summary
    assert "Alice" in summary
    assert "Colleague" in summary
    # Principal should not appear in entities list
    assert "- **Person**: Test User" not in summary


def test_facet_summary_principal_only_entity(tmp_path, monkeypatch):
    """Test facet_summary when principal is the only entity."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Create identity config
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = {"identity": {"name": "Test User", "preferred": "Tester"}}
    (config_dir / "journal.json").write_text(json.dumps(config))

    # Create facet with only principal entity using new structure
    facet_dir = tmp_path / "facets" / "solo"
    facet_dir.mkdir(parents=True)
    (facet_dir / "facet.json").write_text(json.dumps({"title": "Solo"}))
    setup_entities_new_structure(
        tmp_path,
        "solo",
        [
            {
                "type": "Person",
                "name": "Test User",
                "description": "Just me",
                "is_principal": True,
            },
        ],
    )

    summary = facet_summary("solo")

    # Should have principal role line
    assert "**Tester's Role**: Just me" in summary
    # Should NOT have entities section (no other entities)
    assert "## Entities" not in summary


def test_facet_summaries_detailed_with_principal(tmp_path, monkeypatch):
    """Test facet_summaries detailed mode shows principal role."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Create identity config
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = {"identity": {"name": "Test User", "preferred": "Tester"}}
    (config_dir / "journal.json").write_text(json.dumps(config))

    # Create facet with principal using new structure
    facet_dir = tmp_path / "facets" / "project"
    facet_dir.mkdir(parents=True)
    (facet_dir / "facet.json").write_text(
        json.dumps({"title": "Project X", "description": "Secret project"})
    )
    setup_entities_new_structure(
        tmp_path,
        "project",
        [
            {
                "type": "Person",
                "name": "Test User",
                "description": "Project lead",
                "is_principal": True,
            },
            {"type": "Person", "name": "Bob", "description": "Team member"},
        ],
    )

    summary = facet_summaries(detailed=True)

    # Should have principal role
    assert "**Tester's Role**: Project lead" in summary
    # Should have Bob in entities
    assert "Bob: Team member" in summary
    # Principal should not be in entities list
    assert "Test User: Project lead" not in summary


def test_facet_summaries_simple_mode_with_principal(tmp_path, monkeypatch):
    """Test facet_summaries simple mode also filters principal consistently."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Create identity config
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = {"identity": {"name": "Test User", "preferred": "Tester"}}
    (config_dir / "journal.json").write_text(json.dumps(config))

    # Create facet with principal using new structure
    facet_dir = tmp_path / "facets" / "simple"
    facet_dir.mkdir(parents=True)
    (facet_dir / "facet.json").write_text(json.dumps({"title": "Simple"}))
    setup_entities_new_structure(
        tmp_path,
        "simple",
        [
            {
                "type": "Person",
                "name": "Test User",
                "description": "Me",
                "is_principal": True,
            },
            {"type": "Person", "name": "Bob", "description": "Friend"},
        ],
    )

    summary = facet_summaries(detailed=False)

    # Simple mode now shows principal role (consistent with detailed mode)
    assert "**Tester's Role**: Me" in summary
    # Principal should not appear in entity names
    assert "Test User" not in summary
    # Other entities should appear
    assert "Bob" in summary


def test_facet_summaries_detailed_with_activities(monkeypatch):
    """Test facet_summaries detailed mode includes activity details."""
    monkeypatch.setenv("JOURNAL_PATH", str(FIXTURES_PATH))

    summary = facet_summaries(detailed=True)

    # Check activities are included with details
    assert "**Full Featured Facet Activities**:" in summary
    assert "Meetings (high):" in summary
    assert "Video calls, in-person meetings, and conferences" in summary
    assert "Coding:" in summary
    assert "Custom Activity:" in summary
    assert "A custom test activity" in summary
