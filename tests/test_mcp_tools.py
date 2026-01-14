# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from unittest.mock import patch

from apps.entities import tools as entity_tools


def test_entity_add_aka_success():
    """Test successfully adding an aka to an entity."""
    mock_entities = [
        {
            "id": "postgresql",
            "type": "Tool",
            "name": "PostgreSQL",
            "description": "Database system",
            "aka": ["Postgres"],
        },
        {"id": "alice", "type": "Person", "name": "Alice", "description": "Engineer"},
    ]

    with (
        patch("apps.entities.tools.resolve_entity") as mock_resolve,
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
    ):
        # resolve_entity returns (entity, None) for success
        mock_resolve.return_value = (mock_entities[0], None)
        mock_load.return_value = mock_entities
        result = entity_tools.entity_add_aka("work", "PostgreSQL", "PG")

    mock_resolve.assert_called_with("work", "PostgreSQL")
    mock_load.assert_called_with("work", day=None, include_detached=True)
    mock_save.assert_called_once()

    # Verify the entity was updated
    saved_entities = mock_save.call_args[0][1]
    postgres = next(e for e in saved_entities if e["name"] == "PostgreSQL")
    assert "PG" in postgres["aka"]
    assert "Postgres" in postgres["aka"]

    # Verify response
    assert result["facet"] == "work"
    assert "Added alias 'PG'" in result["message"]


def test_entity_add_aka_duplicate():
    """Test adding an aka that already exists (dedup)."""
    mock_entity = {
        "id": "postgresql",
        "type": "Tool",
        "name": "PostgreSQL",
        "description": "Database system",
        "aka": ["Postgres", "PG"],
    }

    with (
        patch("apps.entities.tools.resolve_entity") as mock_resolve,
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
    ):
        mock_resolve.return_value = (mock_entity, None)
        mock_load.return_value = [mock_entity]
        result = entity_tools.entity_add_aka("work", "PostgreSQL", "PG")

    # Should not call save since aka already exists
    mock_save.assert_not_called()

    # Verify response
    assert result["facet"] == "work"
    assert "already exists" in result["message"]


def test_entity_add_aka_initialize_aka_list():
    """Test adding aka to entity that has no aka field yet."""
    mock_entity = {
        "id": "alice_johnson",
        "type": "Person",
        "name": "Alice Johnson",
        "description": "Engineer",
    }
    mock_entities = [mock_entity]

    with (
        patch("apps.entities.tools.resolve_entity") as mock_resolve,
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
    ):
        mock_resolve.return_value = (mock_entity, None)
        mock_load.return_value = mock_entities
        result = entity_tools.entity_add_aka("personal", "Alice Johnson", "Ali")

    mock_save.assert_called_once()

    # Verify the aka list was initialized
    saved_entities = mock_save.call_args[0][1]
    alice = next(e for e in saved_entities if e["name"] == "Alice Johnson")
    assert alice["aka"] == ["Ali"]

    # Verify response
    assert result["facet"] == "personal"
    assert "Added alias 'Ali'" in result["message"]


def test_entity_add_aka_entity_not_found():
    """Test adding aka to non-existent entity."""
    mock_entities = [
        {"id": "alice", "type": "Person", "name": "Alice", "description": "Engineer"},
    ]

    with (
        patch("apps.entities.tools.resolve_entity") as mock_resolve,
        patch("apps.entities.tools.load_entities") as mock_load,
    ):
        # resolve_entity returns (None, candidates) for not found
        mock_resolve.return_value = (None, mock_entities)
        mock_load.return_value = mock_entities
        result = entity_tools.entity_add_aka("work", "PostgreSQL", "PG")

    assert "error" in result
    assert "not found in attached entities" in result["error"]
    assert "suggestion" in result


def test_entity_add_aka_skip_first_word():
    """Test that adding first word of entity name as aka is silently skipped."""
    mock_entity = {
        "id": "jeremie_miller",
        "type": "Person",
        "name": "Jeremie Miller",
        "description": "Software engineer",
    }

    with (
        patch("apps.entities.tools.resolve_entity") as mock_resolve,
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
    ):
        mock_resolve.return_value = (mock_entity, None)
        mock_load.return_value = [mock_entity]
        result = entity_tools.entity_add_aka("personal", "Jeremie Miller", "Jeremie")

    # Should not call save since first word is skipped
    mock_save.assert_not_called()

    # Verify response indicates skip
    assert result["facet"] == "personal"
    assert "first word" in result["message"]
    assert "skipped" in result["message"]


def test_entity_add_aka_skip_first_word_case_insensitive():
    """Test first word skip is case-insensitive."""
    mock_entity = {
        "id": "anthropic_pbc",
        "type": "Organization",
        "name": "Anthropic PBC",
        "description": "AI research company",
    }

    with (
        patch("apps.entities.tools.resolve_entity") as mock_resolve,
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
    ):
        mock_resolve.return_value = (mock_entity, None)
        mock_load.return_value = [mock_entity]
        result = entity_tools.entity_add_aka("work", "Anthropic PBC", "anthropic")

    mock_save.assert_not_called()
    assert "first word" in result["message"]
    assert "skipped" in result["message"]


def test_entity_add_aka_skip_first_word_with_parens():
    """Test first word detection strips parentheses from entity name."""
    mock_entity = {
        "id": "alice_johnson_aj",
        "type": "Person",
        "name": "Alice Johnson (AJ)",
        "description": "Product manager",
    }

    with (
        patch("apps.entities.tools.resolve_entity") as mock_resolve,
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
    ):
        mock_resolve.return_value = (mock_entity, None)
        mock_load.return_value = [mock_entity]
        result = entity_tools.entity_add_aka("work", "Alice Johnson (AJ)", "Alice")

    mock_save.assert_not_called()
    assert "first word" in result["message"]
    assert "skipped" in result["message"]


def test_entity_add_aka_valid_non_first_word():
    """Test that adding a valid aka (not first word) succeeds."""
    mock_entity = {
        "id": "alice_johnson_aj",
        "type": "Person",
        "name": "Alice Johnson (AJ)",
        "description": "Product manager",
    }
    mock_entities = [mock_entity]

    with (
        patch("apps.entities.tools.resolve_entity") as mock_resolve,
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
    ):
        mock_resolve.return_value = (mock_entity, None)
        mock_load.return_value = mock_entities
        result = entity_tools.entity_add_aka("work", "Alice Johnson (AJ)", "AJ")

    mock_save.assert_called_once()
    assert "Added alias 'AJ'" in result["message"]


def test_entity_add_aka_by_id():
    """Test adding aka by entity id (slug) instead of name."""
    mock_entity = {
        "id": "postgresql",
        "type": "Tool",
        "name": "PostgreSQL",
        "description": "Database system",
        "aka": ["Postgres"],
    }
    mock_entities = [mock_entity]

    with (
        patch("apps.entities.tools.resolve_entity") as mock_resolve,
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
    ):
        # Resolve by id "postgresql" to entity
        mock_resolve.return_value = (mock_entity, None)
        mock_load.return_value = mock_entities
        result = entity_tools.entity_add_aka("work", "postgresql", "PG")

    mock_resolve.assert_called_with("work", "postgresql")
    mock_save.assert_called_once()
    assert "Added alias 'PG'" in result["message"]
