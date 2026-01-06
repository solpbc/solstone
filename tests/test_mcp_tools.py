# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from unittest.mock import patch

from apps.entities import tools as entity_tools


def test_entity_add_aka_success():
    """Test successfully adding an aka to an entity."""
    mock_entities = [
        {
            "type": "Tool",
            "name": "PostgreSQL",
            "description": "Database system",
            "aka": ["Postgres"],
        },
        {"type": "Person", "name": "Alice", "description": "Engineer"},
    ]

    with (
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
        patch("apps.entities.tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        result = entity_tools.entity_add_aka("work", "Tool", "PostgreSQL", "PG")

    mock_validate.assert_called_once_with("Tool")
    mock_load.assert_called_once_with("work", day=None)
    mock_save.assert_called_once()

    # Verify the entity was updated
    saved_entities = mock_save.call_args[0][1]
    postgres = next(e for e in saved_entities if e["name"] == "PostgreSQL")
    assert "PG" in postgres["aka"]
    assert "Postgres" in postgres["aka"]

    # Verify response
    assert result["facet"] == "work"
    assert "Added alias 'PG'" in result["message"]
    assert result["entity"]["aka"] == ["Postgres", "PG"]


def test_entity_add_aka_duplicate():
    """Test adding an aka that already exists (dedup)."""
    mock_entities = [
        {
            "type": "Tool",
            "name": "PostgreSQL",
            "description": "Database system",
            "aka": ["Postgres", "PG"],
        },
    ]

    with (
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
        patch("apps.entities.tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        result = entity_tools.entity_add_aka("work", "Tool", "PostgreSQL", "PG")

    # Should not call save since aka already exists
    mock_save.assert_not_called()

    # Verify response
    assert result["facet"] == "work"
    assert "already exists" in result["message"]
    assert result["entity"]["aka"] == ["Postgres", "PG"]


def test_entity_add_aka_initialize_aka_list():
    """Test adding aka to entity that has no aka field yet."""
    mock_entities = [
        {
            "type": "Person",
            "name": "Alice Johnson",
            "description": "Engineer",
        },
    ]

    with (
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
        patch("apps.entities.tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        result = entity_tools.entity_add_aka(
            "personal", "Person", "Alice Johnson", "Ali"
        )

    mock_save.assert_called_once()

    # Verify the aka list was initialized
    saved_entities = mock_save.call_args[0][1]
    alice = next(e for e in saved_entities if e["name"] == "Alice Johnson")
    assert alice["aka"] == ["Ali"]

    # Verify response
    assert result["facet"] == "personal"
    assert "Added alias 'Ali'" in result["message"]
    assert result["entity"]["aka"] == ["Ali"]


def test_entity_add_aka_invalid_type():
    """Test adding aka with invalid entity type."""
    with patch("apps.entities.tools.is_valid_entity_type") as mock_validate:
        mock_validate.return_value = False
        result = entity_tools.entity_add_aka("work", "XY", "PostgreSQL", "PG")

    assert "error" in result
    assert "Invalid entity type 'XY'" in result["error"]
    assert "suggestion" in result


def test_entity_add_aka_entity_not_found():
    """Test adding aka to non-existent entity."""
    mock_entities = [
        {"type": "Person", "name": "Alice", "description": "Engineer"},
    ]

    with (
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        result = entity_tools.entity_add_aka("work", "Tool", "PostgreSQL", "PG")

    assert "error" in result
    assert "not found in attached entities" in result["error"]
    assert "suggestion" in result


def test_entity_add_aka_skip_first_word():
    """Test that adding first word of entity name as aka is silently skipped."""
    mock_entities = [
        {
            "type": "Person",
            "name": "Jeremie Miller",
            "description": "Software engineer",
        },
    ]

    with (
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
        patch("apps.entities.tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        result = entity_tools.entity_add_aka(
            "personal", "Person", "Jeremie Miller", "Jeremie"
        )

    # Should not call save since first word is skipped
    mock_save.assert_not_called()

    # Verify response indicates skip
    assert result["facet"] == "personal"
    assert "first word" in result["message"]
    assert "skipped" in result["message"]


def test_entity_add_aka_skip_first_word_case_insensitive():
    """Test first word skip is case-insensitive."""
    mock_entities = [
        {
            "type": "Organization",
            "name": "Anthropic PBC",
            "description": "AI research company",
        },
    ]

    with (
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
        patch("apps.entities.tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        result = entity_tools.entity_add_aka(
            "work", "Organization", "Anthropic PBC", "anthropic"
        )

    mock_save.assert_not_called()
    assert "first word" in result["message"]
    assert "skipped" in result["message"]


def test_entity_add_aka_skip_first_word_with_parens():
    """Test first word detection strips parentheses from entity name."""
    mock_entities = [
        {
            "type": "Person",
            "name": "Alice Johnson (AJ)",
            "description": "Project manager",
        },
    ]

    with (
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
        patch("apps.entities.tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        result = entity_tools.entity_add_aka(
            "personal", "Person", "Alice Johnson (AJ)", "Alice"
        )

    # Should skip since "Alice" is the first word (ignoring parens)
    mock_save.assert_not_called()
    assert "first word" in result["message"]


def test_entity_add_aka_not_first_word():
    """Test that non-first-word aliases are still added."""
    mock_entities = [
        {
            "type": "Person",
            "name": "Jeremie Miller",
            "description": "Software engineer",
        },
    ]

    with (
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
        patch("apps.entities.tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        result = entity_tools.entity_add_aka(
            "personal", "Person", "Jeremie Miller", "Jer"
        )

    # Should save since "Jer" is not the first word "Jeremie"
    mock_save.assert_called_once()

    # Verify the aka was added
    saved_entities = mock_save.call_args[0][1]
    jeremie = next(e for e in saved_entities if e["name"] == "Jeremie Miller")
    assert "Jer" in jeremie["aka"]

    assert "Added alias 'Jer'" in result["message"]


def test_entity_add_aka_sets_updated_at():
    """Test that entity_add_aka sets updated_at timestamp."""
    mock_entities = [
        {
            "type": "Tool",
            "name": "PostgreSQL",
            "description": "Database system",
            "attached_at": 1700000000000,
            "updated_at": 1700000000000,
        },
    ]

    with (
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
        patch("apps.entities.tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = mock_entities
        entity_tools.entity_add_aka("work", "Tool", "PostgreSQL", "PG")

    mock_save.assert_called_once()
    saved_entities = mock_save.call_args[0][1]
    postgres = next(e for e in saved_entities if e["name"] == "PostgreSQL")

    # updated_at should be newer than original
    assert postgres["updated_at"] > 1700000000000
    # attached_at should remain unchanged
    assert postgres["attached_at"] == 1700000000000


def test_entity_attach_sets_timestamps():
    """Test that entity_attach sets both attached_at and updated_at."""
    with (
        patch("apps.entities.tools.load_entities") as mock_load,
        patch("apps.entities.tools.save_entities") as mock_save,
        patch("apps.entities.tools.is_valid_entity_type") as mock_validate,
    ):
        mock_validate.return_value = True
        mock_load.return_value = []
        entity_tools.entity_attach("work", "Tool", "PostgreSQL", "Database system")

    mock_save.assert_called_once()
    saved_entities = mock_save.call_args[0][1]
    postgres = next(e for e in saved_entities if e["name"] == "PostgreSQL")

    # Both timestamps should be set
    assert "attached_at" in postgres
    assert "updated_at" in postgres
    # Should be recent (within last minute)
    import time

    now = int(time.time() * 1000)
    assert now - postgres["attached_at"] < 60000
    # Initially both should be equal
    assert postgres["attached_at"] == postgres["updated_at"]
