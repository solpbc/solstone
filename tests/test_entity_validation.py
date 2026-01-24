# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for entity type validation."""

from think.entities import ENTITY_TYPES, is_valid_entity_type


def test_entity_types_constant():
    """Test that ENTITY_TYPES constant has expected structure."""
    assert len(ENTITY_TYPES) == 4
    assert all("name" in t for t in ENTITY_TYPES)
    # Verify standard types in expected order
    names = [t["name"] for t in ENTITY_TYPES]
    assert names == ["Person", "Company", "Project", "Tool"]


def test_valid_standard_types():
    """Test that standard 4 entity types are valid."""
    assert is_valid_entity_type("Person") is True
    assert is_valid_entity_type("Company") is True
    assert is_valid_entity_type("Project") is True
    assert is_valid_entity_type("Tool") is True


def test_valid_custom_types():
    """Test that custom entity types are accepted."""
    assert is_valid_entity_type("Location") is True
    assert is_valid_entity_type("Event") is True
    assert is_valid_entity_type("Organization") is True
    assert is_valid_entity_type("Product") is True
    assert is_valid_entity_type("Service") is True


def test_valid_types_with_spaces():
    """Test that entity types with spaces are accepted."""
    assert is_valid_entity_type("Custom Type") is True
    assert is_valid_entity_type("Event Location") is True
    assert is_valid_entity_type("Tech Product") is True


def test_valid_types_with_numbers():
    """Test that entity types with numbers are accepted."""
    assert is_valid_entity_type("Type123") is True
    assert is_valid_entity_type("Event 2024") is True
    assert is_valid_entity_type("Version 2") is True


def test_invalid_too_short():
    """Test that types shorter than 3 characters are rejected."""
    assert is_valid_entity_type("") is False
    assert is_valid_entity_type("A") is False
    assert is_valid_entity_type("AB") is False


def test_invalid_special_characters():
    """Test that types with special characters are rejected."""
    assert is_valid_entity_type("Person!") is False
    assert is_valid_entity_type("Type-Name") is False
    assert is_valid_entity_type("Type_Name") is False
    assert is_valid_entity_type("Type@Name") is False
    assert is_valid_entity_type("Type#Name") is False
    assert is_valid_entity_type("Type$Name") is False
    assert is_valid_entity_type("Type%Name") is False
    assert is_valid_entity_type("Type&Name") is False
    assert is_valid_entity_type("Type*Name") is False
    assert is_valid_entity_type("Type(Name)") is False
    assert is_valid_entity_type("Type/Name") is False
    assert is_valid_entity_type("Type\\Name") is False
    assert is_valid_entity_type("Type.Name") is False
    assert is_valid_entity_type("Type,Name") is False


def test_invalid_unicode():
    """Test that unicode/emoji characters are rejected."""
    assert is_valid_entity_type("Type🚀") is False
    assert is_valid_entity_type("Typé") is False
    assert is_valid_entity_type("类型") is False


def test_none_and_whitespace():
    """Test that None and whitespace-only strings are rejected."""
    assert is_valid_entity_type("   ") is False
    assert is_valid_entity_type("\t") is False
    assert is_valid_entity_type("\n") is False


def test_case_sensitivity():
    """Test that validation is case-insensitive (accepts both)."""
    assert is_valid_entity_type("person") is True
    assert is_valid_entity_type("PERSON") is True
    assert is_valid_entity_type("PeRsOn") is True
    assert is_valid_entity_type("custom type") is True
    assert is_valid_entity_type("CUSTOM TYPE") is True
