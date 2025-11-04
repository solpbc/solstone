"""Tests for facet-scoped entity utilities."""

import os

import pytest

from think.entities import (
    entity_file_path,
    load_all_attached_entities,
    load_entities,
    save_entities,
)


@pytest.fixture
def fixture_journal():
    """Set JOURNAL_PATH to fixtures/journal for testing."""
    os.environ["JOURNAL_PATH"] = "fixtures/journal"
    yield
    # No cleanup needed - just testing reads


def test_entity_file_path_attached(fixture_journal):
    """Test path generation for attached entities."""
    path = entity_file_path("personal")
    assert str(path).endswith("fixtures/journal/facets/personal/entities.jsonl")
    assert path.name == "entities.jsonl"


def test_entity_file_path_detected(fixture_journal):
    """Test path generation for detected entities."""
    path = entity_file_path("personal", "20250101")
    assert str(path).endswith(
        "fixtures/journal/facets/personal/entities/20250101.jsonl"
    )
    assert path.name == "20250101.jsonl"


def test_load_entities_attached(fixture_journal):
    """Test loading attached entities from fixtures."""
    entities = load_entities("personal")
    assert len(entities) == 3

    # Check entities are dicts with expected fields
    alice = next(e for e in entities if e.get("name") == "Alice Johnson")
    assert alice["type"] == "Person"
    assert alice["description"] == "Close friend from college"
    # Check extended fields are preserved
    assert alice.get("tags") == ["friend"]
    assert alice.get("contact") == "alice@example.com"

    bob = next(e for e in entities if e.get("name") == "Bob Smith")
    assert bob["type"] == "Person"
    assert bob["description"] == "Neighbor"

    acme = next(e for e in entities if e.get("name") == "Acme Corp")
    assert acme["type"] == "Company"
    assert acme["description"] == "Local tech startup"


def test_load_entities_detected(fixture_journal):
    """Test loading detected entities from fixtures."""
    entities = load_entities("personal", "20250101")
    assert len(entities) == 2

    charlie = next(e for e in entities if e.get("name") == "Charlie Brown")
    assert charlie["type"] == "Person"
    assert charlie["description"] == "Met at coffee shop"

    project = next(e for e in entities if e.get("name") == "Home Renovation")
    assert project["type"] == "Project"
    assert project["description"] == "Kitchen remodel project"


def test_load_entities_missing_file(fixture_journal):
    """Test loading from non-existent file returns empty list."""
    entities = load_entities("personal", "20991231")
    assert entities == []


def test_load_entities_missing_facet(fixture_journal):
    """Test loading from non-existent facet returns empty list."""
    entities = load_entities("nonexistent")
    assert entities == []


def test_save_and_load_entities(fixture_journal, tmp_path):
    """Test saving and loading entities with real files."""
    # Create a temporary facet structure
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)

    # Update JOURNAL_PATH to temp directory
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save some entities (dicts with extended fields)
    test_entities = [
        {
            "type": "Person",
            "name": "Test Person",
            "description": "Test description",
            "role": "tester",
        },
        {"type": "Company", "name": "Test Co", "description": "Test company"},
    ]
    save_entities("test_facet", test_entities, "20250101")

    # Load them back
    loaded = load_entities("test_facet", "20250101")
    assert len(loaded) == 2

    person = next(e for e in loaded if e.get("name") == "Test Person")
    assert person["type"] == "Person"
    assert person["description"] == "Test description"
    assert person.get("role") == "tester"  # Extended field preserved

    company = next(e for e in loaded if e.get("name") == "Test Co")
    assert company["type"] == "Company"
    assert company["description"] == "Test company"

    # Verify file exists and has correct JSONL format
    entity_file = entities_dir / "20250101.jsonl"
    assert entity_file.exists()
    content = entity_file.read_text()
    # Should be valid JSONL
    lines = [line for line in content.strip().split("\n") if line]
    assert len(lines) == 2
    import json

    for line in lines:
        assert json.loads(line)  # Should not raise


def test_save_entities_sorting(fixture_journal, tmp_path):
    """Test that saved entities are sorted by type then name."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save unsorted entities
    import json

    unsorted = [
        {
            "type": "Project",
            "name": "Zebra Project",
            "description": "Last alphabetically",
        },
        {"type": "Company", "name": "Acme", "description": "Company name"},
        {"type": "Person", "name": "Alice", "description": "Person name"},
        {"type": "Company", "name": "Beta Corp", "description": "Another company"},
    ]
    save_entities("test_facet", unsorted)

    # Verify sorting in file (JSONL format)
    entity_file = facet_path / "entities.jsonl"
    lines = entity_file.read_text().strip().split("\n")
    entities = [json.loads(line) for line in lines if line]

    assert entities[0]["type"] == "Company" and entities[0]["name"] == "Acme"
    assert entities[1]["type"] == "Company" and entities[1]["name"] == "Beta Corp"
    assert entities[2]["type"] == "Person" and entities[2]["name"] == "Alice"
    assert entities[3]["type"] == "Project" and entities[3]["name"] == "Zebra Project"


def test_load_all_attached_entities(fixture_journal):
    """Test loading all attached entities from all facets."""
    all_entities = load_all_attached_entities()

    # Should have entities from both personal and full-featured facets
    assert len(all_entities) >= 3  # At least the personal facet entities

    # Check personal facet entities are present
    entity_names = [e.get("name") for e in all_entities]
    assert "Alice Johnson" in entity_names
    assert "Bob Smith" in entity_names
    assert "Acme Corp" in entity_names


def test_load_all_attached_entities_deduplication(fixture_journal, tmp_path):
    """Test that load_all_attached_entities deduplicates by name."""
    # Create two facets with overlapping entity names
    facet1_path = tmp_path / "facets" / "facet1"
    facet2_path = tmp_path / "facets" / "facet2"
    facet1_path.mkdir(parents=True)
    facet2_path.mkdir(parents=True)

    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save same entity name in both facets with different descriptions
    entities1 = [
        {
            "type": "Person",
            "name": "John Smith",
            "description": "Description from facet1",
        }
    ]
    entities2 = [
        {
            "type": "Person",
            "name": "John Smith",
            "description": "Description from facet2",
        }
    ]

    save_entities("facet1", entities1)
    save_entities("facet2", entities2)

    # Load all entities
    all_entities = load_all_attached_entities()

    # Should only have one "John Smith" (from first facet alphabetically)
    john_smiths = [e for e in all_entities if e.get("name") == "John Smith"]
    assert len(john_smiths) == 1
    # Should be from facet1 (alphabetically first)
    assert john_smiths[0]["description"] == "Description from facet1"


def test_aka_field_preservation(fixture_journal, tmp_path):
    """Test that aka field is preserved during save/load operations."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save entities with aka fields
    test_entities = [
        {
            "type": "Person",
            "name": "Alice Johnson",
            "description": "Lead engineer",
            "aka": ["Ali", "AJ"],
        },
        {
            "type": "Company",
            "name": "PostgreSQL",
            "description": "Database system",
            "aka": ["Postgres", "PG"],
        },
    ]
    save_entities("test_facet", test_entities)

    # Load them back
    loaded = load_entities("test_facet")
    assert len(loaded) == 2

    alice = next(e for e in loaded if e.get("name") == "Alice Johnson")
    assert alice.get("aka") == ["Ali", "AJ"]

    postgres = next(e for e in loaded if e.get("name") == "PostgreSQL")
    assert postgres.get("aka") == ["Postgres", "PG"]
