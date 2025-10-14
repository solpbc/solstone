"""Tests for domain-scoped entity utilities."""

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
    assert str(path).endswith("fixtures/journal/domains/personal/entities.md")
    assert path.name == "entities.md"


def test_entity_file_path_detected(fixture_journal):
    """Test path generation for detected entities."""
    path = entity_file_path("personal", "20250101")
    assert str(path).endswith("fixtures/journal/domains/personal/entities/20250101.md")
    assert path.name == "20250101.md"


def test_load_entities_attached(fixture_journal):
    """Test loading attached entities from fixtures."""
    entities = load_entities("personal")
    assert len(entities) == 3
    assert ("Person", "Alice Johnson", "Close friend from college") in entities
    assert ("Person", "Bob Smith", "Neighbor") in entities
    assert ("Company", "Acme Corp", "Local tech startup") in entities


def test_load_entities_detected(fixture_journal):
    """Test loading detected entities from fixtures."""
    entities = load_entities("personal", "20250101")
    assert len(entities) == 2
    assert ("Person", "Charlie Brown", "Met at coffee shop") in entities
    assert ("Project", "Home Renovation", "Kitchen remodel project") in entities


def test_load_entities_missing_file(fixture_journal):
    """Test loading from non-existent file returns empty list."""
    entities = load_entities("personal", "20991231")
    assert entities == []


def test_load_entities_missing_domain(fixture_journal):
    """Test loading from non-existent domain returns empty list."""
    entities = load_entities("nonexistent")
    assert entities == []


def test_save_and_load_entities(fixture_journal, tmp_path):
    """Test saving and loading entities with real files."""
    # Create a temporary domain structure
    domain_path = tmp_path / "domains" / "test_domain"
    entities_dir = domain_path / "entities"
    entities_dir.mkdir(parents=True)

    # Update JOURNAL_PATH to temp directory
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save some entities
    test_entities = [
        ("Person", "Test Person", "Test description"),
        ("Company", "Test Co", "Test company"),
    ]
    save_entities("test_domain", test_entities, "20250101")

    # Load them back
    loaded = load_entities("test_domain", "20250101")
    assert len(loaded) == 2
    assert ("Person", "Test Person", "Test description") in loaded
    assert ("Company", "Test Co", "Test company") in loaded

    # Verify file exists and has correct format
    entity_file = entities_dir / "20250101.md"
    assert entity_file.exists()
    content = entity_file.read_text()
    assert "- **Company**: Test Co - Test company" in content
    assert "- **Person**: Test Person - Test description" in content


def test_save_entities_sorting(fixture_journal, tmp_path):
    """Test that saved entities are sorted by type then name."""
    domain_path = tmp_path / "domains" / "test_domain"
    domain_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save unsorted entities
    unsorted = [
        ("Project", "Zebra Project", "Last alphabetically"),
        ("Company", "Acme", "Company name"),
        ("Person", "Alice", "Person name"),
        ("Company", "Beta Corp", "Another company"),
    ]
    save_entities("test_domain", unsorted)

    # Verify sorting in file
    entity_file = domain_path / "entities.md"
    lines = entity_file.read_text().strip().split("\n")
    assert lines[0].startswith("- **Company**: Acme")
    assert lines[1].startswith("- **Company**: Beta Corp")
    assert lines[2].startswith("- **Person**: Alice")
    assert lines[3].startswith("- **Project**: Zebra Project")


def test_load_all_attached_entities(fixture_journal):
    """Test loading all attached entities from all domains."""
    all_entities = load_all_attached_entities()

    # Should have entities from both personal and full-featured domains
    assert len(all_entities) >= 3  # At least the personal domain entities

    # Check personal domain entities are present
    entity_names = [name for _, name, _ in all_entities]
    assert "Alice Johnson" in entity_names
    assert "Bob Smith" in entity_names
    assert "Acme Corp" in entity_names


def test_load_all_attached_entities_deduplication(fixture_journal, tmp_path):
    """Test that load_all_attached_entities deduplicates by name."""
    # Create two domains with overlapping entity names
    domain1_path = tmp_path / "domains" / "domain1"
    domain2_path = tmp_path / "domains" / "domain2"
    domain1_path.mkdir(parents=True)
    domain2_path.mkdir(parents=True)

    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save same entity name in both domains with different descriptions
    entities1 = [("Person", "John Smith", "Description from domain1")]
    entities2 = [("Person", "John Smith", "Description from domain2")]

    save_entities("domain1", entities1)
    save_entities("domain2", entities2)

    # Load all entities
    all_entities = load_all_attached_entities()

    # Should only have one "John Smith" (from first domain alphabetically)
    john_smiths = [e for e in all_entities if e[1] == "John Smith"]
    assert len(john_smiths) == 1
    # Should be from domain1 (alphabetically first)
    assert john_smiths[0][2] == "Description from domain1"
