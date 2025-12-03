"""Tests for facet-scoped entity utilities."""

import os

import pytest

from think.entities import (
    entity_file_path,
    load_all_attached_entities,
    load_detected_entities_recent,
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


# Tests for load_detected_entities_recent


def test_load_detected_entities_recent_basic(fixture_journal):
    """Test loading detected entities with count and last_seen."""
    # Fixture has detected entities in 20250101 and 20250102
    # But these dates are old (> 30 days from now), so we need to use a large days value
    detected = load_detected_entities_recent("personal", days=36500)  # ~100 years

    # Should have 4 detected entities (Charlie Brown, Home Renovation, City Fitness, Diana Prince)
    # Note: excludes Alice Johnson, Bob Smith, Acme Corp which are attached
    assert len(detected) == 4

    # Check structure includes count and last_seen
    for entity in detected:
        assert "type" in entity
        assert "name" in entity
        assert "description" in entity
        assert "count" in entity
        assert "last_seen" in entity


def test_load_detected_entities_recent_excludes_attached(fixture_journal, tmp_path):
    """Test that attached entities and their akas are excluded from detected results."""
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create attached entity with aka
    attached = [
        {
            "type": "Person",
            "name": "Alice Johnson",
            "description": "Attached person",
            "aka": ["Ali", "AJ"],
        }
    ]
    save_entities("test_facet", attached)

    # Create detected entities including some that match attached/aka
    detected_entities = [
        {
            "type": "Person",
            "name": "Alice Johnson",
            "description": "Should be excluded",
        },
        {"type": "Person", "name": "Ali", "description": "Should be excluded (aka)"},
        {
            "type": "Person",
            "name": "Charlie Brown",
            "description": "Should be included",
        },
    ]
    save_entities("test_facet", detected_entities, "20250101")

    # Load detected - should only get Charlie Brown
    detected = load_detected_entities_recent("test_facet", days=36500)
    assert len(detected) == 1
    assert detected[0]["name"] == "Charlie Brown"


def test_load_detected_entities_recent_count_tracking(fixture_journal, tmp_path):
    """Test that count tracks occurrences across multiple days."""
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create same entity across multiple days
    save_entities(
        "test_facet",
        [{"type": "Person", "name": "Charlie", "description": "Day 1 desc"}],
        "20250101",
    )
    save_entities(
        "test_facet",
        [{"type": "Person", "name": "Charlie", "description": "Day 2 desc"}],
        "20250102",
    )
    save_entities(
        "test_facet",
        [{"type": "Person", "name": "Charlie", "description": "Day 3 desc"}],
        "20250103",
    )

    detected = load_detected_entities_recent("test_facet", days=36500)
    assert len(detected) == 1

    charlie = detected[0]
    assert charlie["name"] == "Charlie"
    assert charlie["count"] == 3


def test_load_detected_entities_recent_last_seen(fixture_journal, tmp_path):
    """Test that last_seen is the most recent day and description is from that day."""
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create entity across multiple days with different descriptions
    save_entities(
        "test_facet",
        [{"type": "Person", "name": "Charlie", "description": "Oldest description"}],
        "20250101",
    )
    save_entities(
        "test_facet",
        [
            {
                "type": "Person",
                "name": "Charlie",
                "description": "Most recent description",
            }
        ],
        "20250103",
    )
    save_entities(
        "test_facet",
        [{"type": "Person", "name": "Charlie", "description": "Middle description"}],
        "20250102",
    )

    detected = load_detected_entities_recent("test_facet", days=36500)
    assert len(detected) == 1

    charlie = detected[0]
    assert charlie["last_seen"] == "20250103"
    assert charlie["description"] == "Most recent description"


def test_load_detected_entities_recent_days_filter(fixture_journal, tmp_path):
    """Test that days parameter limits results to recent days."""
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    from datetime import datetime, timedelta

    # Create entities at various dates relative to today
    today = datetime.now()
    recent_day = (today - timedelta(days=5)).strftime("%Y%m%d")
    old_day = (today - timedelta(days=60)).strftime("%Y%m%d")

    save_entities(
        "test_facet",
        [{"type": "Person", "name": "Recent Person", "description": "Recent"}],
        recent_day,
    )
    save_entities(
        "test_facet",
        [{"type": "Person", "name": "Old Person", "description": "Old"}],
        old_day,
    )

    # With default 30 days, should only get recent person
    detected = load_detected_entities_recent("test_facet", days=30)
    assert len(detected) == 1
    assert detected[0]["name"] == "Recent Person"

    # With 90 days, should get both
    detected = load_detected_entities_recent("test_facet", days=90)
    assert len(detected) == 2


def test_load_detected_entities_recent_empty_facet(fixture_journal, tmp_path):
    """Test that empty or non-existent facet returns empty list."""
    facet_path = tmp_path / "facets" / "empty_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # No entities directory
    detected = load_detected_entities_recent("empty_facet")
    assert detected == []


def test_load_detected_entities_recent_type_name_key(fixture_journal, tmp_path):
    """Test that deduplication is by (type, name) tuple, not just name."""
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Same name, different types - should be treated as separate entities
    save_entities(
        "test_facet",
        [
            {"type": "Person", "name": "Mercury", "description": "Roman god"},
            {"type": "Project", "name": "Mercury", "description": "Space program"},
        ],
        "20250101",
    )

    detected = load_detected_entities_recent("test_facet", days=36500)
    assert len(detected) == 2

    names_and_types = {(e["type"], e["name"]) for e in detected}
    assert ("Person", "Mercury") in names_and_types
    assert ("Project", "Mercury") in names_and_types


def test_timestamp_preservation(fixture_journal, tmp_path):
    """Test that attached_at and updated_at timestamps are preserved through save/load."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save entities with timestamps
    test_entities = [
        {
            "type": "Person",
            "name": "Alice",
            "description": "Test person",
            "attached_at": 1700000000000,
            "updated_at": 1700000001000,
        },
        {
            "type": "Company",
            "name": "Acme",
            "description": "Test company",
            "attached_at": 1700000002000,
            "updated_at": 1700000002000,
        },
    ]
    save_entities("test_facet", test_entities)

    # Load them back
    loaded = load_entities("test_facet")
    assert len(loaded) == 2

    alice = next(e for e in loaded if e.get("name") == "Alice")
    assert alice["attached_at"] == 1700000000000
    assert alice["updated_at"] == 1700000001000

    acme = next(e for e in loaded if e.get("name") == "Acme")
    assert acme["attached_at"] == 1700000002000
    assert acme["updated_at"] == 1700000002000
