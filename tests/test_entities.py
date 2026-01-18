# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for facet-scoped entity utilities."""

import os

import pytest

from think.entities import (
    ObservationNumberError,
    add_observation,
    ensure_entity_memory,
    entity_file_path,
    entity_memory_path,
    entity_slug,
    find_matching_attached_entity,
    load_all_attached_entities,
    load_detected_entities_recent,
    load_entities,
    load_observations,
    load_recent_entity_names,
    observations_file_path,
    parse_knowledge_graph_entities,
    rename_entity_memory,
    save_entities,
    save_observations,
    touch_entities_from_activity,
    touch_entity,
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


def test_load_all_attached_entities_sort_by_last_seen(fixture_journal, tmp_path):
    """Test sorting entities by last_seen."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create entities with varying last_seen values
    entities = [
        {"type": "Person", "name": "Old Entity", "description": "No last_seen"},
        {
            "type": "Person",
            "name": "Recent Entity",
            "description": "Most recent",
            "last_seen": "20260108",
        },
        {
            "type": "Person",
            "name": "Middle Entity",
            "description": "Middle",
            "last_seen": "20260105",
        },
    ]
    save_entities("test_facet", entities)

    # Load with sorting
    result = load_all_attached_entities(sort_by="last_seen")

    # Most recent should be first, no last_seen should be last
    assert result[0]["name"] == "Recent Entity"
    assert result[1]["name"] == "Middle Entity"
    assert result[2]["name"] == "Old Entity"


def test_load_all_attached_entities_limit(fixture_journal, tmp_path):
    """Test limiting number of entities returned."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create 5 entities
    entities = [
        {"type": "Person", "name": f"Entity {i}", "description": f"Desc {i}"}
        for i in range(5)
    ]
    save_entities("test_facet", entities)

    # Load with limit
    result = load_all_attached_entities(limit=3)
    assert len(result) == 3


def test_load_all_attached_entities_sort_and_limit(fixture_journal, tmp_path):
    """Test sorting and limiting together."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create entities with last_seen
    entities = [
        {"type": "Person", "name": "A", "last_seen": "20260101"},
        {"type": "Person", "name": "B", "last_seen": "20260108"},
        {"type": "Person", "name": "C", "last_seen": "20260105"},
        {"type": "Person", "name": "D", "last_seen": "20260103"},
    ]
    save_entities("test_facet", entities)

    # Get top 2 most recent
    result = load_all_attached_entities(sort_by="last_seen", limit=2)
    assert len(result) == 2
    assert result[0]["name"] == "B"  # 20260108
    assert result[1]["name"] == "C"  # 20260105


# Tests for load_recent_entity_names


def test_load_recent_entity_names_basic(fixture_journal, tmp_path):
    """Test basic functionality of load_recent_entity_names."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create entities with last_seen
    entities = [
        {"type": "Person", "name": "Alice Johnson", "last_seen": "20260108"},
        {"type": "Company", "name": "Acme Corp", "last_seen": "20260107"},
    ]
    save_entities("test_facet", entities)

    result = load_recent_entity_names()

    # Should return list of spoken forms
    assert result is not None
    assert isinstance(result, list)
    assert "Alice" in result
    assert "Acme" in result


def test_load_recent_entity_names_returns_list(fixture_journal, tmp_path):
    """Test that result is a list of names."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create 10 entities
    entities = [
        {"type": "Person", "name": f"Name{i}", "last_seen": f"202601{i:02d}"}
        for i in range(10, 0, -1)  # Descending so we get predictable order
    ]
    save_entities("test_facet", entities)

    result = load_recent_entity_names(limit=10)

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 10


def test_load_recent_entity_names_empty(fixture_journal, tmp_path):
    """Test with no entities returns None."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    result = load_recent_entity_names()
    assert result is None


def test_load_recent_entity_names_with_aka(fixture_journal, tmp_path):
    """Test that aka values are included in spoken names."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    entities = [
        {
            "type": "Person",
            "name": "Robert Johnson",
            "aka": ["Bob", "Bobby"],
            "last_seen": "20260108",
        },
    ]
    save_entities("test_facet", entities)

    result = load_recent_entity_names()

    assert result is not None
    assert isinstance(result, list)
    assert "Robert" in result
    assert "Bob" in result
    assert "Bobby" in result


def test_load_recent_entity_names_respects_limit(fixture_journal, tmp_path):
    """Test that limit parameter is respected."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create 30 entities
    entities = [
        {"type": "Person", "name": f"Person{i}", "last_seen": f"202601{i:02d}"}
        for i in range(1, 31)
    ]
    save_entities("test_facet", entities)

    # Request only 5
    result = load_recent_entity_names(limit=5)

    assert result is not None
    assert isinstance(result, list)
    # Most recent 5 should be included (Person30, Person29, Person28, Person27, Person26)
    assert "Person30" in result
    assert "Person26" in result
    # Earlier ones should not be included
    assert "Person1" not in result


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


# Tests for detached entity functionality


def test_load_entities_excludes_detached_by_default(fixture_journal, tmp_path):
    """Test that load_entities excludes detached entities by default."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save entities with one detached
    test_entities = [
        {"type": "Person", "name": "Alice", "description": "Active person"},
        {
            "type": "Person",
            "name": "Bob",
            "description": "Detached person",
            "detached": True,
        },
        {"type": "Company", "name": "Acme", "description": "Active company"},
    ]
    save_entities("test_facet", test_entities)

    # Load without include_detached (default)
    loaded = load_entities("test_facet")
    assert len(loaded) == 2
    names = [e["name"] for e in loaded]
    assert "Alice" in names
    assert "Acme" in names
    assert "Bob" not in names


def test_load_entities_includes_detached_when_requested(fixture_journal, tmp_path):
    """Test that load_entities includes detached entities when include_detached=True."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save entities with one detached
    test_entities = [
        {"type": "Person", "name": "Alice", "description": "Active person"},
        {
            "type": "Person",
            "name": "Bob",
            "description": "Detached person",
            "detached": True,
        },
    ]
    save_entities("test_facet", test_entities)

    # Load with include_detached=True
    loaded = load_entities("test_facet", include_detached=True)
    assert len(loaded) == 2
    names = [e["name"] for e in loaded]
    assert "Alice" in names
    assert "Bob" in names

    # Verify detached flag is preserved
    bob = next(e for e in loaded if e["name"] == "Bob")
    assert bob.get("detached") is True


def test_load_all_attached_entities_excludes_detached(fixture_journal, tmp_path):
    """Test that load_all_attached_entities excludes detached entities."""
    facet1_path = tmp_path / "facets" / "facet1"
    facet2_path = tmp_path / "facets" / "facet2"
    facet1_path.mkdir(parents=True)
    facet2_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save entities - one active, one detached per facet
    save_entities(
        "facet1",
        [
            {"type": "Person", "name": "Alice", "description": "Active in facet1"},
            {
                "type": "Person",
                "name": "Bob",
                "description": "Detached in facet1",
                "detached": True,
            },
        ],
    )
    save_entities(
        "facet2",
        [
            {"type": "Person", "name": "Charlie", "description": "Active in facet2"},
        ],
    )

    all_entities = load_all_attached_entities()

    # Should only have active entities
    names = [e["name"] for e in all_entities]
    assert "Alice" in names
    assert "Charlie" in names
    assert "Bob" not in names


def test_load_detected_entities_recent_shows_detached_entity_names(
    fixture_journal, tmp_path
):
    """Test that detached entities appear in detected list again (not excluded)."""
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create attached entity with detached=True
    attached = [
        {"type": "Person", "name": "Alice", "description": "Active person"},
        {
            "type": "Person",
            "name": "Bob",
            "description": "Detached person",
            "detached": True,
        },
    ]
    save_entities("test_facet", attached)

    # Create detected entities including the detached name
    detected_entities = [
        {
            "type": "Person",
            "name": "Alice",
            "description": "Should be excluded (active)",
        },
        {
            "type": "Person",
            "name": "Bob",
            "description": "Should be INCLUDED (detached)",
        },
        {
            "type": "Person",
            "name": "Charlie",
            "description": "Should be included (new)",
        },
    ]
    save_entities("test_facet", detected_entities, "20250101")

    # Load detected - Alice excluded (active), Bob included (detached), Charlie included (new)
    detected = load_detected_entities_recent("test_facet", days=36500)
    names = [e["name"] for e in detected]

    assert "Alice" not in names  # Excluded - still active
    assert "Bob" in names  # Included - detached, so shows up in detected
    assert "Charlie" in names  # Included - new entity


def test_detached_entity_preserves_all_fields(fixture_journal, tmp_path):
    """Test that detached entities preserve all fields including custom ones."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save entity with custom fields and detached flag
    test_entities = [
        {
            "type": "Person",
            "name": "Alice",
            "description": "Test person",
            "attached_at": 1700000000000,
            "updated_at": 1700000001000,
            "aka": ["Ali", "AJ"],
            "tags": ["friend", "colleague"],
            "custom_field": "custom_value",
            "detached": True,
        },
    ]
    save_entities("test_facet", test_entities)

    # Load with include_detached to verify all fields preserved
    loaded = load_entities("test_facet", include_detached=True)
    assert len(loaded) == 1

    alice = loaded[0]
    assert alice["name"] == "Alice"
    assert alice["description"] == "Test person"
    assert alice["attached_at"] == 1700000000000
    assert alice["updated_at"] == 1700000001000
    assert alice["aka"] == ["Ali", "AJ"]
    assert alice["tags"] == ["friend", "colleague"]
    assert alice["custom_field"] == "custom_value"
    assert alice["detached"] is True


def test_detached_flag_for_detected_entities_not_filtered(fixture_journal, tmp_path):
    """Test that include_detached only affects attached entities, not detected."""
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create detected entity for a specific day
    detected_entities = [
        {"type": "Person", "name": "Alice", "description": "Detected person"},
    ]
    save_entities("test_facet", detected_entities, "20250101")

    # Load detected entities - should always return all (no detached filtering for detected)
    loaded = load_entities("test_facet", "20250101")
    assert len(loaded) == 1

    # include_detached should have no effect on detected entities
    loaded_with_flag = load_entities("test_facet", "20250101", include_detached=True)
    assert len(loaded_with_flag) == 1


# Tests for entity memory utilities


def test_entity_slug_basic():
    """Test basic name slug generation."""
    assert entity_slug("Alice Johnson") == "alice_johnson"
    assert entity_slug("Acme Corp") == "acme_corp"
    assert entity_slug("PostgreSQL") == "postgresql"


def test_entity_slug_special_chars():
    """Test slug generation with special characters."""
    assert entity_slug("O'Brien") == "o_brien"
    assert entity_slug("AT&T") == "at_t"
    assert entity_slug("C++") == "c"


def test_entity_slug_unicode():
    """Test slug generation with unicode names."""
    assert entity_slug("José García") == "jose_garcia"
    assert entity_slug("Müller") == "muller"
    # Chinese characters are transliterated to pinyin by python-slugify
    assert entity_slug("北京") == "bei_jing"


def test_entity_slug_whitespace():
    """Test slug generation handles various whitespace."""
    assert entity_slug("  Spaced  Out  ") == "spaced_out"
    assert entity_slug("Tab\tSeparated") == "tab_separated"
    assert entity_slug("New\nLine") == "new_line"


def test_entity_slug_empty():
    """Test slug generation with empty/blank names."""
    assert entity_slug("") == ""
    assert entity_slug("   ") == ""
    assert entity_slug(None) == ""  # type: ignore


def test_entity_slug_long():
    """Test slug generation with very long names."""
    long_name = "A" * 300
    slug = entity_slug(long_name)
    # Should be truncated with hash suffix
    assert len(slug) <= 200
    assert "_" in slug[-9:]  # Hash suffix pattern


def test_entity_memory_path(fixture_journal, tmp_path):
    """Test entity memory path generation."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    path = entity_memory_path("personal", "Alice Johnson")
    expected = tmp_path / "facets" / "personal" / "entities" / "alice_johnson"
    assert path == expected


def test_entity_memory_path_empty_name(fixture_journal, tmp_path):
    """Test entity memory path with empty name raises ValueError."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    with pytest.raises(ValueError, match="slugifies to empty string"):
        entity_memory_path("personal", "")


def test_ensure_entity_memory(fixture_journal, tmp_path):
    """Test entity memory folder creation."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    folder = ensure_entity_memory("personal", "Bob Smith")
    assert folder.exists()
    assert folder.is_dir()
    assert folder == tmp_path / "facets" / "personal" / "entities" / "bob_smith"


def test_ensure_entity_memory_idempotent(fixture_journal, tmp_path):
    """Test that ensure_entity_memory is idempotent."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    folder1 = ensure_entity_memory("personal", "Charlie Brown")
    folder2 = ensure_entity_memory("personal", "Charlie Brown")
    assert folder1 == folder2
    assert folder1.exists()


def test_rename_entity_memory(fixture_journal, tmp_path):
    """Test renaming entity memory folder."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create original folder
    old_folder = ensure_entity_memory("work", "Alice Johnson")
    assert old_folder.exists()

    # Create a file inside to verify contents are moved
    (old_folder / "notes.md").write_text("Test notes")

    # Rename
    result = rename_entity_memory("work", "Alice Johnson", "Alice Smith")
    assert result is True

    # Old folder should not exist
    assert not old_folder.exists()

    # New folder should exist with contents
    new_folder = tmp_path / "facets" / "work" / "entities" / "alice_smith"
    assert new_folder.exists()
    assert (new_folder / "notes.md").read_text() == "Test notes"


def test_rename_entity_memory_not_exists(fixture_journal, tmp_path):
    """Test renaming non-existent folder returns False."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    result = rename_entity_memory("work", "NonExistent", "NewName")
    assert result is False


def test_rename_entity_memory_same_normalized(fixture_journal, tmp_path):
    """Test renaming when normalized names are the same."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create folder
    ensure_entity_memory("work", "Alice Johnson")

    # Rename with different casing (normalizes to same)
    result = rename_entity_memory("work", "Alice Johnson", "alice johnson")
    assert result is False  # No rename needed


def test_rename_entity_memory_target_exists(fixture_journal, tmp_path):
    """Test renaming when target folder already exists raises OSError."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create both folders
    ensure_entity_memory("work", "Alice")
    ensure_entity_memory("work", "Bob")

    # Try to rename Alice to Bob
    with pytest.raises(OSError, match="already exists"):
        rename_entity_memory("work", "Alice", "Bob")


# Tests for find_matching_attached_entity


def test_find_matching_attached_entity_exact_name():
    """Test exact name matching."""
    attached = [
        {"name": "Alice Johnson", "type": "Person"},
        {"name": "Bob Smith", "type": "Person"},
    ]
    result = find_matching_attached_entity("Alice Johnson", attached)
    assert result is not None
    assert result["name"] == "Alice Johnson"


def test_find_matching_attached_entity_exact_aka():
    """Test exact aka matching."""
    attached = [
        {"name": "Robert Johnson", "type": "Person", "aka": ["Bob", "Bobby"]},
    ]
    result = find_matching_attached_entity("Bob", attached)
    assert result is not None
    assert result["name"] == "Robert Johnson"


def test_find_matching_attached_entity_case_insensitive():
    """Test case-insensitive matching."""
    attached = [
        {"name": "Alice Johnson", "type": "Person"},
    ]
    result = find_matching_attached_entity("alice johnson", attached)
    assert result is not None
    assert result["name"] == "Alice Johnson"


def test_find_matching_attached_entity_case_insensitive_aka():
    """Test case-insensitive aka matching."""
    attached = [
        {"name": "Robert Johnson", "type": "Person", "aka": ["Bob"]},
    ]
    result = find_matching_attached_entity("bob", attached)
    assert result is not None
    assert result["name"] == "Robert Johnson"


def test_find_matching_attached_entity_normalized():
    """Test normalized (slugified) matching."""
    attached = [
        {"name": "José García", "type": "Person"},
    ]
    # "Jose Garcia" should match via normalization
    result = find_matching_attached_entity("Jose Garcia", attached)
    assert result is not None
    assert result["name"] == "José García"


def test_find_matching_attached_entity_first_word_unambiguous():
    """Test first-word matching when unambiguous."""
    attached = [
        {"name": "Sarah Chen", "type": "Person"},
        {"name": "Bob Smith", "type": "Person"},
    ]
    # "Sarah" should match "Sarah Chen" (only one Sarah)
    result = find_matching_attached_entity("Sarah", attached)
    assert result is not None
    assert result["name"] == "Sarah Chen"


def test_find_matching_attached_entity_first_word_ambiguous():
    """Test first-word matching skipped when ambiguous."""
    attached = [
        {"name": "John Smith", "type": "Person"},
        {"name": "John Doe", "type": "Person"},
    ]
    # "John" matches multiple entities - should not match
    result = find_matching_attached_entity("John", attached)
    assert result is None


def test_find_matching_attached_entity_first_word_too_short():
    """Test first-word matching requires minimum 3 characters."""
    attached = [
        {"name": "Al Smith", "type": "Person"},
    ]
    # "Al" is too short (< 3 chars)
    result = find_matching_attached_entity("Al", attached)
    assert result is None


def test_find_matching_attached_entity_fuzzy():
    """Test fuzzy matching catches typos."""
    attached = [
        {"name": "Robert Johnson", "type": "Person"},
    ]
    # Typo: "Robet Johnson" should match "Robert Johnson"
    result = find_matching_attached_entity("Robet Johnson", attached)
    assert result is not None
    assert result["name"] == "Robert Johnson"


def test_find_matching_attached_entity_fuzzy_word_order():
    """Test fuzzy matching handles word order differences."""
    attached = [
        {"name": "Sarah Chen", "type": "Person"},
    ]
    # Different word order
    result = find_matching_attached_entity("Chen Sarah", attached)
    assert result is not None
    assert result["name"] == "Sarah Chen"


def test_find_matching_attached_entity_no_match():
    """Test no match returns None."""
    attached = [
        {"name": "Alice Johnson", "type": "Person"},
    ]
    result = find_matching_attached_entity("Charlie Brown", attached)
    assert result is None


def test_find_matching_attached_entity_empty_inputs():
    """Test empty inputs return None."""
    assert find_matching_attached_entity("", []) is None
    assert find_matching_attached_entity("Alice", []) is None
    assert find_matching_attached_entity("", [{"name": "Alice"}]) is None


# Tests for touch_entity


def test_touch_entity_updates_last_seen(fixture_journal, tmp_path):
    """Test touch_entity updates last_seen on attached entity."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create attached entity without last_seen
    entities = [
        {"type": "Person", "name": "Alice Johnson", "description": "Test"},
    ]
    save_entities("test_facet", entities)

    # Touch the entity
    result = touch_entity("test_facet", "Alice Johnson", "20250115")
    assert result == "updated"

    # Verify last_seen was set
    loaded = load_entities("test_facet")
    alice = next(e for e in loaded if e["name"] == "Alice Johnson")
    assert alice["last_seen"] == "20250115"


def test_touch_entity_updates_only_if_more_recent(fixture_journal, tmp_path):
    """Test touch_entity only updates if day is more recent."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create attached entity with existing last_seen
    entities = [
        {
            "type": "Person",
            "name": "Alice Johnson",
            "description": "Test",
            "last_seen": "20250115",
        },
    ]
    save_entities("test_facet", entities)

    # Try to touch with older day
    result = touch_entity("test_facet", "Alice Johnson", "20250110")
    assert result == "skipped"  # Entity found but not updated

    # Verify last_seen was NOT updated (still 20250115)
    loaded = load_entities("test_facet")
    alice = next(e for e in loaded if e["name"] == "Alice Johnson")
    assert alice["last_seen"] == "20250115"

    # Touch with newer day
    result = touch_entity("test_facet", "Alice Johnson", "20250120")
    assert result == "updated"

    # Verify last_seen was updated
    loaded = load_entities("test_facet")
    alice = next(e for e in loaded if e["name"] == "Alice Johnson")
    assert alice["last_seen"] == "20250120"


def test_touch_entity_not_found(fixture_journal, tmp_path):
    """Test touch_entity returns False when entity not found."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create attached entity
    entities = [
        {"type": "Person", "name": "Alice Johnson", "description": "Test"},
    ]
    save_entities("test_facet", entities)

    # Try to touch non-existent entity
    result = touch_entity("test_facet", "Charlie Brown", "20250115")
    assert result == "not_found"


def test_touch_entity_skips_detached(fixture_journal, tmp_path):
    """Test touch_entity skips detached entities."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create detached entity
    entities = [
        {
            "type": "Person",
            "name": "Alice Johnson",
            "description": "Test",
            "detached": True,
        },
    ]
    save_entities("test_facet", entities)

    # Try to touch detached entity
    result = touch_entity("test_facet", "Alice Johnson", "20250115")
    assert result == "not_found"


# Tests for fuzzy exclusion in load_detected_entities_recent


def test_load_detected_entities_recent_fuzzy_exclusion(fixture_journal, tmp_path):
    """Test that fuzzy matching excludes detected entities matching attached."""
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create attached entity
    attached = [
        {
            "type": "Person",
            "name": "Robert Johnson",
            "description": "Attached",
            "aka": ["Bob"],
        },
    ]
    save_entities("test_facet", attached)

    # Create detected entities including variations
    detected_entities = [
        {"type": "Person", "name": "Robert Johnson", "description": "Exact match"},
        {"type": "Person", "name": "Bob", "description": "Aka match"},
        {"type": "Person", "name": "robert johnson", "description": "Case insensitive"},
        {
            "type": "Person",
            "name": "Charlie Brown",
            "description": "Should be included",
        },
    ]
    save_entities("test_facet", detected_entities, "20250101")

    # Load detected - only Charlie Brown should be included
    detected = load_detected_entities_recent("test_facet", days=36500)
    names = [e["name"] for e in detected]

    assert "Robert Johnson" not in names  # Exact match excluded
    assert "Bob" not in names  # Aka excluded
    assert "robert johnson" not in names  # Case insensitive excluded
    assert "Charlie Brown" in names  # Not matched, included


def test_load_detected_entities_recent_first_word_exclusion(fixture_journal, tmp_path):
    """Test that first-word matching excludes detected entities."""
    facet_path = tmp_path / "facets" / "test_facet"
    entities_dir = facet_path / "entities"
    entities_dir.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create attached entity
    attached = [
        {"type": "Person", "name": "Sarah Chen", "description": "Attached"},
    ]
    save_entities("test_facet", attached)

    # Create detected entities
    detected_entities = [
        {"type": "Person", "name": "Sarah", "description": "First word match"},
        {
            "type": "Person",
            "name": "Charlie Brown",
            "description": "Should be included",
        },
    ]
    save_entities("test_facet", detected_entities, "20250101")

    # Load detected - Sarah should be excluded (first word of Sarah Chen)
    detected = load_detected_entities_recent("test_facet", days=36500)
    names = [e["name"] for e in detected]

    assert "Sarah" not in names  # First word excluded
    assert "Charlie Brown" in names  # Not matched, included


# Tests for parse_knowledge_graph_entities


def test_parse_knowledge_graph_entities(tmp_path):
    """Test parsing entity names from knowledge graph markdown."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create a knowledge graph file
    day_dir = tmp_path / "20260108" / "insights"
    day_dir.mkdir(parents=True)

    kg_content = """# Knowledge Graph Report

## 1. Entity Extraction

### People
| Entity Name | Type | First Appearance |
| :--- | :--- | :--- |
| **Alice Johnson** | Person | 09:00 |
| **Bob Smith** | Person | 10:00 |

### Projects
| Entity Name | Type | First Appearance |
| :--- | :--- | :--- |
| **Project Alpha** | Project | 11:00 |

## 2. Relationship Mapping

| Source Name | Target Name | Relationship Type |
| :--- | :--- | :--- |
| **Alice Johnson** | **Project Alpha** | `works-on` |
| **Bob Smith** | **Alice Johnson** | `collaborates-with` |
"""
    (day_dir / "knowledge_graph.md").write_text(kg_content)

    # Parse entities
    entities = parse_knowledge_graph_entities("20260108")

    assert "Alice Johnson" in entities
    assert "Bob Smith" in entities
    assert "Project Alpha" in entities
    assert len(entities) == 3  # Unique names only


def test_parse_knowledge_graph_entities_missing_file(tmp_path):
    """Test parsing returns empty list when KG doesn't exist."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    entities = parse_knowledge_graph_entities("20260108")
    assert entities == []


def test_parse_knowledge_graph_entities_empty_file(tmp_path):
    """Test parsing returns empty list for empty KG."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    day_dir = tmp_path / "20260108" / "insights"
    day_dir.mkdir(parents=True)
    (day_dir / "knowledge_graph.md").write_text("")

    entities = parse_knowledge_graph_entities("20260108")
    assert entities == []


# Tests for touch_entities_from_activity


def test_touch_entities_from_activity_basic(tmp_path):
    """Test updating last_seen from activity names."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create attached entities
    attached = [
        {"type": "Person", "name": "Alice Johnson", "description": "Test"},
        {
            "type": "Person",
            "name": "Robert Smith",
            "description": "Test",
            "aka": ["Bob"],
        },
    ]
    save_entities("test_facet", attached)

    # Touch from activity names
    result = touch_entities_from_activity(
        "test_facet", ["Alice Johnson", "Bob", "Unknown Person"], "20260108"
    )

    # Alice matched exactly, Bob matched via aka
    assert len(result["matched"]) == 2
    assert ("Alice Johnson", "Alice Johnson") in result["matched"]
    assert ("Bob", "Robert Smith") in result["matched"]

    # Both should be updated
    assert "Alice Johnson" in result["updated"]
    assert "Robert Smith" in result["updated"]

    # Verify last_seen was set
    entities = load_entities("test_facet")
    alice = next(e for e in entities if e["name"] == "Alice Johnson")
    bob = next(e for e in entities if e["name"] == "Robert Smith")
    assert alice["last_seen"] == "20260108"
    assert bob["last_seen"] == "20260108"


def test_touch_entities_from_activity_empty_names(tmp_path):
    """Test with empty names list."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    attached = [{"type": "Person", "name": "Alice", "description": "Test"}]
    save_entities("test_facet", attached)

    result = touch_entities_from_activity("test_facet", [], "20260108")

    assert result["matched"] == []
    assert result["updated"] == []
    assert result["skipped"] == []


def test_touch_entities_from_activity_no_attached(tmp_path):
    """Test with no attached entities."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    result = touch_entities_from_activity("test_facet", ["Alice"], "20260108")

    assert result["matched"] == []
    assert result["updated"] == []
    assert result["skipped"] == []


def test_touch_entities_from_activity_deduplicates(tmp_path):
    """Test that same entity matched multiple times is only updated once."""
    facet_path = tmp_path / "facets" / "test_facet"
    facet_path.mkdir(parents=True)
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    attached = [
        {
            "type": "Person",
            "name": "Robert Smith",
            "description": "Test",
            "aka": ["Bob"],
        },
    ]
    save_entities("test_facet", attached)

    # Both names map to same entity
    result = touch_entities_from_activity(
        "test_facet", ["Robert Smith", "Bob"], "20260108"
    )

    # Two matches but only one unique entity updated
    assert len(result["matched"]) == 2
    assert len(result["updated"]) == 1
    assert "Robert Smith" in result["updated"]


# Tests for entity observations


def test_observations_file_path(fixture_journal, tmp_path):
    """Test observations file path generation."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    path = observations_file_path("personal", "Alice Johnson")
    expected = (
        tmp_path
        / "facets"
        / "personal"
        / "entities"
        / "alice_johnson"
        / "observations.jsonl"
    )
    assert path == expected


def test_load_observations_empty(fixture_journal, tmp_path):
    """Test loading observations for entity with no observations."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # No file exists yet
    observations = load_observations("personal", "Alice Johnson")
    assert observations == []


def test_save_and_load_observations(fixture_journal, tmp_path):
    """Test saving and loading observations."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save observations
    test_observations = [
        {
            "content": "Prefers morning meetings",
            "observed_at": 1700000000000,
            "source_day": "20250113",
        },
        {"content": "Expert in Kubernetes", "observed_at": 1700000001000},
    ]
    save_observations("personal", "Alice Johnson", test_observations)

    # Load them back
    loaded = load_observations("personal", "Alice Johnson")
    assert len(loaded) == 2
    assert loaded[0]["content"] == "Prefers morning meetings"
    assert loaded[0]["observed_at"] == 1700000000000
    assert loaded[0]["source_day"] == "20250113"
    assert loaded[1]["content"] == "Expert in Kubernetes"


def test_add_observation_success(fixture_journal, tmp_path):
    """Test adding observation with correct guard."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # First observation (observation_number=1 for empty list)
    result = add_observation(
        "personal", "Alice", "Prefers async communication", 1, "20250113"
    )
    assert result["count"] == 1
    assert len(result["observations"]) == 1
    assert result["observations"][0]["content"] == "Prefers async communication"
    assert result["observations"][0]["source_day"] == "20250113"
    assert "observed_at" in result["observations"][0]

    # Second observation (observation_number=2)
    result = add_observation("personal", "Alice", "Works PST timezone", 2)
    assert result["count"] == 2
    assert len(result["observations"]) == 2

    # Verify persistence
    loaded = load_observations("personal", "Alice")
    assert len(loaded) == 2


def test_add_observation_guard_failure(fixture_journal, tmp_path):
    """Test adding observation with wrong guard fails."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # First observation
    add_observation("personal", "Alice", "First observation", 1)

    # Try to add with wrong observation_number (should be 2, not 1)
    with pytest.raises(ObservationNumberError) as exc_info:
        add_observation("personal", "Alice", "Second observation", 1)

    assert exc_info.value.expected == 2
    assert exc_info.value.actual == 1


def test_add_observation_empty_content(fixture_journal, tmp_path):
    """Test adding observation with empty content fails."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    with pytest.raises(ValueError, match="cannot be empty"):
        add_observation("personal", "Alice", "", 1)

    with pytest.raises(ValueError, match="cannot be empty"):
        add_observation("personal", "Alice", "   ", 1)


def test_observations_with_entity_rename(fixture_journal, tmp_path):
    """Test that observations are preserved when entity memory folder is renamed."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Create entity memory folder and add observations
    ensure_entity_memory("work", "Alice Johnson")
    add_observation("work", "Alice Johnson", "Test observation", 1)

    # Verify observation exists
    observations = load_observations("work", "Alice Johnson")
    assert len(observations) == 1

    # Rename entity memory folder
    result = rename_entity_memory("work", "Alice Johnson", "Alice Smith")
    assert result is True

    # Old name should have no observations (folder moved)
    old_observations = load_observations("work", "Alice Johnson")
    assert old_observations == []

    # New name should have observations
    new_observations = load_observations("work", "Alice Smith")
    assert len(new_observations) == 1
    assert new_observations[0]["content"] == "Test observation"


def test_observations_atomic_write(fixture_journal, tmp_path):
    """Test that observations are written atomically."""
    os.environ["JOURNAL_PATH"] = str(tmp_path)

    # Save observations
    test_observations = [
        {"content": "Test 1", "observed_at": 1700000000000},
        {"content": "Test 2", "observed_at": 1700000001000},
    ]
    save_observations("personal", "Bob", test_observations)

    # Verify file exists at expected location
    path = observations_file_path("personal", "Bob")
    assert path.exists()

    # Verify JSONL format
    import json

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        assert json.loads(line)  # Should not raise
