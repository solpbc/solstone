"""Integration tests for the domain-scoped entities indexer."""

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from think.indexer import (
    reset_index,
    scan_entities,
    search_entities,
)


@pytest.fixture
def test_env():
    """Set up test environment with fixtures journal and temporary index."""
    journal_path = Path(__file__).parent.parent.parent / "fixtures" / "journal"

    if not journal_path.exists():
        pytest.skip("fixtures/journal not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        old_journal = os.environ.get("JOURNAL_PATH")
        os.environ["JOURNAL_PATH"] = str(journal_path)

        test_index_dir = Path(tmpdir) / "indexer"

        # Monkey-patch get_index to use temporary directory
        import think.indexer.core

        old_get_index = think.indexer.core.get_index

        def test_get_index(*, index, journal=None, day=None):
            db_name = think.indexer.core.DB_NAMES[index]
            db_path = test_index_dir / db_name
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path))
            # Ensure schema is created
            for statement in think.indexer.core.SCHEMAS[index]:
                conn.execute(statement)
            return conn, str(db_path)

        think.indexer.core.get_index = test_get_index

        try:
            # Reset and scan entities
            reset_index(str(journal_path), "entities")
            scan_entities(str(journal_path))

            yield journal_path

        finally:
            # Restore original environment
            if old_journal:
                os.environ["JOURNAL_PATH"] = old_journal
            elif "JOURNAL_PATH" in os.environ:
                del os.environ["JOURNAL_PATH"]

            think.indexer.core.get_index = old_get_index


@pytest.mark.integration
def test_scan_finds_attached_entities(test_env):
    """Test that scanning finds attached entities from domains/*/entities.md."""
    # Search for Alice Johnson (attached entity in personal domain)
    total, results = search_entities("Alice Johnson")
    assert total >= 1, "Should find at least one result for Alice Johnson"

    alice = next((r for r in results if "Alice" in r["metadata"]["name"]), None)
    assert alice is not None, "Should find Alice Johnson"
    assert alice["metadata"]["domain"] == "personal"
    assert alice["metadata"]["attached"] is True
    assert alice["metadata"]["day"] is None


@pytest.mark.integration
def test_scan_finds_detected_entities(test_env):
    """Test that scanning finds detected entities from domains/*/entities/*.md."""
    # Search for Charlie Brown (detected entity in personal/entities/20250101.md)
    total, results = search_entities("Charlie Brown")
    assert total >= 1, "Should find at least one result for Charlie Brown"

    charlie = next((r for r in results if "Charlie" in r["metadata"]["name"]), None)
    assert charlie is not None, "Should find Charlie Brown"
    assert charlie["metadata"]["domain"] == "personal"
    assert charlie["metadata"]["attached"] is False
    assert charlie["metadata"]["day"] == "20250101"


@pytest.mark.integration
def test_attached_flag_correct(test_env):
    """Test that attached entities have attached=1, detected have attached=0."""
    # Alice is attached
    total, results = search_entities("Alice", attached=True)
    assert total >= 1, "Should find attached entity Alice"
    alice = results[0]
    assert alice["metadata"]["attached"] is True
    assert alice["metadata"]["day"] is None

    # Charlie is detected
    total, results = search_entities("Charlie", attached=False)
    assert total >= 1, "Should find detected entity Charlie"
    charlie = results[0]
    assert charlie["metadata"]["attached"] is False
    assert charlie["metadata"]["day"] == "20250101"


@pytest.mark.integration
def test_search_filter_domain(test_env):
    """Test that search_entities(domain='personal') filters to domain."""
    total, results = search_entities("", domain="personal", limit=100)

    # Should find personal entities but not others
    domains = {r["metadata"]["domain"] for r in results}
    assert "personal" in domains
    assert domains == {"personal"}, f"Should only find personal domain, got {domains}"

    # Verify we found some personal entities
    assert total >= 3, "Should find at least 3 personal entities"


@pytest.mark.integration
def test_search_filter_day(test_env):
    """Test that search_entities(day='20250101') filters to day."""
    total, results = search_entities("", day="20250101", limit=100)

    # Should only find entities from that day
    days = {r["metadata"]["day"] for r in results}
    assert days == {"20250101"}, f"Should only find day 20250101, got {days}"

    # Should find both Charlie (personal) and pytest (full-featured)
    assert total >= 2, "Should find at least 2 entities from 20250101"


@pytest.mark.integration
def test_search_filter_attached(test_env):
    """Test that search_entities(attached=True/False) filters correctly."""
    # Search for only attached entities
    total_attached, results_attached = search_entities("", attached=True, limit=100)
    assert total_attached >= 3, "Should find at least 3 attached entities"
    for r in results_attached:
        assert r["metadata"]["attached"] is True
        assert r["metadata"]["day"] is None

    # Search for only detected entities
    total_detected, results_detected = search_entities("", attached=False, limit=100)
    assert total_detected >= 3, "Should find at least 3 detected entities"
    for r in results_detected:
        assert r["metadata"]["attached"] is False
        assert r["metadata"]["day"] is not None


@pytest.mark.integration
def test_search_filter_etype(test_env):
    """Test that search_entities(etype='Person') filters by type."""
    total, results = search_entities("", etype="Person", limit=100)

    # Should only find Person type entities
    types = {r["metadata"]["type"] for r in results}
    assert types == {"Person"}, f"Should only find Person type, got {types}"

    # Should find Alice, Bob, Charlie, Diana
    assert total >= 4, "Should find at least 4 Person entities"


@pytest.mark.integration
def test_search_filter_name(test_env):
    """Test that search_entities(name='Alice') searches name field."""
    total, results = search_entities("", name="Alice", limit=100)

    # Should find Alice Johnson
    assert total >= 1, "Should find at least 1 result for name Alice"
    alice = results[0]
    assert "Alice" in alice["metadata"]["name"]


@pytest.mark.integration
def test_search_combined_filters(test_env):
    """Test combining multiple filters."""
    # Search for Person entities in personal domain
    total, results = search_entities("", domain="personal", etype="Person", limit=100)

    # Should find Alice, Bob, Charlie, Diana (4 people in personal domain)
    assert total >= 4, "Should find at least 4 Person entities in personal domain"

    for r in results:
        assert r["metadata"]["domain"] == "personal"
        assert r["metadata"]["type"] == "Person"


@pytest.mark.integration
def test_fts_search_name_and_description(test_env):
    """Test FTS search finds entities by name and description content."""
    # Search by name
    total, results = search_entities("Alice")
    assert total >= 1, "Should find Alice by name"

    # Search by description
    total, results = search_entities("coffee shop")
    assert total >= 1, "Should find Charlie by description"
    charlie = next((r for r in results if "coffee" in r["text"]), None)
    assert charlie is not None, "Should find entity with coffee shop description"


@pytest.mark.integration
def test_cross_domain_independence(test_env):
    """Test that same entity name can exist in multiple domains independently."""
    # Acme Corp exists in personal domain
    total, results = search_entities("Acme", domain="personal")
    assert total >= 1, "Should find Acme Corp in personal domain"
    personal_acme = results[0]
    assert personal_acme["metadata"]["domain"] == "personal"


@pytest.mark.integration
def test_result_format(test_env):
    """Test that search results have correct structure."""
    total, results = search_entities("Alice", limit=5)

    assert total > 0, "Should find results"
    assert len(results) > 0, "Should return results"

    result = results[0]

    # Check required fields
    assert "id" in result
    assert "text" in result
    assert "metadata" in result
    assert "score" in result

    # Check metadata structure
    metadata = result["metadata"]
    assert "domain" in metadata
    assert "day" in metadata  # Can be None
    assert "type" in metadata
    assert "name" in metadata
    assert "attached" in metadata

    # Check ID format
    # Format: {domain}/entities.md:{name} or {domain}/entities/{day}.md:{name}
    assert "/" in result["id"]
    assert ":" in result["id"]


@pytest.mark.integration
def test_order_by_rank(test_env):
    """Test that order='rank' sorts by BM25 relevance."""
    total, results = search_entities("person", order="rank", limit=10)

    # Should return results ordered by relevance
    assert len(results) > 0, "Should find results"

    # Scores should be in ascending order (BM25 scores are negative, smaller = better)
    scores = [r["score"] for r in results]
    assert scores == sorted(
        scores
    ), "Scores should be in ascending order (smaller BM25 = better)"


@pytest.mark.integration
def test_order_by_day(test_env):
    """Test that order='day' sorts chronologically."""
    # Search only detected entities to ensure they all have days
    total, results = search_entities("", attached=False, order="day", limit=10)

    assert len(results) > 0, "Should find detected entities"

    # Days should be in descending order (newest first)
    days = [r["metadata"]["day"] for r in results if r["metadata"]["day"]]
    if len(days) > 1:
        assert days == sorted(
            days, reverse=True
        ), f"Days should be in descending order, got {days}"


@pytest.mark.integration
def test_limit_and_offset(test_env):
    """Test that limit and offset parameters work correctly."""
    # Get first page
    total, page1 = search_entities("", limit=2, offset=0)

    assert total >= 4, "Should have enough results to test pagination"
    assert len(page1) == 2, "First page should have 2 results"

    # Get second page
    _, page2 = search_entities("", limit=2, offset=2)
    assert len(page2) == 2, "Second page should have 2 results"

    # Pages should be different
    page1_ids = {r["id"] for r in page1}
    page2_ids = {r["id"] for r in page2}
    assert page1_ids.isdisjoint(page2_ids), "Pages should not overlap"
