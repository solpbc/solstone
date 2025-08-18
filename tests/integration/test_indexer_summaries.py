"""Integration tests for the summaries indexer."""

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from think.indexer import (
    get_index,
    reset_index,
    scan_summaries,
    search_summaries,
)


@pytest.mark.integration
def test_summaries_indexer_scan_and_search():
    """Test scanning and searching summary files from fixtures."""
    # Use fixtures journal path
    journal_path = Path(__file__).parent.parent.parent / "fixtures" / "journal"

    if not journal_path.exists():
        pytest.skip("fixtures/journal not found")

    # Create a temporary index directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set environment to use temp index location
        old_journal = os.environ.get("JOURNAL_PATH")
        os.environ["JOURNAL_PATH"] = str(journal_path)

        try:
            # Get the index database
            index_path = Path(tmpdir) / "indexer" / "summaries.sqlite"
            index_path.parent.mkdir(parents=True, exist_ok=True)

            # Override the index path for testing
            original_index_dir = Path(journal_path) / "indexer"
            test_index_dir = Path(tmpdir) / "indexer"

            # Monkey-patch the get_index function to use our temp directory
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

            # Reset index to ensure clean state
            reset_index(str(journal_path), "summaries")

            # Scan the summaries
            scan_count = scan_summaries(str(journal_path))

            # We should have scanned some summary files
            assert scan_count > 0, f"Expected to scan summary files, got {scan_count}"

            # Search for specific terms we know are in the fixtures

            # Search for "authentication" (from flow.md)
            total, results = search_summaries("authentication")
            assert total > 0, "Should find results for 'authentication'"
            assert len(results) > 0, "Should have actual results for 'authentication'"
            # Verify the result contains expected content
            found_auth = False
            for result in results:
                if "authentication module" in result["text"].lower():
                    found_auth = True
                    break
            assert found_auth, "Should find 'authentication module' in results"

            # Search for "sprint planning" (from flow.md)
            total, results = search_summaries("sprint planning")
            assert total > 0, "Should find results for 'sprint planning'"

            # Search for "Docker" (from day 2)
            total, results = search_summaries("Docker")
            assert total > 0, "Should find results for 'Docker'"

            # Search for "FastAPI" (from day 2)
            total, results = search_summaries("FastAPI")
            assert total > 0, "Should find results for 'FastAPI'"

            # Verify results have expected structure
            if results:
                result = results[0]
                assert "text" in result
                assert "metadata" in result
                assert "score" in result
                assert "day" in result["metadata"]
                assert "topic" in result["metadata"]
                assert "path" in result["metadata"]
                assert "index" in result["metadata"]

        finally:
            # Restore original environment
            if old_journal:
                os.environ["JOURNAL_PATH"] = old_journal
            elif "JOURNAL_PATH" in os.environ:
                del os.environ["JOURNAL_PATH"]

            # Restore original get_index function
            think.indexer.core.get_index = old_get_index


@pytest.mark.integration
def test_summaries_indexer_rescan():
    """Test that rescanning summaries updates the index properly."""
    journal_path = Path(__file__).parent.parent.parent / "fixtures" / "journal"

    if not journal_path.exists():
        pytest.skip("fixtures/journal not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        old_journal = os.environ.get("JOURNAL_PATH")
        os.environ["JOURNAL_PATH"] = str(journal_path)

        try:
            # Set up test index
            test_index_dir = Path(tmpdir) / "indexer"

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

            # Reset and scan initially
            reset_index(str(journal_path), "summaries")
            first_scan = scan_summaries(str(journal_path))
            assert first_scan > 0

            # Search for initial content
            total1, results1 = search_summaries("authentication")
            initial_count = total1

            # Rescan (should handle existing content gracefully)
            second_scan = scan_summaries(str(journal_path))

            # Search again - should get same results
            total2, results2 = search_summaries("authentication")
            assert total2 == initial_count, "Rescan should not duplicate entries"

        finally:
            if old_journal:
                os.environ["JOURNAL_PATH"] = old_journal
            elif "JOURNAL_PATH" in os.environ:
                del os.environ["JOURNAL_PATH"]

            think.indexer.core.get_index = old_get_index
