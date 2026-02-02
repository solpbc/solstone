# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the unified journal index."""

import json
import os

import pytest

from think.indexer import sanitize_fts_query


class TestSanitizeFtsQuery:
    """Tests for FTS5 query sanitization."""

    def test_simple_words(self):
        """Simple words pass through unchanged."""
        assert sanitize_fts_query("foo bar baz") == "foo bar baz"

    def test_preserves_or_operator(self):
        """OR operator is preserved."""
        assert sanitize_fts_query("foo OR bar") == "foo OR bar"

    def test_preserves_and_operator(self):
        """AND operator is preserved."""
        assert sanitize_fts_query("foo AND bar") == "foo AND bar"

    def test_preserves_not_operator(self):
        """NOT operator is preserved."""
        assert sanitize_fts_query("foo NOT bar") == "foo NOT bar"

    def test_preserves_asterisk_prefix_match(self):
        """Asterisk for prefix matching is preserved."""
        assert sanitize_fts_query("test*") == "test*"

    def test_preserves_quoted_phrases(self):
        """Quoted phrases are preserved."""
        assert sanitize_fts_query('"public benefit"') == '"public benefit"'

    def test_complex_query_with_or_and_quotes(self):
        """Complex query with OR and quoted phrases."""
        result = sanitize_fts_query('solstone OR pbc OR "public benefit"')
        assert result == 'solstone OR pbc OR "public benefit"'

    def test_dot_replaced_with_space(self):
        """Dots are replaced with spaces."""
        assert sanitize_fts_query("config.json") == "config json"

    def test_colon_replaced_with_space(self):
        """Colons are replaced with spaces."""
        assert sanitize_fts_query("foo:bar") == "foo bar"

    def test_special_chars_replaced_with_space(self):
        """Various special characters are replaced with spaces."""
        assert sanitize_fts_query("a@b#c$d") == "a b c d"

    def test_preserves_apostrophe(self):
        """Apostrophes in contractions are preserved."""
        assert sanitize_fts_query("what's up") == "what's up"

    def test_unbalanced_quote_removed(self):
        """Unbalanced quotes are removed entirely."""
        assert sanitize_fts_query('"unbalanced') == "unbalanced"

    def test_unbalanced_quote_removes_all(self):
        """When quotes are unbalanced, all quotes are removed."""
        assert sanitize_fts_query('foo "bar" baz "qux') == "foo bar baz qux"

    def test_balanced_quotes_preserved(self):
        """Balanced quotes are kept."""
        assert sanitize_fts_query('"foo" "bar"') == '"foo" "bar"'


@pytest.fixture
def journal_fixture(tmp_path):
    """Create a temporary journal with test data."""
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)

    # Create daily insight
    day = journal / "20240101"
    day.mkdir()
    agents_dir = day / "agents"
    agents_dir.mkdir()
    (agents_dir / "flow.md").write_text("# Flow Summary\n\nWorked on project alpha.\n")

    # Create segment with audio transcript
    segment = day / "100000_300"
    segment.mkdir()
    (segment / "audio.jsonl").write_text(
        json.dumps({"topics": ["test"], "setting": "personal"})
        + "\n"
        + json.dumps(
            {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "hello world"}
        )
        + "\n"
    )

    # Create segment markdown
    (segment / "screen.md").write_text("# Screen Summary\n\nViewed documentation.\n")

    # Create facet events
    events_dir = journal / "facets" / "work" / "events"
    events_dir.mkdir(parents=True)
    event = {
        "type": "meeting",
        "start": "09:00:00",
        "end": "09:30:00",
        "title": "Standup",
        "summary": "Daily sync meeting",
        "facet": "work",
        "topic": "meetings",
        "occurred": True,
    }
    (events_dir / "20240101.jsonl").write_text(json.dumps(event))

    # Create facet entities
    entities_dir = journal / "facets" / "work" / "entities"
    entities_dir.mkdir(parents=True)
    entity = {
        "name": "Project Alpha",
        "type": "project",
        "description": "Main project",
    }
    (entities_dir / "20240101.jsonl").write_text(json.dumps(entity))

    # Create facet news
    news_dir = journal / "facets" / "work" / "news"
    news_dir.mkdir(parents=True)
    (news_dir / "20240101.md").write_text(
        "# News\n\nImportant update about the project.\n"
    )

    return journal


def test_scan_journal(journal_fixture):
    """Test scanning journal creates index."""
    from think.indexer.journal import scan_journal

    changed = scan_journal(str(journal_fixture), verbose=True)
    assert changed is True

    # Index file should exist
    index_path = journal_fixture / "indexer" / "journal.sqlite"
    assert index_path.exists()


def test_search_journal_outputs(journal_fixture):
    """Test searching returns agent output chunks."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    total, results = search_journal("project alpha")
    assert total >= 1
    # Should find the flow output mentioning "project alpha"
    found = any("alpha" in r["text"].lower() for r in results)
    assert found


def test_search_journal_events(journal_fixture):
    """Test searching returns event chunks."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    total, results = search_journal("Standup", topic="event")
    assert total >= 1
    assert any("Standup" in r["text"] for r in results)


def test_search_journal_filter_by_day(journal_fixture):
    """Test filtering search by day."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    # Search with day filter
    total, results = search_journal("", day="20240101")
    assert total >= 1
    for r in results:
        assert r["metadata"]["day"] == "20240101"


def test_search_journal_filter_by_facet(journal_fixture):
    """Test filtering search by facet."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    # Search with facet filter
    total, results = search_journal("", facet="work")
    assert total >= 1
    for r in results:
        assert r["metadata"]["facet"] == "work"


def test_search_journal_filter_by_topic(journal_fixture):
    """Test filtering search by topic."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    # Search events by topic
    total, results = search_journal("", topic="event")
    assert total >= 1
    for r in results:
        assert r["metadata"]["topic"] == "event"


def test_search_journal_facet_case_insensitive(journal_fixture):
    """Test facet filtering is case-insensitive."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    # Search with uppercase facet filter should find lowercase-indexed data
    total_upper, results_upper = search_journal("", facet="WORK")
    total_lower, _ = search_journal("", facet="work")
    total_mixed, _ = search_journal("", facet="Work")

    assert total_upper == total_lower == total_mixed
    assert total_upper >= 1
    # All results should have lowercase facet in metadata
    for r in results_upper:
        assert r["metadata"]["facet"] == "work"


def test_search_journal_topic_case_insensitive(journal_fixture):
    """Test topic filtering is case-insensitive."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    # Search with uppercase topic filter should find lowercase-indexed data
    total_upper, results_upper = search_journal("", topic="EVENT")
    total_lower, _ = search_journal("", topic="event")
    total_mixed, _ = search_journal("", topic="Event")

    assert total_upper == total_lower == total_mixed
    assert total_upper >= 1
    # All results should have lowercase topic in metadata
    for r in results_upper:
        assert r["metadata"]["topic"] == "event"


def test_get_events(journal_fixture):
    """Test get_events returns structured event data."""
    from think.indexer.journal import get_events

    events = get_events("20240101")
    assert len(events) == 1
    assert events[0]["title"] == "Standup"
    assert events[0]["start"] == "09:00:00"
    assert events[0]["facet"] == "work"


def test_get_events_filter_by_facet(journal_fixture):
    """Test get_events with facet filter."""
    from think.indexer.journal import get_events

    # Should find work facet events
    events = get_events("20240101", facet="work")
    assert len(events) == 1

    # Should not find nonexistent facet
    events = get_events("20240101", facet="personal")
    assert len(events) == 0


def test_reset_journal_index(journal_fixture):
    """Test resetting the journal index."""
    from think.indexer.journal import reset_journal_index, scan_journal

    scan_journal(str(journal_fixture))
    index_path = journal_fixture / "indexer" / "journal.sqlite"
    assert index_path.exists()

    reset_journal_index(str(journal_fixture))
    assert not index_path.exists()


def test_index_caching(journal_fixture):
    """Test that unchanged files are not re-indexed."""
    from think.indexer.journal import scan_journal

    # First scan indexes files
    changed = scan_journal(str(journal_fixture))
    assert changed is True

    # Second scan should be a no-op (all cached)
    changed = scan_journal(str(journal_fixture))
    assert changed is False


def test_is_historical_day():
    """Test _is_historical_day helper function."""
    from think.indexer.journal import _is_historical_day

    # Non-day paths are never historical
    assert _is_historical_day("facets/work/events/20240101.jsonl") is False
    assert _is_historical_day("imports/123/summary.md") is False
    assert _is_historical_day("apps/home/agents/foo.md") is False

    # Future dates are not historical
    assert _is_historical_day("29991231/agents/flow.md") is False

    # Path without slash is not historical
    assert _is_historical_day("20240101") is False
    assert _is_historical_day("") is False

    # Day paths before today are historical (tested with a very old date)
    assert _is_historical_day("20000101/agents/flow.md") is True


def test_scan_journal_full_mode(journal_fixture):
    """Test full mode includes all files including historical days."""
    from think.indexer.journal import scan_journal, search_journal

    # Full scan should include everything
    changed = scan_journal(str(journal_fixture), full=True)
    assert changed is True

    # Should find content from historical day
    total, results = search_journal("project alpha")
    assert total >= 1


def test_find_formattable_files(journal_fixture):
    """Test file discovery function."""
    from think.formatters import find_formattable_files

    files = find_formattable_files(str(journal_fixture))

    # Should find various file types
    paths = set(files.keys())

    # Daily agent outputs
    assert "20240101/agents/flow.md" in paths

    # Segment content
    assert "20240101/100000_300/screen.md" in paths
    assert "20240101/100000_300/audio.jsonl" in paths

    # Facet content
    assert "facets/work/events/20240101.jsonl" in paths
    assert "facets/work/entities/20240101.jsonl" in paths
    assert "facets/work/news/20240101.md" in paths


def test_search_journal_empty_query(journal_fixture):
    """Test search with empty query returns all results."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    # Empty query should return all chunks
    total, results = search_journal("")
    assert total > 0


def test_search_journal_pagination(journal_fixture):
    """Test search pagination."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    # Get first page
    total, results1 = search_journal("", limit=2, offset=0)

    # Get second page
    _, results2 = search_journal("", limit=2, offset=2)

    # Results should be different (if enough data)
    if total > 2:
        ids1 = {r["id"] for r in results1}
        ids2 = {r["id"] for r in results2}
        assert ids1 != ids2


def test_search_journal_date_range(journal_fixture):
    """Test filtering search by date range."""
    from think.indexer.journal import scan_journal, search_journal

    scan_journal(str(journal_fixture))

    # Search with date range that includes our test day
    total, results = search_journal("", day_from="20240101", day_to="20240101")
    assert total >= 1
    for r in results:
        assert r["metadata"]["day"] == "20240101"

    # Search with date range that excludes our test day
    total, results = search_journal("", day_from="20240102", day_to="20240105")
    assert total == 0


def test_search_counts_date_range(journal_fixture):
    """Test search_counts with date range filtering."""
    from think.indexer.journal import scan_journal, search_counts

    scan_journal(str(journal_fixture))

    # Counts with date range including test data
    counts = search_counts("", day_from="20240101", day_to="20240101")
    assert counts["total"] >= 1
    assert "20240101" in counts["days"]

    # Counts with date range excluding test data
    counts = search_counts("", day_from="20240102", day_to="20240105")
    assert counts["total"] == 0


def test_mcp_search_journal_returns_counts():
    """Test MCP wrapper returns counts aggregation."""
    from think.tools.search import search_journal

    # Use fixtures journal
    os.environ["JOURNAL_PATH"] = "fixtures/journal"

    result = search_journal("test")

    # Should have counts structure
    assert "counts" in result
    counts = result["counts"]
    assert "facets" in counts
    assert "topics" in counts
    assert "recent_days" in counts
    assert "top_days" in counts
    assert "bucketed_days" in counts

    # recent_days should have 7 entries (including zeros)
    assert len(counts["recent_days"]) == 7


def test_mcp_search_journal_returns_query_echo():
    """Test MCP wrapper returns query echo."""
    from think.tools.search import search_journal

    os.environ["JOURNAL_PATH"] = "fixtures/journal"

    result = search_journal("test query", facet="work", topic="audio")

    assert "query" in result
    assert result["query"]["text"] == "test query"
    assert result["query"]["filters"]["facet"] == "work"
    assert result["query"]["filters"]["topic"] == "audio"


def test_mcp_search_journal_results_include_path():
    """Test MCP wrapper results include path and idx."""
    from think.tools.search import search_journal

    os.environ["JOURNAL_PATH"] = "fixtures/journal"

    result = search_journal("")

    if result.get("results"):
        item = result["results"][0]
        assert "path" in item
        assert "idx" in item


def test_bucket_day_counts():
    """Test day bucketing logic."""
    from datetime import datetime, timedelta

    from think.tools.search import _bucket_day_counts

    today = datetime.now()

    # Create test data with various dates
    day_counts = {}

    # Add recent days (within last 7 days)
    for i in range(3):
        d = (today - timedelta(days=i)).strftime("%Y%m%d")
        day_counts[d] = 5 + i

    # Add older days (more than 7 days ago)
    for i in range(10, 25):
        d = (today - timedelta(days=i)).strftime("%Y%m%d")
        day_counts[d] = 2

    result = _bucket_day_counts(day_counts)

    # recent_days should have 7 entries
    assert len(result["recent_days"]) == 7

    # top_days should have entries
    assert len(result["top_days"]) > 0

    # bucketed_days should have entries for older days
    assert len(result["bucketed_days"]) > 0

    # Bucketed day keys should be in YYYYMMDD-YYYYMMDD format
    for key in result["bucketed_days"]:
        assert "-" in key
        parts = key.split("-")
        assert len(parts) == 2
        assert len(parts[0]) == 8
        assert len(parts[1]) == 8


def test_light_scan_removes_deleted_facet_content(journal_fixture):
    """Test that light scan detects and removes deleted facet files."""
    from think.indexer.journal import scan_journal, search_journal

    # Initial scan
    scan_journal(str(journal_fixture), full=True)

    # Verify event is indexed
    total, _ = search_journal("Standup", topic="event")
    assert total >= 1

    # Delete the facet event file
    events_file = journal_fixture / "facets" / "work" / "events" / "20240101.jsonl"
    events_file.unlink()

    # Light rescan should detect the deletion (facet content is in scope)
    changed = scan_journal(str(journal_fixture), full=False)
    assert changed is True

    # Event should no longer be searchable
    total, _ = search_journal("Standup", topic="event")
    assert total == 0


def test_light_scan_removes_deleted_today_segment(tmp_path):
    """Test that light scan detects and removes deleted content from today."""
    from datetime import datetime

    from think.indexer.journal import scan_journal, search_journal

    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)

    # Create content for today (which is in light scan scope)
    today = datetime.now().strftime("%Y%m%d")
    day_dir = journal / today
    day_dir.mkdir()
    agents_dir = day_dir / "agents"
    agents_dir.mkdir()
    output_file = agents_dir / "flow.md"
    output_file.write_text("# Today Flow\n\nWorked on unique_today_content.\n")

    # Initial scan
    scan_journal(str(journal), full=False)

    # Verify content is indexed
    total, _ = search_journal("unique_today_content")
    assert total >= 1

    # Delete the output file
    output_file.unlink()

    # Light rescan should detect the deletion
    changed = scan_journal(str(journal), full=False)
    assert changed is True

    # Content should no longer be searchable
    total, _ = search_journal("unique_today_content")
    assert total == 0


def test_light_scan_preserves_historical_content(tmp_path):
    """Test that light scan does NOT remove historical day content from index."""
    from think.indexer.journal import scan_journal, search_journal

    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)

    # Create historical day content
    day_dir = journal / "20200101"
    day_dir.mkdir()
    agents_dir = day_dir / "agents"
    agents_dir.mkdir()
    output_file = agents_dir / "flow.md"
    output_file.write_text("# Historical Flow\n\nWorked on historical_content.\n")

    # Full scan to index historical content
    scan_journal(str(journal), full=True)

    # Verify content is indexed
    total, _ = search_journal("historical_content")
    assert total >= 1

    # Delete the historical file
    output_file.unlink()

    # Light rescan should NOT remove the historical content (out of scope)
    changed = scan_journal(str(journal), full=False)
    # No changes because the historical path is out of scope
    assert changed is False

    # Content should still be searchable (not removed)
    total, _ = search_journal("historical_content")
    assert total >= 1


def test_full_scan_removes_historical_content(tmp_path):
    """Test that full scan removes deleted historical day content."""
    from think.indexer.journal import scan_journal, search_journal

    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)

    # Create historical day content
    day_dir = journal / "20200101"
    day_dir.mkdir()
    agents_dir = day_dir / "agents"
    agents_dir.mkdir()
    output_file = agents_dir / "flow.md"
    output_file.write_text("# Historical Flow\n\nWorked on historical_full_test.\n")

    # Full scan to index historical content
    scan_journal(str(journal), full=True)

    # Verify content is indexed
    total, _ = search_journal("historical_full_test")
    assert total >= 1

    # Delete the historical file
    output_file.unlink()

    # Full rescan SHOULD remove the historical content
    changed = scan_journal(str(journal), full=True)
    assert changed is True

    # Content should no longer be searchable
    total, _ = search_journal("historical_full_test")
    assert total == 0


def test_index_file_valid(journal_fixture):
    """Test indexing a single valid file."""
    from think.indexer.journal import index_file, search_journal

    # Index a specific file
    result = index_file(str(journal_fixture), "20240101/agents/flow.md", verbose=True)
    assert result is True

    # Should be searchable
    total, results = search_journal("project alpha")
    assert total >= 1


def test_index_file_absolute_path(journal_fixture):
    """Test indexing with absolute path."""
    from think.indexer.journal import index_file, search_journal

    abs_path = str(journal_fixture / "20240101" / "agents" / "flow.md")
    result = index_file(str(journal_fixture), abs_path, verbose=True)
    assert result is True

    # Should be searchable
    total, _ = search_journal("project alpha")
    assert total >= 1


def test_index_file_updates_existing(journal_fixture):
    """Test that re-indexing a file replaces existing chunks."""
    from think.indexer.journal import index_file, search_journal

    # Index the file
    index_file(str(journal_fixture), "20240101/agents/flow.md")

    # Get initial count
    total1, _ = search_journal("project alpha")

    # Re-index the same file
    index_file(str(journal_fixture), "20240101/agents/flow.md")

    # Count should be the same (not doubled)
    total2, _ = search_journal("project alpha")
    assert total2 == total1


def test_index_file_not_found(journal_fixture):
    """Test indexing non-existent file raises error."""
    from think.indexer.journal import index_file

    with pytest.raises(FileNotFoundError, match="File not found"):
        index_file(str(journal_fixture), "nonexistent/file.md")


def test_index_file_outside_journal(journal_fixture, tmp_path_factory):
    """Test indexing file outside journal raises error."""
    from think.indexer.journal import index_file

    # Create a file in a separate temp directory (outside the journal)
    outside_dir = tmp_path_factory.mktemp("outside")
    outside_file = outside_dir / "outside.md"
    outside_file.write_text("# Outside\n\nThis is outside the journal.\n")

    with pytest.raises(ValueError, match="outside journal directory"):
        index_file(str(journal_fixture), str(outside_file))


def test_index_file_no_formatter(journal_fixture):
    """Test indexing file without formatter raises error."""
    from think.indexer.journal import index_file

    # Create a file with no formatter (e.g., .txt)
    txt_file = journal_fixture / "20240101" / "notes.txt"
    txt_file.write_text("Just some text notes.\n")

    with pytest.raises(ValueError, match="No formatter found"):
        index_file(str(journal_fixture), str(txt_file))
