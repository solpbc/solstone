import importlib
import json
import os

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
        result = sanitize_fts_query('sunstone OR pbc OR "public benefit"')
        assert result == 'sunstone OR pbc OR "public benefit"'

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


def test_parse_entity_line():
    """Test removed - parse_entity_line no longer exists in JSONL format."""
    # This function was removed as part of the markdown -> JSONL migration
    # Entity parsing is now done via JSON.loads() directly
    pass


def test_occurrence_index(tmp_path):
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)
    day = journal / "20240101"
    day.mkdir()
    data = {
        "day": "20240101",
        "occurrences": [
            {
                "type": "meeting",
                "source": "insights/meetings.md",
                "start": "09:00:00",
                "end": "09:30:00",
                "title": "Standup",
                "summary": "Daily sync",
                "details": "progress",
            }
        ],
    }
    insights_dir = day / "insights"
    insights_dir.mkdir()
    (insights_dir / "meetings.json").write_text(json.dumps(data))
    mod.scan_events(str(journal), verbose=True)
    total, results = mod.search_events("Standup")
    assert total == 1
    assert results and results[0]["metadata"]["day"] == "20240101"
    assert results[0]["event"]["title"] == "Standup"
    # Occurrences should have occurred=True
    assert results[0]["metadata"]["occurred"] is True


def test_anticipation_index(tmp_path):
    """Test that anticipations are indexed with occurred=False and by event date."""
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)

    # Create day directory where schedule.json is captured
    day = journal / "20240101"
    day.mkdir()
    insights_dir = day / "insights"
    insights_dir.mkdir()

    # Create anticipations data - events scheduled for future dates
    data = {
        "day": "20240101",
        "anticipations": [
            {
                "type": "meeting",
                "date": "2024-01-05",
                "start": "09:00:00",
                "end": "10:00:00",
                "title": "Project kickoff",
                "summary": "Initial project meeting",
                "work": True,
                "participants": ["Alice", "Bob"],
                "facet": "work",
                "details": "Virtual meeting",
            },
            {
                "type": "deadline",
                "date": "2024-01-10",
                "start": None,
                "end": None,
                "title": "Q1 Planning Due",
                "summary": "Submit planning document",
                "work": True,
                "participants": [],
                "facet": "work",
                "details": "Full day deadline",
            },
        ],
    }
    (insights_dir / "schedule.json").write_text(json.dumps(data))

    mod.scan_events(str(journal), verbose=True)

    # Search for anticipations by text
    total, results = mod.search_events("kickoff")
    assert total == 1
    assert results[0]["event"]["title"] == "Project kickoff"
    # Anticipations should have occurred=False
    assert results[0]["metadata"]["occurred"] is False
    # Anticipations are indexed by their event date, not capture date
    assert results[0]["metadata"]["day"] == "20240105"

    # Search by event date (the date the event is scheduled for)
    total, results = mod.search_events("", day="20240110")
    assert total == 1
    assert results[0]["event"]["title"] == "Q1 Planning Due"
    assert results[0]["metadata"]["occurred"] is False

    # Filter by occurred status
    total, results = mod.search_events("", occurred=False)
    assert total == 2  # Both anticipations

    total, results = mod.search_events("", occurred=True)
    assert total == 0  # No occurrences in this test


def test_mixed_occurrences_and_anticipations(tmp_path):
    """Test that occurrences and anticipations can coexist in the index."""
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)

    day = journal / "20240101"
    day.mkdir()
    insights_dir = day / "insights"
    insights_dir.mkdir()

    # Create occurrences (what happened today)
    occurrences_data = {
        "day": "20240101",
        "occurrences": [
            {
                "type": "meeting",
                "start": "09:00:00",
                "end": "09:30:00",
                "title": "Morning standup",
                "summary": "Team sync",
                "facet": "work",
            }
        ],
    }
    (insights_dir / "meetings.json").write_text(json.dumps(occurrences_data))

    # Create anticipations (what's scheduled for future)
    anticipations_data = {
        "day": "20240101",
        "anticipations": [
            {
                "type": "meeting",
                "date": "2024-01-05",
                "start": "14:00:00",
                "end": "15:00:00",
                "title": "Client demo",
                "summary": "Product demonstration",
                "facet": "work",
            }
        ],
    }
    (insights_dir / "schedule.json").write_text(json.dumps(anticipations_data))

    mod.scan_events(str(journal), verbose=True)

    # All events should be searchable
    total, _ = mod.search_events("")
    assert total == 2

    # Filter to only occurrences
    total, results = mod.search_events("", occurred=True)
    assert total == 1
    assert results[0]["event"]["title"] == "Morning standup"
    assert results[0]["metadata"]["day"] == "20240101"

    # Filter to only anticipations
    total, results = mod.search_events("", occurred=False)
    assert total == 1
    assert results[0]["event"]["title"] == "Client demo"
    assert results[0]["metadata"]["day"] == "20240105"  # Indexed by event date


def test_ponder_index(tmp_path):
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)
    day = journal / "20240102"
    day.mkdir()
    insights_dir = day / "insights"
    insights_dir.mkdir()
    (insights_dir / "files.md").write_text("This is a test sentence.\n")
    mod.scan_insights(str(journal), verbose=True)
    total, results = mod.search_insights("test")
    assert total == 1
    assert results and results[0]["metadata"]["path"] == "20240102/insights/files.md"
    assert total == 1
    assert results and results[0]["metadata"]["path"] == "20240102/insights/files.md"


def test_raw_index(tmp_path):
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)
    day = journal / "20240103"
    day.mkdir()
    # Write JSONL format: metadata first, then entries in segment directory
    ts_dir = day / "123000"
    ts_dir.mkdir()
    (ts_dir / "audio.jsonl").write_text(
        json.dumps({"topics": ["hi"], "setting": "personal"})
        + "\n"
        + json.dumps(
            {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "hello"}
        )
        + "\n"
    )
    # Write screen.jsonl instead of legacy format
    (ts_dir / "screen.jsonl").write_text(
        '{"raw": "screen.webm"}\n'
        + json.dumps(
            {
                "frame_id": 1,
                "timestamp": 1.0,
                "analysis": {"visual_description": "screen"},
                "extracted_text": "some ocr",
            }
        )
        + "\n"
    )
    mod.scan_transcripts(str(journal), verbose=True)
    total, results = mod.search_transcripts("hello")
    assert total == 1
    assert results and results[0]["metadata"]["type"] == "audio"
    assert (day / "indexer" / "transcripts.sqlite").exists()


def test_index_caching(tmp_path):
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)
    day = journal / "20240104"
    day.mkdir()
    insights_dir = day / "insights"
    insights_dir.mkdir()
    md = insights_dir / "files.md"
    md.write_text("Cached sentence.\n")

    # First scan indexes the file
    assert mod.scan_insights(str(journal)) is True

    # Second scan without modification should be a no-op
    assert mod.scan_insights(str(journal)) is False

    # Modify file to trigger reindex
    import time as _time

    _time.sleep(1)
    md.write_text("Updated sentence.\n")
    assert mod.scan_insights(str(journal)) is True


def test_search_raws_day(tmp_path):
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)

    day1 = journal / "20240105"
    day1.mkdir()
    # Write JSONL format: metadata first, then entries in segment directory
    ts_dir1 = day1 / "123000"
    ts_dir1.mkdir()
    (ts_dir1 / "audio.jsonl").write_text(
        json.dumps({"topics": ["hi"], "setting": "personal"})
        + "\n"
        + json.dumps(
            {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "hello"}
        )
        + "\n"
    )

    day2 = journal / "20240106"
    day2.mkdir()
    # Write JSONL format: metadata first, then entries in segment directory
    ts_dir2 = day2 / "090000"
    ts_dir2.mkdir()
    (ts_dir2 / "audio.jsonl").write_text(
        json.dumps({"topics": ["hi"], "setting": "personal"})
        + "\n"
        + json.dumps(
            {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "hello"}
        )
        + "\n"
    )

    mod.scan_transcripts(str(journal), verbose=True)

    total_all, _ = mod.search_transcripts("hello", limit=10)
    assert total_all == 2

    total_day1, results_day1 = mod.search_transcripts("hello", limit=10, day="20240105")
    assert total_day1 == 1
    assert results_day1[0]["metadata"]["day"] == "20240105"


def test_search_raws_time_order(tmp_path):
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)

    day = journal / "20240107"
    day.mkdir()
    # Write JSONL format: metadata first, then entries in segment directory
    ts_dir1 = day / "090000"
    ts_dir1.mkdir()
    (ts_dir1 / "audio.jsonl").write_text(
        json.dumps({"topics": ["hi"], "setting": "personal"})
        + "\n"
        + json.dumps(
            {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "hello"}
        )
        + "\n"
    )
    # Write JSONL format: metadata first, then entries
    ts_dir2 = day / "123000"
    ts_dir2.mkdir()
    (ts_dir2 / "audio.jsonl").write_text(
        json.dumps({"topics": ["hi"], "setting": "personal"})
        + "\n"
        + json.dumps(
            {"start": "00:00:02", "source": "mic", "speaker": 1, "text": "hello"}
        )
        + "\n"
    )

    mod.scan_transcripts(str(journal), verbose=True)

    total, results = mod.search_transcripts("hello", limit=10, day="20240107")
    assert total == 2
    assert [r["metadata"]["time"] for r in results] == ["090000", "123000"]


# test_entities_index removed - tested old behavior where top-level and day-level
# entities were indexed. Now only facet-scoped entities are indexed.


def test_scan_transcripts_single_day(tmp_path):
    """Test scanning transcripts for a single day only."""
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)

    # Create two days with transcripts
    day1 = journal / "20240108"
    day1.mkdir()
    ts_dir1 = day1 / "100000"
    ts_dir1.mkdir()
    (ts_dir1 / "audio.jsonl").write_text(
        json.dumps({"topics": ["test"], "setting": "personal"})
        + "\n"
        + json.dumps(
            {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "day one"}
        )
        + "\n"
    )

    day2 = journal / "20240109"
    day2.mkdir()
    ts_dir2 = day2 / "110000"
    ts_dir2.mkdir()
    (ts_dir2 / "audio.jsonl").write_text(
        json.dumps({"topics": ["test"], "setting": "personal"})
        + "\n"
        + json.dumps(
            {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "day two"}
        )
        + "\n"
    )

    # Scan only day 1
    changed = mod.scan_transcripts(str(journal), verbose=True, day="20240108")
    assert changed is True

    # Day 1 should have index
    assert (day1 / "indexer" / "transcripts.sqlite").exists()

    # Day 2 should NOT have index yet
    assert not (day2 / "indexer" / "transcripts.sqlite").exists()

    # Search day 1 should find result
    total, results = mod.search_transcripts("day", limit=10, day="20240108")
    assert total == 1
    assert results[0]["metadata"]["day"] == "20240108"

    # Now scan only day 2
    changed = mod.scan_transcripts(str(journal), verbose=True, day="20240109")
    assert changed is True

    # Day 2 should now have index
    assert (day2 / "indexer" / "transcripts.sqlite").exists()

    # Search day 2 should find result
    total, results = mod.search_transcripts("day", limit=10, day="20240109")
    assert total == 1
    assert results[0]["metadata"]["day"] == "20240109"

    # Full scan should find both
    total, _ = mod.search_transcripts("day", limit=10)
    assert total == 2


def test_scan_transcripts_single_segment(tmp_path):
    """Test scanning transcripts for a specific segment within a day."""
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)

    # Create a day with multiple segments
    day = journal / "20240110"
    day.mkdir()

    # Segment 1: 100000
    segment1 = day / "100000"
    segment1.mkdir()
    (segment1 / "audio.jsonl").write_text(
        json.dumps({"topics": ["test"], "setting": "personal"})
        + "\n"
        + json.dumps(
            {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "segment one"}
        )
        + "\n"
    )

    # Segment 2: 110000_300 (with duration)
    segment2 = day / "110000_300"
    segment2.mkdir()
    (segment2 / "audio.jsonl").write_text(
        json.dumps({"topics": ["test"], "setting": "personal"})
        + "\n"
        + json.dumps(
            {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "segment two"}
        )
        + "\n"
    )

    # Segment 3: 120000
    segment3 = day / "120000"
    segment3.mkdir()
    (segment3 / "audio.jsonl").write_text(
        json.dumps({"topics": ["test"], "setting": "personal"})
        + "\n"
        + json.dumps(
            {
                "start": "00:00:01",
                "source": "mic",
                "speaker": 1,
                "text": "segment three",
            }
        )
        + "\n"
    )

    # Scan only segment 2 with duration format
    changed = mod.scan_transcripts(
        str(journal), verbose=True, day="20240110", segment="110000_300"
    )
    assert changed is True

    # Day should have index
    assert (day / "indexer" / "transcripts.sqlite").exists()

    # Search should only find segment 2
    total, results = mod.search_transcripts("segment", limit=10, day="20240110")
    assert total == 1
    assert results[0]["metadata"]["time"] == "110000_300"
    assert "segment two" in results[0]["text"]

    # Now scan segment 1 (without duration format)
    # This replaces the index with only segment 1 (removes segment 2)
    changed = mod.scan_transcripts(
        str(journal), verbose=True, day="20240110", segment="100000"
    )
    assert changed is True

    # Search should now find only segment 1 (segment 2 was removed)
    total, results = mod.search_transcripts("segment", limit=10, day="20240110")
    assert total == 1
    assert results[0]["metadata"]["time"] == "100000"
    assert "segment one" in results[0]["text"]

    # Scan all segments in the day - this is the proper way to get all segments
    changed = mod.scan_transcripts(str(journal), verbose=True, day="20240110")
    assert changed is True

    # Search should now find all 3 segments
    total, results = mod.search_transcripts("segment", limit=10, day="20240110")
    assert total == 3
    assert {r["metadata"]["time"] for r in results} == {
        "100000",
        "110000_300",
        "120000",
    }

    # Verify segment-specific scan with duration still works after full scan
    changed = mod.scan_transcripts(
        str(journal), verbose=True, day="20240110", segment="110000_300"
    )
    # Should return False because file hasn't changed (mtime caching)
    # Actually returns True because it removes other segments
    assert changed is True
    total, results = mod.search_transcripts("segment", limit=10, day="20240110")
    assert total == 1
    assert results[0]["metadata"]["time"] == "110000_300"
