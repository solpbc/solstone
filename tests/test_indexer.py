import importlib
import json
import os


def test_parse_entity_line():
    """Test removed - parse_entity_line no longer exists in JSONL format."""
    # This function was removed as part of the markdown -> JSONL migration
    # Entity parsing is now done via JSON.loads() directly
    pass


def test_parse_entities(tmp_path):
    indexer = importlib.import_module("think.indexer")
    jsonl = tmp_path / "entities.jsonl"
    jsonl.write_text('{"type": "Person", "name": "Jane", "description": "info"}\n')
    result = indexer.parse_entities(str(tmp_path))
    assert len(result) == 1
    assert result[0]["type"] == "Person"
    assert result[0]["name"] == "Jane"
    assert result[0]["description"] == "info"


# These tests are deprecated since entities.json caching and Entities class are removed


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
                "source": "topics/meetings.md",
                "start": "09:00:00",
                "end": "09:30:00",
                "title": "Standup",
                "summary": "Daily sync",
                "details": "progress",
            }
        ],
    }
    topics_dir = day / "topics"
    topics_dir.mkdir()
    (topics_dir / "meetings.json").write_text(json.dumps(data))
    mod.scan_events(str(journal), verbose=True)
    total, results = mod.search_events("Standup")
    assert total == 1
    assert results and results[0]["metadata"]["day"] == "20240101"
    assert results[0]["event"]["title"] == "Standup"


def test_ponder_index(tmp_path):
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)
    day = journal / "20240102"
    day.mkdir()
    topics_dir = day / "topics"
    topics_dir.mkdir()
    (topics_dir / "files.md").write_text("This is a test sentence.\n")
    mod.scan_summaries(str(journal), verbose=True)
    total, results = mod.search_summaries("test")
    assert total == 1
    assert results and results[0]["metadata"]["path"] == "20240102/topics/files.md"
    assert total == 1
    assert results and results[0]["metadata"]["path"] == "20240102/topics/files.md"


def test_raw_index(tmp_path):
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)
    day = journal / "20240103"
    day.mkdir()
    # Write JSONL format: metadata first, then entries
    (day / "123000_audio.jsonl").write_text(
        json.dumps({"topics": ["hi"], "setting": "personal"})
        + "\n"
        + json.dumps(
            {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "hello"}
        )
        + "\n"
    )
    (day / "123000_monitor_1_diff.json").write_text(
        json.dumps(
            {
                "visual_description": "screen",
                "full_ocr": "some ocr",
            }
        )
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
    topics_dir = day / "topics"
    topics_dir.mkdir()
    md = topics_dir / "files.md"
    md.write_text("Cached sentence.\n")

    # First scan indexes the file
    assert mod.scan_summaries(str(journal)) is True

    # Second scan without modification should be a no-op
    assert mod.scan_summaries(str(journal)) is False

    # Modify file to trigger reindex
    import time as _time

    _time.sleep(1)
    md.write_text("Updated sentence.\n")
    assert mod.scan_summaries(str(journal)) is True


def test_search_raws_day(tmp_path):
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)

    day1 = journal / "20240105"
    day1.mkdir()
    # Write JSONL format: metadata first, then entries
    (day1 / "123000_audio.jsonl").write_text(
        json.dumps({"topics": ["hi"], "setting": "personal"})
        + "\n"
        + json.dumps(
            {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "hello"}
        )
        + "\n"
    )

    day2 = journal / "20240106"
    day2.mkdir()
    # Write JSONL format: metadata first, then entries
    (day2 / "090000_audio.jsonl").write_text(
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
    # Write JSONL format: metadata first, then entries
    (day / "090000_audio.jsonl").write_text(
        json.dumps({"topics": ["hi"], "setting": "personal"})
        + "\n"
        + json.dumps(
            {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "hello"}
        )
        + "\n"
    )
    # Write JSONL format: metadata first, then entries
    (day / "123000_audio.jsonl").write_text(
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
