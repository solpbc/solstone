import importlib
import json
import os


def test_parse_entity_line():
    indexer = importlib.import_module("think.entities")
    line = "* Person: John Doe - desc"
    etype, name, desc = indexer.parse_entity_line(line)
    assert etype == "Person" and name == "John Doe" and desc == "desc"
    assert indexer.parse_entity_line("invalid") is None


def test_parse_entities(tmp_path):
    indexer = importlib.import_module("think.entities")
    md = tmp_path / "entities.md"
    md.write_text("* Person: Jane - info\n")
    result = indexer.parse_entities(str(tmp_path))
    assert result == [("Person", "Jane", "info")]


def test_build_entities_and_cache(tmp_path):
    indexer = importlib.import_module("think.entities")
    cache = {
        "20240101": {
            "entries": [("Person", "Jane", "info")],
            "mtime": 1,
            "file": "f",
        }
    }
    built = indexer.build_entities(cache)
    assert "Person" in built and "Jane" in built["Person"]
    indexer.save_cache(str(tmp_path), cache)
    loaded = indexer.load_cache(str(tmp_path))
    assert loaded["20240101"]["entries"][0][0] == "Person"


def test_entities_class(tmp_path):
    indexer = importlib.import_module("think.entities")
    day = tmp_path / "20240101"
    day.mkdir()
    (day / "entities.md").write_text("* Person: Jane\n")
    ent = indexer.Entities(str(tmp_path))
    ent.rescan()
    result = ent.index()
    assert "Person" in result and "Jane" in result["Person"]


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
    mod.scan_occurrences(str(journal), verbose=True)
    total, results = mod.search_occurrences("Standup")
    assert total == 1
    assert results and results[0]["metadata"]["day"] == "20240101"
    assert results[0]["occurrence"]["title"] == "Standup"


def test_ponder_index(tmp_path):
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)
    day = journal / "20240102"
    day.mkdir()
    topics_dir = day / "topics"
    topics_dir.mkdir()
    (topics_dir / "files.md").write_text("This is a test sentence.\n")
    mod.scan_topics(str(journal), verbose=True)
    total, results = mod.search_topics("test")
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
    (day / "123000_audio.json").write_text(json.dumps({"text": "hello"}))
    (day / "123000_monitor_1_diff.json").write_text(json.dumps({"desc": "screen"}))
    mod.scan_raws(str(journal), verbose=True)
    total, results = mod.search_raws("hello")
    assert total == 1
    assert results and results[0]["metadata"]["type"] == "audio"
    assert (day / "indexer" / "indexer.sqlite").exists()


def test_index_caching(tmp_path):
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    day = journal / "20240104"
    day.mkdir()
    topics_dir = day / "topics"
    topics_dir.mkdir()
    md = topics_dir / "files.md"
    md.write_text("Cached sentence.\n")

    # First scan indexes the file
    assert mod.scan_topics(str(journal)) is True

    # Second scan without modification should be a no-op
    assert mod.scan_topics(str(journal)) is False

    # Modify file to trigger reindex
    import time as _time

    _time.sleep(1)
    md.write_text("Updated sentence.\n")
    assert mod.scan_topics(str(journal)) is True


def test_search_raws_day(tmp_path):
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)

    day1 = journal / "20240105"
    day1.mkdir()
    (day1 / "123000_audio.json").write_text(json.dumps({"text": "hello"}))

    day2 = journal / "20240106"
    day2.mkdir()
    (day2 / "090000_audio.json").write_text(json.dumps({"text": "hello"}))

    mod.scan_raws(str(journal), verbose=True)

    total_all, _ = mod.search_raws("hello", limit=10)
    assert total_all == 2

    total_day1, results_day1 = mod.search_raws("hello", limit=10, day="20240105")
    assert total_day1 == 1
    assert results_day1[0]["metadata"]["day"] == "20240105"
