import importlib
import json


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
    cache: dict = {}
    mod.scan_occurrences(str(journal), cache, verbose=True)
    results = mod.search_occurrences(str(journal), "Standup")
    assert results and results[0]["metadata"]["day"] == "20240101"


def test_ponder_index(tmp_path):
    mod = importlib.import_module("think.indexer")
    journal = tmp_path
    day = journal / "20240102"
    day.mkdir()
    topics_dir = day / "topics"
    topics_dir.mkdir()
    (topics_dir / "files.md").write_text("This is a test sentence.\n")
    cache: dict = {}
    mod.scan_ponders(str(journal), cache, verbose=True)
    total, results = mod.search_ponders(str(journal), "test")
    assert total == 1
    assert results and results[0]["metadata"]["path"] == "20240102/topics/files.md"
