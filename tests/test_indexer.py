import importlib
import json
from pathlib import Path


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


def test_get_entities(tmp_path):
    indexer = importlib.import_module("think.entities")
    day = tmp_path / "20240101"
    day.mkdir()
    (day / "entities.md").write_text("* Person: Jane\n")
    result = indexer.get_entities(str(tmp_path))
    assert "Person" in result and "Jane" in result["Person"]
