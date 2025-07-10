import importlib
import json


def test_format_date():
    review = importlib.import_module("dream")
    assert "2024" not in review.format_date("20240102")
    assert review.format_date("bad") == "bad"


def test_time_since(monkeypatch):
    review = importlib.import_module("dream")
    monkeypatch.setattr("time.time", lambda: 120)
    assert review.time_since(60) == "1 minute ago"


def test_modify_and_update(tmp_path):
    review = importlib.import_module("dream")
    md = tmp_path / "entities.md"
    md.write_text("* Person: Jane - desc\n")
    review.modify_entity_in_file(str(md), "Person", "Jane", operation="remove")
    assert md.read_text() == ""
    md.write_text("* Person: Jane - desc\n")
    review.modify_entity_in_file(str(md), "Person", "Jane", new_name="J", operation="rename")
    assert "J" in md.read_text()
    review.update_top_entry(str(tmp_path), "Person", "J", "info")
    top_path = tmp_path / "entities.md"
    assert top_path.read_text()


def test_build_index_occurrence_format(tmp_path):
    review = importlib.import_module("dream")
    day = tmp_path / "20240101"
    day.mkdir()
    data = {
        "day": "20240101",
        "occurrences": [
            {
                "type": "meeting",
                "start": "09:00:00",
                "end": "09:30:00",
                "title": "Standup",
                "summary": "Daily sync",
                "details": {"topicsDiscussed": "progress"},
            }
        ],
    }
    (day / "ponder_meetings.json").write_text(json.dumps(data))
    index = review.build_occurrence_index(str(tmp_path))
    assert index["20240101"][0]["title"] == "Standup"
    assert index["20240101"][0]["startTime"].endswith("T09:00:00")


def test_build_index_old_format(tmp_path):
    review = importlib.import_module("dream")
    day = tmp_path / "20240102"
    day.mkdir()
    meetings = [{"title": "Old", "startTime": "2024-01-02T10:00:00"}]
    (day / "ponder_meetings.json").write_text(json.dumps(meetings))
    index = review.build_occurrence_index(str(tmp_path))
    assert index["20240102"][0]["title"] == "Old"
