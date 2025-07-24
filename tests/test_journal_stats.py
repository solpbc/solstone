import importlib
import json


def test_scan_day(tmp_path, monkeypatch):
    stats_mod = importlib.import_module("think.journal_stats")
    journal = tmp_path
    day = journal / "20240101"
    day.mkdir()
    (day / "123456_audio.flac").write_bytes(b"RIFF")
    (day / "123456_monitor_1_diff.png").write_bytes(b"PNG")
    (day / "123456_monitor_1_diff_box.json").write_text("{}")
    (day / "123456_monitor_1_diff.json").write_text("{}")
    (day / "123456_screen.md").write_text("hi")
    (day / "entities.md").write_text("")
    (day / "topics").mkdir()
    (day / "topics" / "day.md").write_text("")
    data = {
        "day": "20240101",
        "occurrences": [
            {
                "type": "meeting",
                "start": "00:00:00",
                "end": "00:05:00",
                "title": "t",
                "summary": "s",
                "work": True,
                "participants": [],
                "details": "",
            }
        ],
    }
    (day / "topics" / "meetings.json").write_text(json.dumps(data))
    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    js = stats_mod.JournalStats()
    js.scan_day("20240101", str(day))
    assert js.days["20240101"].get("audio_flac", 0) == 0
    assert js.days["20240101"]["repair_hear"] == 1
    assert js.totals["diff_png"] == 0
    assert js.topic_counts["meetings"] == 1
    assert js.heatmap[0][0] == 5


def test_markdown(tmp_path, monkeypatch):
    stats_mod = importlib.import_module("think.journal_stats")
    journal = tmp_path
    day = journal / "20240101"
    day.mkdir()
    (day / "123456_audio.flac").write_bytes(b"RIFF")
    (day / "123456_audio.json").write_text("{}")
    (day / "topics").mkdir()
    (day / "topics" / "meetings.json").write_text(
        json.dumps({"day": "20240101", "occurrences": []})
    )
    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    js = stats_mod.JournalStats()
    js.scan(str(journal))
    md = js.to_markdown()
    assert "Days scanned: 1" in md
    js.save_markdown(str(journal))
    assert (journal / "summary.md").exists()
    assert "Ponder processed" in md
