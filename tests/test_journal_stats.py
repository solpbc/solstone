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
    (day / "topics" / "flow.md").write_text("")
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
    assert js.days["20240101"]["repair_observe"] == 1
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


def test_token_usage(tmp_path, monkeypatch):
    stats_mod = importlib.import_module("think.journal_stats")
    journal = tmp_path
    day1 = journal / "20240101"
    day1.mkdir()
    day2 = journal / "20240102"
    day2.mkdir()

    # Create tokens directory with test token files
    tokens_dir = journal / "tokens"
    tokens_dir.mkdir()

    # Create token files for different models on the same day
    token1 = {
        "timestamp": 1704067200.0,
        "timestamp_str": "20240101_120000",
        "model": "gemini-2.5-flash",
        "context": "test_context",
        "usage": {
            "prompt_tokens": 100,
            "candidates_tokens": 50,
            "cached_tokens": 10,
            "thoughts_tokens": 5,
            "total_tokens": 165,
        },
    }

    token2 = {
        "timestamp": 1704070800.0,
        "timestamp_str": "20240101_130000",
        "model": "gemini-2.5-flash",
        "context": "test_context2",
        "usage": {
            "prompt_tokens": 200,
            "candidates_tokens": 100,
            "cached_tokens": 20,
            "thoughts_tokens": 10,
            "total_tokens": 330,
        },
    }

    token3 = {
        "timestamp": 1704074400.0,
        "timestamp_str": "20240101_140000",
        "model": "claude-3-opus",
        "context": "test_context3",
        "usage": {"prompt_tokens": 500, "candidates_tokens": 250, "total_tokens": 750},
    }

    # Token from different day (should not be included)
    token4 = {
        "timestamp": 1704153600.0,
        "timestamp_str": "20240102_120000",
        "model": "gemini-2.5-flash",
        "context": "test_context4",
        "usage": {
            "prompt_tokens": 1000,
            "candidates_tokens": 500,
            "total_tokens": 1500,
        },
    }

    (tokens_dir / "1704067200000.json").write_text(json.dumps(token1))
    (tokens_dir / "1704070800000.json").write_text(json.dumps(token2))
    (tokens_dir / "1704074400000.json").write_text(json.dumps(token3))
    (tokens_dir / "1704153600000.json").write_text(json.dumps(token4))

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    js = stats_mod.JournalStats()
    js.scan(str(journal))

    # Check token usage for the day
    assert "20240101" in js.token_usage
    assert "gemini-2.5-flash" in js.token_usage["20240101"]
    assert "claude-3-opus" in js.token_usage["20240101"]

    # Check gemini totals for the day (sum of token1 and token2)
    gemini_usage = js.token_usage["20240101"]["gemini-2.5-flash"]
    assert gemini_usage["prompt_tokens"] == 300  # 100 + 200
    assert gemini_usage["candidates_tokens"] == 150  # 50 + 100
    assert gemini_usage["cached_tokens"] == 30  # 10 + 20
    assert gemini_usage["thoughts_tokens"] == 15  # 5 + 10
    assert gemini_usage["total_tokens"] == 495  # 165 + 330

    # Check claude totals for the day
    claude_usage = js.token_usage["20240101"]["claude-3-opus"]
    assert claude_usage["prompt_tokens"] == 500
    assert claude_usage["candidates_tokens"] == 250
    assert claude_usage["total_tokens"] == 750

    # Check overall model totals
    assert (
        js.token_totals["gemini-2.5-flash"]["prompt_tokens"] == 1300
    )  # 300 from day1 + 1000 from day2
    assert js.token_totals["claude-3-opus"]["prompt_tokens"] == 500

    # Test markdown generation includes token usage
    js2 = stats_mod.JournalStats()
    js2.scan(str(journal))
    md = js2.to_markdown()
    assert "Token Usage by Model" in md
    assert "gemini-2.5-flash" in md
    assert "claude-3-opus" in md

    # Test JSON output includes token usage
    data = js2.to_dict()
    assert "token_usage_by_day" in data
    assert "token_totals_by_model" in data
    assert (
        data["token_usage_by_day"]["20240101"]["gemini-2.5-flash"]["total_tokens"]
        == 495
    )
