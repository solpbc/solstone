# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib
import json


def test_scan_day(tmp_path, monkeypatch):
    stats_mod = importlib.import_module("think.journal_stats")
    journal = tmp_path
    day = journal / "20240101"
    day.mkdir()

    # Create an audio jsonl file in segment directory (already processed)
    ts_dir = day / "default" / "123456_300"
    ts_dir.mkdir(parents=True)
    (ts_dir / "audio.jsonl").write_text(
        '{"raw": "raw.flac"}\n'
        '{"start": "10:00:00", "text": "hello"}\n'
        '{"start": "10:01:00", "text": "world"}\n'
    )

    # Create unprocessed media files in a second segment directory (no jsonl output yet)
    ts_dir2 = day / "default" / "134500_300"
    ts_dir2.mkdir(parents=True)
    (ts_dir2 / "audio.flac").write_bytes(b"RIFF")
    (ts_dir2 / "center_DP-1_screen.webm").write_bytes(b"WEBM")

    (day / "entities.md").write_text("")
    (day / "agents").mkdir()
    (day / "agents" / "flow.md").write_text("")

    # Create event in new JSONL format: facets/{facet}/events/YYYYMMDD.jsonl
    events_dir = journal / "facets" / "work" / "events"
    events_dir.mkdir(parents=True)
    event = {
        "type": "meeting",
        "start": "00:00:00",
        "end": "00:05:00",
        "title": "t",
        "summary": "s",
        "work": True,
        "participants": [],
        "details": "",
        "facet": "work",
        "topic": "meetings",
        "occurred": True,
        "source": "20240101/agents/meetings.md",
    }
    (events_dir / "20240101.jsonl").write_text(json.dumps(event))

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    js = stats_mod.JournalStats()
    day_data = js.scan_day("20240101", str(day))
    js._apply_day_stats("20240101", day_data)
    assert js.days["20240101"]["audio_sessions"] == 1
    assert js.days["20240101"]["audio_segments"] == 2
    assert (
        js.days["20240101"]["pending_segments"] == 1
    )  # Both files belong to same segment
    assert js.topic_counts["meetings"] == 1
    assert js.facet_counts["work"] == 1
    assert js.facet_minutes["work"] == 5.0
    assert js.heatmap[0][0] == 5
    assert js.days["20240101"]["day_bytes"] > 0


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

    # Create token files for different models on the same day (using new normalized format)
    token1 = {
        "timestamp": 1704067200.0,
        "timestamp_str": "20240101_120000",
        "model": "gemini-2.5-flash",
        "context": "test_context",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cached_tokens": 10,
            "reasoning_tokens": 5,
            "total_tokens": 165,
        },
    }

    token2 = {
        "timestamp": 1704070800.0,
        "timestamp_str": "20240101_130000",
        "model": "gemini-2.5-flash",
        "context": "test_context2",
        "usage": {
            "input_tokens": 200,
            "output_tokens": 100,
            "cached_tokens": 20,
            "reasoning_tokens": 10,
            "total_tokens": 330,
        },
    }

    token3 = {
        "timestamp": 1704074400.0,
        "timestamp_str": "20240101_140000",
        "model": "claude-3-opus",
        "context": "test_context3",
        "usage": {"input_tokens": 500, "output_tokens": 250, "total_tokens": 750},
    }

    # Token from different day (new normalized format)
    token4 = {
        "timestamp": 1704153600.0,
        "timestamp_str": "20240102_120000",
        "model": "gemini-2.5-flash",
        "context": "test_context4",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
        },
    }

    # Write tokens as JSONL format (one per line in daily file)
    (tokens_dir / "20240101.jsonl").write_text(
        json.dumps(token1)
        + "\n"
        + json.dumps(token2)
        + "\n"
        + json.dumps(token3)
        + "\n"
    )
    (tokens_dir / "20240102.jsonl").write_text(json.dumps(token4) + "\n")

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    js = stats_mod.JournalStats()
    js.scan(str(journal))

    # Check token usage for the day
    assert "20240101" in js.token_usage
    assert "gemini-2.5-flash" in js.token_usage["20240101"]
    assert "claude-3-opus" in js.token_usage["20240101"]

    # Check gemini totals for the day (sum of token1 and token2)
    gemini_usage = js.token_usage["20240101"]["gemini-2.5-flash"]
    assert gemini_usage["input_tokens"] == 300  # 100 + 200
    assert gemini_usage["output_tokens"] == 150  # 50 + 100
    assert gemini_usage["cached_tokens"] == 30  # 10 + 20
    assert gemini_usage["reasoning_tokens"] == 15  # 5 + 10
    assert gemini_usage["total_tokens"] == 495  # 165 + 330

    # Check claude totals for the day
    claude_usage = js.token_usage["20240101"]["claude-3-opus"]
    assert claude_usage["input_tokens"] == 500
    assert claude_usage["output_tokens"] == 250
    assert claude_usage["total_tokens"] == 750

    # Check overall model totals
    assert (
        js.token_totals["gemini-2.5-flash"]["input_tokens"] == 1300
    )  # 300 from day1 + 1000 from day2
    assert js.token_totals["claude-3-opus"]["input_tokens"] == 500

    # Test JSON output includes token usage
    data = js.to_dict()
    assert "token_usage_by_day" in data
    assert "token_totals_by_model" in data
    assert "total_audio_duration" in data
    assert "total_screen_duration" in data
    assert (
        data["token_usage_by_day"]["20240101"]["gemini-2.5-flash"]["total_tokens"]
        == 495
    )


def test_caching(tmp_path, monkeypatch):
    """Test that per-day caching works correctly."""
    stats_mod = importlib.import_module("think.journal_stats")
    journal = tmp_path
    day = journal / "20240101"
    day.mkdir()

    # Create an audio jsonl file in segment directory
    ts_dir = day / "default" / "123456_300"
    ts_dir.mkdir(parents=True)
    (ts_dir / "audio.jsonl").write_text(
        '{"raw": "raw.flac"}\n'
        '{"start": "10:00:00", "text": "hello"}\n'
        '{"start": "10:01:00", "text": "world"}\n'
    )

    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    # First scan - should create cache
    js1 = stats_mod.JournalStats()
    js1.scan(str(journal), verbose=False, use_cache=True)
    assert js1.days["20240101"]["audio_sessions"] == 1
    assert (day / "stats.json").exists()

    # Load cache and verify contents
    with open(day / "stats.json") as f:
        cached = json.load(f)
    assert cached["stats"]["audio_sessions"] == 1
    assert cached["stats"]["audio_segments"] == 2

    # Second scan - should use cache
    js2 = stats_mod.JournalStats()
    js2.scan(str(journal), verbose=False, use_cache=True)
    assert js2.days["20240101"]["audio_sessions"] == 1
    assert js2.days["20240101"]["audio_segments"] == 2

    # Third scan with --no-cache - should re-scan
    js3 = stats_mod.JournalStats()
    js3.scan(str(journal), verbose=False, use_cache=False)
    assert js3.days["20240101"]["audio_sessions"] == 1


def test_token_usage_new_format(tmp_path, monkeypatch):
    """Test that the new unified token format is properly handled."""
    stats_mod = importlib.import_module("think.journal_stats")
    journal = tmp_path
    day1 = journal / "20240101"
    day1.mkdir()

    # Create tokens directory with new format token files
    tokens_dir = journal / "tokens"
    tokens_dir.mkdir()

    # New format: input_tokens, output_tokens, reasoning_tokens
    token_new = {
        "timestamp": 1704067200.0,
        "timestamp_str": "20240101_120000",
        "model": "gemini-2.5-flash",
        "context": "models._log_token_usage:241",
        "usage": {
            "input_tokens": 1716,
            "output_tokens": 3710,
            "total_tokens": 10114,
            "reasoning_tokens": 4688,
        },
    }

    # Write token as JSONL format
    (tokens_dir / "20240101.jsonl").write_text(json.dumps(token_new) + "\n")

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    js = stats_mod.JournalStats()
    js.scan(str(journal))

    # Check token usage is properly parsed
    assert "20240101" in js.token_usage
    assert "gemini-2.5-flash" in js.token_usage["20240101"]

    # Check new format fields are present
    gemini_usage = js.token_usage["20240101"]["gemini-2.5-flash"]
    assert gemini_usage["input_tokens"] == 1716
    assert gemini_usage["output_tokens"] == 3710
    assert gemini_usage["total_tokens"] == 10114
    assert gemini_usage["reasoning_tokens"] == 4688

    # Check overall model totals
    assert js.token_totals["gemini-2.5-flash"]["input_tokens"] == 1716
    assert js.token_totals["gemini-2.5-flash"]["output_tokens"] == 3710
    assert js.token_totals["gemini-2.5-flash"]["reasoning_tokens"] == 4688


def test_process_token_entry_counts_all_int_usage_fields(tmp_path, monkeypatch):
    """Int-valued fields in usage are all counted; top-level metadata is ignored."""
    stats_mod = importlib.import_module("think.journal_stats")
    journal = tmp_path
    day1 = journal / "20240101"
    day1.mkdir()

    tokens_dir = journal / "tokens"
    tokens_dir.mkdir()

    token_entry_with_duration = {
        "timestamp": 1704067200.0,
        "model": "gemini-2.5-flash",
        "context": "muse.system.meetings",
        "type": "cogitate",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "duration_ms": 3000,
        },
    }

    token_entry_without_duration = {
        "timestamp": 1704067300.0,
        "model": "gemini-2.5-pro",
        "context": "think.detect_transcript.detect",
        "type": "generate",
        "usage": {
            "input_tokens": 80,
            "output_tokens": 20,
            "total_tokens": 100,
        },
    }

    (tokens_dir / "20240101.jsonl").write_text(
        json.dumps(token_entry_with_duration)
        + "\n"
        + json.dumps(token_entry_without_duration)
        + "\n"
    )

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    js = stats_mod.JournalStats()
    js.scan(str(journal))

    usage_with_duration = js.token_usage["20240101"]["gemini-2.5-flash"]
    assert usage_with_duration["input_tokens"] == 100
    assert usage_with_duration["output_tokens"] == 50
    assert usage_with_duration["total_tokens"] == 150
    assert usage_with_duration["duration_ms"] == 3000
    assert "type" not in usage_with_duration

    usage_without_duration = js.token_usage["20240101"]["gemini-2.5-pro"]
    assert usage_without_duration["input_tokens"] == 80
    assert usage_without_duration["output_tokens"] == 20
    assert usage_without_duration["total_tokens"] == 100
    assert "duration_ms" not in usage_without_duration
    assert "type" not in usage_without_duration
