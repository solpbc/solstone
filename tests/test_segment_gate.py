# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for segment density classification."""

from __future__ import annotations

import json


def test_missing_segment_dir_is_active(tmp_path, monkeypatch):
    from think.dream import _classify_segment_density

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path / "journal"))
    assert _classify_segment_density("20240115", "120000_300", "default") == "active"


def test_idle_segment(tmp_path, monkeypatch):
    from think.dream import _classify_segment_density

    seg_dir = tmp_path / "journal" / "20240115" / "default" / "120000_300"
    seg_dir.mkdir(parents=True)
    (seg_dir / "audio.jsonl").write_text(json.dumps({"raw": "audio.flac"}) + "\n")

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path / "journal"))
    assert _classify_segment_density("20240115", "120000_300", "default") == "idle"


def test_low_change_segment(tmp_path, monkeypatch):
    from think.dream import _classify_segment_density

    seg_dir = tmp_path / "journal" / "20240115" / "default" / "120000_300"
    seg_dir.mkdir(parents=True)
    audio_lines = [json.dumps({"raw": "audio.flac"})]
    audio_lines.extend(
        json.dumps({"start": f"00:00:0{i}", "text": "line"}) for i in range(1, 9)
    )
    (seg_dir / "audio.jsonl").write_text("\n".join(audio_lines) + "\n")
    screen_lines = [json.dumps({"raw": "screen.webm"})]
    screen_lines.extend(
        json.dumps({"timestamp": i, "analysis": {"visual_description": "screen"}})
        for i in range(3)
    )
    (seg_dir / "screen.jsonl").write_text("\n".join(screen_lines) + "\n")

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path / "journal"))
    assert _classify_segment_density("20240115", "120000_300", "default") == "low_change"


def test_active_segment_with_imported_md(tmp_path, monkeypatch):
    from think.dream import _classify_segment_density

    seg_dir = tmp_path / "journal" / "20240115" / "default" / "120000_300"
    seg_dir.mkdir(parents=True)
    (seg_dir / "imported.md").write_text("\n".join(f"line {i}" for i in range(12)) + "\n")

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path / "journal"))
    assert _classify_segment_density("20240115", "120000_300", "default") == "active"


def test_tmux_screen_has_no_header(tmp_path, monkeypatch):
    from think.dream import _classify_segment_density

    seg_dir = tmp_path / "journal" / "20240115" / "default" / "120000_300"
    seg_dir.mkdir(parents=True)
    screen_lines = [
        json.dumps({"frame_id": 1, "timestamp": 0, "analysis": {"primary": "tmux"}}),
        json.dumps({"frame_id": 2, "timestamp": 1, "analysis": {"primary": "tmux"}}),
    ]
    (seg_dir / "tmux_0_screen.jsonl").write_text("\n".join(screen_lines) + "\n")

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path / "journal"))
    assert _classify_segment_density("20240115", "120000_300", "default") == "low_change"
