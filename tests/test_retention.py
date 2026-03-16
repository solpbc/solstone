# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.retention — media retention service."""

import json
from pathlib import Path

import pytest

from think.retention import (
    RetentionConfig,
    RetentionPolicy,
    StorageSummary,
    _human_bytes,
    get_raw_media_files,
    is_raw_media,
    is_segment_complete,
    load_retention_config,
    purge,
)


# ---------------------------------------------------------------------------
# is_raw_media
# ---------------------------------------------------------------------------


class TestIsRawMedia:
    def test_audio_extensions(self, tmp_path):
        for ext in (".flac", ".opus", ".ogg", ".m4a"):
            p = tmp_path / f"audio{ext}"
            p.touch()
            assert is_raw_media(p), f"{ext} should be raw media"

    def test_video_extensions(self, tmp_path):
        for ext in (".webm", ".mov"):
            p = tmp_path / f"screen{ext}"
            p.touch()
            assert is_raw_media(p), f"{ext} should be raw media"

    def test_monitor_diff_png(self, tmp_path):
        p = tmp_path / "monitor_1_diff.png"
        p.touch()
        assert is_raw_media(p)

        p2 = tmp_path / "monitor_2_diff.png"
        p2.touch()
        assert is_raw_media(p2)

    def test_not_raw_media(self, tmp_path):
        for name in (
            "audio.jsonl",
            "screen.jsonl",
            "stream.json",
            "speaker_labels.json",
            "audio.npz",
            "summary.md",
            "regular.png",
        ):
            p = tmp_path / name
            p.touch()
            assert not is_raw_media(p), f"{name} should NOT be raw media"


# ---------------------------------------------------------------------------
# get_raw_media_files
# ---------------------------------------------------------------------------


class TestGetRawMediaFiles:
    def test_returns_only_raw(self, tmp_path):
        (tmp_path / "audio.flac").write_bytes(b"x" * 100)
        (tmp_path / "screen.webm").write_bytes(b"x" * 200)
        (tmp_path / "audio.jsonl").write_text("transcript")
        (tmp_path / "stream.json").write_text("{}")

        raw = get_raw_media_files(tmp_path)
        names = {f.name for f in raw}
        assert names == {"audio.flac", "screen.webm"}

    def test_empty_dir(self, tmp_path):
        assert get_raw_media_files(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path):
        assert get_raw_media_files(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# is_segment_complete
# ---------------------------------------------------------------------------


def _make_segment(tmp_path, *, audio=False, video=False, embeddings=False,
                  audio_extract=True, screen_extract=True, speaker_labels=True,
                  active_agents=False):
    """Create a segment directory with specified contents."""
    seg = tmp_path / "segment"
    seg.mkdir(exist_ok=True)
    agents_dir = seg / "agents"
    agents_dir.mkdir(exist_ok=True)

    if audio:
        (seg / "audio.flac").write_bytes(b"audio")
    if video:
        (seg / "screen.webm").write_bytes(b"video")
    if embeddings:
        (seg / "audio.npz").write_bytes(b"npz")
    if audio and audio_extract:
        (seg / "audio.jsonl").write_text('{"raw":"audio.flac"}\n')
    if video and screen_extract:
        (seg / "screen.jsonl").write_text('{"raw":"screen.webm"}\n')
    if embeddings and speaker_labels:
        (agents_dir / "speaker_labels.json").write_text("{}")
    if active_agents:
        (agents_dir / "1234_active.jsonl").write_text("{}")

    (seg / "stream.json").write_text('{"stream":"default"}')
    return seg


class TestIsSegmentComplete:
    def test_complete_audio_video(self, tmp_path):
        seg = _make_segment(tmp_path, audio=True, video=True, embeddings=True)
        assert is_segment_complete(seg)

    def test_complete_audio_only(self, tmp_path):
        seg = _make_segment(tmp_path, audio=True)
        assert is_segment_complete(seg)

    def test_complete_video_only(self, tmp_path):
        seg = _make_segment(tmp_path, video=True)
        assert is_segment_complete(seg)

    def test_incomplete_missing_audio_extract(self, tmp_path):
        seg = _make_segment(tmp_path, audio=True, audio_extract=False)
        assert not is_segment_complete(seg)

    def test_incomplete_missing_screen_extract(self, tmp_path):
        seg = _make_segment(tmp_path, video=True, screen_extract=False)
        assert not is_segment_complete(seg)

    def test_incomplete_missing_speaker_labels(self, tmp_path):
        seg = _make_segment(tmp_path, audio=True, embeddings=True,
                           speaker_labels=False)
        assert not is_segment_complete(seg)

    def test_incomplete_active_agents(self, tmp_path):
        seg = _make_segment(tmp_path, audio=True, active_agents=True)
        assert not is_segment_complete(seg)

    def test_no_raw_media_is_complete(self, tmp_path):
        """Segment with only derived content is considered complete."""
        seg = tmp_path / "segment"
        seg.mkdir()
        (seg / "audio.jsonl").write_text("transcript")
        (seg / "stream.json").write_text("{}")
        assert is_segment_complete(seg)

    def test_no_agents_dir_is_ok(self, tmp_path):
        """No agents/ directory = no active agents = passes check 1."""
        seg = tmp_path / "segment"
        seg.mkdir()
        (seg / "stream.json").write_text("{}")
        assert is_segment_complete(seg)


# ---------------------------------------------------------------------------
# RetentionPolicy
# ---------------------------------------------------------------------------


class TestRetentionPolicy:
    def test_keep_never_eligible(self):
        p = RetentionPolicy(mode="keep")
        assert not p.is_eligible(0)
        assert not p.is_eligible(365)

    def test_processed_always_eligible(self):
        p = RetentionPolicy(mode="processed")
        assert p.is_eligible(0)
        assert p.is_eligible(1)

    def test_days_threshold(self):
        p = RetentionPolicy(mode="days", days=30)
        assert not p.is_eligible(29)
        assert p.is_eligible(30)
        assert p.is_eligible(31)

    def test_days_no_value(self):
        p = RetentionPolicy(mode="days", days=None)
        assert not p.is_eligible(100)


class TestRetentionConfig:
    def test_default_policy(self):
        cfg = RetentionConfig()
        assert cfg.policy_for_stream("default").mode == "keep"

    def test_per_stream_override(self):
        cfg = RetentionConfig(
            default=RetentionPolicy(mode="keep"),
            per_stream={
                "archon.plaud": RetentionPolicy(mode="days", days=7),
            },
        )
        assert cfg.policy_for_stream("archon.plaud").mode == "days"
        assert cfg.policy_for_stream("archon.plaud").days == 7
        assert cfg.policy_for_stream("default").mode == "keep"


# ---------------------------------------------------------------------------
# load_retention_config
# ---------------------------------------------------------------------------


class TestLoadRetentionConfig:
    def test_default_config(self, monkeypatch):
        monkeypatch.setattr("think.utils.get_config", lambda: {})
        cfg = load_retention_config()
        assert cfg.default.mode == "keep"
        assert cfg.per_stream == {}

    def test_custom_config(self, monkeypatch):
        monkeypatch.setattr(
            "think.utils.get_config",
            lambda: {
                "retention": {
                    "raw_media": "days",
                    "raw_media_days": 30,
                    "per_stream": {
                        "default": {"raw_media": "processed"},
                    },
                }
            },
        )
        cfg = load_retention_config()
        assert cfg.default.mode == "days"
        assert cfg.default.days == 30
        assert cfg.per_stream["default"].mode == "processed"


# ---------------------------------------------------------------------------
# purge
# ---------------------------------------------------------------------------


class TestPurge:
    def _setup_journal(self, tmp_path, monkeypatch):
        """Create a journal structure with test segments."""
        journal = tmp_path / "journal"

        # Day 1: 60 days old — two complete segments
        day1 = journal / "20260115" / "default" / "100000_300"
        day1.mkdir(parents=True)
        (day1 / "audio.flac").write_bytes(b"x" * 1000)
        (day1 / "audio.jsonl").write_text('{"raw":"audio.flac"}\n')
        (day1 / "stream.json").write_text('{"stream":"default"}')
        (day1 / "agents").mkdir()

        day1b = journal / "20260115" / "plaud" / "103000_300"
        day1b.mkdir(parents=True)
        (day1b / "audio.m4a").write_bytes(b"x" * 500)
        (day1b / "audio.jsonl").write_text('{"raw":"audio.m4a"}\n')
        (day1b / "stream.json").write_text('{"stream":"plaud"}')
        (day1b / "agents").mkdir()

        # Day 2: 10 days old — one complete segment
        day2 = journal / "20260306" / "default" / "120000_300"
        day2.mkdir(parents=True)
        (day2 / "audio.flac").write_bytes(b"x" * 800)
        (day2 / "audio.jsonl").write_text('{"raw":"audio.flac"}\n')
        (day2 / "stream.json").write_text('{"stream":"default"}')
        (day2 / "agents").mkdir()

        # Day 3: incomplete segment (no audio.jsonl)
        day3 = journal / "20260101" / "default" / "140000_300"
        day3.mkdir(parents=True)
        (day3 / "audio.flac").write_bytes(b"x" * 600)
        (day3 / "stream.json").write_text('{"stream":"default"}')

        monkeypatch.setenv("JOURNAL_PATH", str(journal))
        # Clear cached journal path
        import think.utils
        think.utils._journal_path_cache = None

        return journal

    def test_dry_run(self, tmp_path, monkeypatch):
        journal = self._setup_journal(tmp_path, monkeypatch)

        result = purge(older_than_days=30, dry_run=True)

        # Should report but not delete
        assert result.files_deleted == 2  # day1 default + plaud
        assert result.bytes_freed == 1500
        assert (journal / "20260115" / "default" / "100000_300" / "audio.flac").exists()
        assert (journal / "20260115" / "plaud" / "103000_300" / "audio.m4a").exists()
        # No retention log for dry run
        assert not (journal / "health" / "retention.log").exists()

    def test_actual_purge(self, tmp_path, monkeypatch):
        journal = self._setup_journal(tmp_path, monkeypatch)

        result = purge(older_than_days=30, dry_run=False)

        assert result.files_deleted == 2
        # Files should be gone
        assert not (journal / "20260115" / "default" / "100000_300" / "audio.flac").exists()
        assert not (journal / "20260115" / "plaud" / "103000_300" / "audio.m4a").exists()
        # Derived content preserved
        assert (journal / "20260115" / "default" / "100000_300" / "audio.jsonl").exists()
        # Retention log written
        assert (journal / "health" / "retention.log").exists()

    def test_skips_incomplete(self, tmp_path, monkeypatch):
        self._setup_journal(tmp_path, monkeypatch)

        result = purge(older_than_days=0, dry_run=True)

        # Day3 segment should be skipped (incomplete)
        assert result.segments_skipped_incomplete == 1

    def test_stream_filter(self, tmp_path, monkeypatch):
        self._setup_journal(tmp_path, monkeypatch)

        result = purge(older_than_days=30, stream_filter="plaud", dry_run=True)

        assert result.files_deleted == 1
        assert result.details[0]["stream"] == "plaud"

    def test_policy_based_purge(self, tmp_path, monkeypatch):
        self._setup_journal(tmp_path, monkeypatch)

        config = RetentionConfig(
            default=RetentionPolicy(mode="keep"),
            per_stream={
                "plaud": RetentionPolicy(mode="days", days=7),
            },
        )

        result = purge(dry_run=True, config=config)

        # Only plaud segment (60 days old) should be eligible
        assert result.files_deleted == 1
        assert result.details[0]["stream"] == "plaud"


# ---------------------------------------------------------------------------
# _human_bytes
# ---------------------------------------------------------------------------


class TestHumanBytes:
    def test_bytes(self):
        assert _human_bytes(0) == "0 B"
        assert _human_bytes(512) == "512 B"

    def test_kilobytes(self):
        assert _human_bytes(1024) == "1.0 KB"

    def test_megabytes(self):
        assert _human_bytes(1024 * 1024) == "1.0 MB"

    def test_gigabytes(self):
        assert _human_bytes(1024 ** 3) == "1.0 GB"

    def test_large(self):
        result = _human_bytes(12_400_000_000)
        assert "GB" in result
