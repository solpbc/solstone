# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.retention — media retention service."""

import hashlib
import json
import os
import shutil
from datetime import datetime

from solstone.think.retention import (
    RetentionConfig,
    RetentionPolicy,
    StorageSummary,
    _human_bytes,
    check_storage_health,
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
        for ext in (".flac", ".opus", ".ogg", ".m4a", ".mp3", ".wav"):
            p = tmp_path / f"audio{ext}"
            p.touch()
            assert is_raw_media(p), f"{ext} should be raw media"

    def test_video_extensions(self, tmp_path):
        for ext in (".webm", ".mov", ".mp4"):
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


def _make_segment(
    tmp_path,
    *,
    audio=False,
    video=False,
    video_name="screen.webm",
    embeddings=False,
    audio_extract=True,
    screen_extract=True,
    speaker_labels=True,
    active_agents=False,
):
    """Create a segment directory with specified contents."""
    seg = tmp_path / "segment"
    seg.mkdir(exist_ok=True)
    agents_dir = seg / "talents"
    agents_dir.mkdir(exist_ok=True)

    if audio:
        (seg / "audio.flac").write_bytes(b"audio")
    if video:
        (seg / video_name).write_bytes(b"video")
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

    def test_incomplete_missing_screen_extract_for_mp4(self, tmp_path):
        seg = _make_segment(
            tmp_path, video=True, video_name="screen.mp4", screen_extract=False
        )
        assert not is_segment_complete(seg)

    def test_complete_mp4_with_screen_extract(self, tmp_path):
        seg = _make_segment(tmp_path, video=True, video_name="screen.mp4")
        assert is_segment_complete(seg)

    def test_incomplete_missing_speaker_labels(self, tmp_path):
        seg = _make_segment(tmp_path, audio=True, embeddings=True, speaker_labels=False)
        assert not is_segment_complete(seg)

    def test_complete_with_stub_speaker_labels(self, tmp_path):
        """Stub speaker_labels.json (skipped=True, labels=[]) unblocks retention."""
        seg = _make_segment(tmp_path, audio=True, embeddings=True, speaker_labels=False)
        stub = seg / "talents" / "speaker_labels.json"
        stub.write_text(
            json.dumps({"labels": [], "skipped": True, "reason": "no_owner_centroid"})
        )
        assert is_segment_complete(seg)

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
        assert cfg.policy_for_stream("default").days is None

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
        monkeypatch.setattr("solstone.think.utils.get_config", lambda: {})
        cfg = load_retention_config()
        assert cfg.default.mode == "keep"
        assert cfg.default.days is None
        assert cfg.per_stream == {}

    def test_custom_config(self, monkeypatch):
        monkeypatch.setattr(
            "solstone.think.utils.get_config",
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

    def test_existing_journal_days_config_unchanged(self, monkeypatch):
        monkeypatch.setattr(
            "solstone.think.utils.get_config",
            lambda: {
                "retention": {
                    "raw_media": "days",
                    "raw_media_days": 7,
                }
            },
        )
        cfg = load_retention_config()
        assert cfg.default.mode == "days"
        assert cfg.default.days == 7


# ---------------------------------------------------------------------------
# purge
# ---------------------------------------------------------------------------


class TestPurge:
    def _setup_journal(self, tmp_path, monkeypatch):
        """Create a journal structure with test segments."""
        journal = tmp_path / "journal"

        # Day 1: 60 days old — two complete segments
        day1 = journal / "chronicle" / "20260115" / "default" / "100000_300"
        day1.mkdir(parents=True)
        (day1 / "audio.flac").write_bytes(b"x" * 1000)
        (day1 / "audio.jsonl").write_text('{"raw":"audio.flac"}\n')
        (day1 / "stream.json").write_text('{"stream":"default"}')
        (day1 / "talents").mkdir()

        day1b = journal / "chronicle" / "20260115" / "plaud" / "103000_300"
        day1b.mkdir(parents=True)
        (day1b / "audio.m4a").write_bytes(b"x" * 500)
        (day1b / "audio.jsonl").write_text('{"raw":"audio.m4a"}\n')
        (day1b / "stream.json").write_text('{"stream":"plaud"}')
        (day1b / "talents").mkdir()

        # Day 2: recent — one complete segment (must stay within 30d window)
        day2 = journal / "chronicle" / "20260401" / "default" / "120000_300"
        day2.mkdir(parents=True)
        (day2 / "audio.flac").write_bytes(b"x" * 800)
        (day2 / "audio.jsonl").write_text('{"raw":"audio.flac"}\n')
        (day2 / "stream.json").write_text('{"stream":"default"}')
        (day2 / "talents").mkdir()

        # Day 3: incomplete segment (no audio.jsonl)
        day3 = journal / "chronicle" / "20260101" / "default" / "140000_300"
        day3.mkdir(parents=True)
        (day3 / "audio.flac").write_bytes(b"x" * 600)
        (day3 / "stream.json").write_text('{"stream":"default"}')

        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))
        fixed_now = datetime(2026, 4, 15)

        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is not None:
                    return fixed_now.replace(tzinfo=tz)
                return fixed_now

        monkeypatch.setattr("solstone.think.retention.datetime", FixedDateTime)
        # Clear cached journal path
        import solstone.think.utils as think_utils

        think_utils._journal_path_cache = None

        return journal

    def test_dry_run(self, tmp_path, monkeypatch):
        journal = self._setup_journal(tmp_path, monkeypatch)

        result = purge(older_than_days=30, dry_run=True)

        # Should report but not delete
        assert result.files_deleted == 2  # day1 default + plaud
        assert result.bytes_freed == 1500
        assert (
            journal / "chronicle" / "20260115" / "default" / "100000_300" / "audio.flac"
        ).exists()
        assert (
            journal / "chronicle" / "20260115" / "plaud" / "103000_300" / "audio.m4a"
        ).exists()
        # No retention log for dry run
        assert not (journal / "health" / "retention.log").exists()

    def test_actual_purge(self, tmp_path, monkeypatch):
        journal = self._setup_journal(tmp_path, monkeypatch)

        result = purge(older_than_days=30, dry_run=False)

        assert result.files_deleted == 2
        # Files should be gone
        assert not (
            journal / "chronicle" / "20260115" / "default" / "100000_300" / "audio.flac"
        ).exists()
        assert not (
            journal / "chronicle" / "20260115" / "plaud" / "103000_300" / "audio.m4a"
        ).exists()
        # Derived content preserved
        assert (
            journal
            / "chronicle"
            / "20260115"
            / "default"
            / "100000_300"
            / "audio.jsonl"
        ).exists()
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


class TestPurgeProvenance:
    def _setup_journal(self, tmp_path, monkeypatch):
        return TestPurge()._setup_journal(tmp_path, monkeypatch)

    def test_hash_field_in_dry_run(self, tmp_path, monkeypatch):
        self._setup_journal(tmp_path, monkeypatch)

        result = purge(older_than_days=30, dry_run=True)
        expected_hash = hashlib.sha256(b"x" * 1000).hexdigest()

        for detail in result.details:
            for file_info in detail["files"]:
                file_hash = file_info["hash"]
                assert len(file_hash) == 64
                assert all(c in "0123456789abcdef" for c in file_hash)

        default_detail = next(
            detail
            for detail in result.details
            if detail["stream"] == "default" and detail["segment"] == "100000_300"
        )
        assert default_detail["files"][0]["hash"] == expected_hash

    def test_hash_field_in_actual_purge(self, tmp_path, monkeypatch):
        self._setup_journal(tmp_path, monkeypatch)

        result = purge(older_than_days=30, dry_run=False)
        expected_hash = hashlib.sha256(b"x" * 1000).hexdigest()

        for detail in result.details:
            for file_info in detail["files"]:
                file_hash = file_info["hash"]
                assert len(file_hash) == 64
                assert all(c in "0123456789abcdef" for c in file_hash)

        default_detail = next(
            detail
            for detail in result.details
            if detail["stream"] == "default" and detail["segment"] == "100000_300"
        )
        assert default_detail["files"][0]["hash"] == expected_hash

    def test_processed_at_field(self, tmp_path, monkeypatch):
        self._setup_journal(tmp_path, monkeypatch)

        result = purge(older_than_days=30, dry_run=True)

        for detail in result.details:
            assert "processed_at" in detail
            assert isinstance(detail["processed_at"], str)
            datetime.fromisoformat(detail["processed_at"])

    def test_processed_at_reflects_latest_mtime(self, tmp_path, monkeypatch):
        journal = self._setup_journal(tmp_path, monkeypatch)
        segment = journal / "chronicle" / "20260115" / "default" / "100000_300"
        audio_jsonl = segment / "audio.jsonl"
        alternate_audio_jsonl = segment / "meeting_audio.jsonl"
        speaker_labels = segment / "talents" / "speaker_labels.json"

        alternate_audio_jsonl.write_text('{"raw":"audio.flac"}\n')
        speaker_labels.write_text("{}")

        older_ts = datetime(2026, 1, 15, 10, 0, 0).timestamp()
        middle_ts = datetime(2026, 1, 15, 11, 0, 0).timestamp()
        latest_ts = datetime(2026, 1, 15, 12, 0, 0).timestamp()

        os.utime(audio_jsonl, (older_ts, older_ts))
        os.utime(speaker_labels, (middle_ts, middle_ts))
        os.utime(alternate_audio_jsonl, (latest_ts, latest_ts))

        result = purge(older_than_days=30, dry_run=True)

        default_detail = next(
            detail
            for detail in result.details
            if detail["stream"] == "default" and detail["segment"] == "100000_300"
        )
        assert (
            default_detail["processed_at"]
            == datetime.fromtimestamp(latest_ts).isoformat()
        )


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
        assert _human_bytes(1024**3) == "1.0 GB"

    def test_large(self):
        result = _human_bytes(12_400_000_000)
        assert "GB" in result


class TestCheckStorageHealth:
    """Tests for check_storage_health threshold evaluation."""

    def _make_summary(self, raw_media_bytes=0, derived_bytes=0):
        return StorageSummary(
            raw_media_bytes=raw_media_bytes,
            derived_bytes=derived_bytes,
            total_segments=10,
            segments_with_raw=5,
            segments_purged=3,
        )

    def test_no_warnings_when_healthy(self, tmp_path, monkeypatch):
        """No warnings when disk is below threshold and raw media GB is null."""
        usage_type = type(shutil.disk_usage(tmp_path))
        monkeypatch.setattr(
            "shutil.disk_usage",
            lambda path: usage_type(1000, 500, 500),  # 50% used
        )
        config = {
            "retention": {
                "storage_warning_disk_percent": 80,
                "storage_warning_raw_media_gb": None,
            }
        }
        summary = self._make_summary()
        warnings = check_storage_health(summary, tmp_path, config=config)
        assert warnings == []

    def test_disk_percent_exceeded(self, tmp_path, monkeypatch):
        """Warning when disk usage exceeds threshold."""
        config = {
            "retention": {
                "storage_warning_disk_percent": 1,
            }
        }
        summary = self._make_summary()
        warnings = check_storage_health(summary, tmp_path, config=config)
        assert len(warnings) == 1
        assert warnings[0]["type"] == "disk_percent"
        assert warnings[0]["level"] == "warning"
        assert warnings[0]["current"] >= 1
        assert warnings[0]["threshold"] == 1
        assert "retention settings" in warnings[0]["message"]
        assert "Clean Up Now" in warnings[0]["message"]

    def test_disk_percent_not_exceeded(self, tmp_path, monkeypatch):
        """No warning when disk is well below threshold."""
        config = {
            "retention": {
                "storage_warning_disk_percent": 100,
            }
        }
        summary = self._make_summary()
        warnings = check_storage_health(summary, tmp_path, config=config)
        assert warnings == []

    def test_raw_media_gb_exceeded(self, tmp_path, monkeypatch):
        """Warning when raw media exceeds GB threshold."""
        raw_bytes = int(5.5 * 1024**3)
        config = {
            "retention": {
                "storage_warning_disk_percent": None,
                "storage_warning_raw_media_gb": 5.0,
            }
        }
        summary = self._make_summary(raw_media_bytes=raw_bytes)
        warnings = check_storage_health(summary, tmp_path, config=config)
        assert len(warnings) == 1
        assert warnings[0]["type"] == "raw_media_gb"
        assert warnings[0]["level"] == "warning"
        assert warnings[0]["current"] >= 5.0
        assert warnings[0]["threshold"] == 5.0
        assert "retention settings" in warnings[0]["message"]

    def test_raw_media_gb_not_exceeded(self, tmp_path, monkeypatch):
        """No warning when raw media is below threshold."""
        raw_bytes = int(2.0 * 1024**3)
        config = {
            "retention": {
                "storage_warning_disk_percent": None,
                "storage_warning_raw_media_gb": 5.0,
            }
        }
        summary = self._make_summary(raw_media_bytes=raw_bytes)
        warnings = check_storage_health(summary, tmp_path, config=config)
        assert warnings == []

    def test_both_thresholds_exceeded(self, tmp_path, monkeypatch):
        """Both warnings when both thresholds exceeded."""
        raw_bytes = int(10 * 1024**3)
        config = {
            "retention": {
                "storage_warning_disk_percent": 1,
                "storage_warning_raw_media_gb": 5.0,
            }
        }
        summary = self._make_summary(raw_media_bytes=raw_bytes)
        warnings = check_storage_health(summary, tmp_path, config=config)
        assert len(warnings) == 2
        types = {w["type"] for w in warnings}
        assert types == {"disk_percent", "raw_media_gb"}

    def test_null_thresholds_disables_checks(self, tmp_path, monkeypatch):
        """Both thresholds null means no warnings ever."""
        raw_bytes = int(100 * 1024**3)
        config = {
            "retention": {
                "storage_warning_disk_percent": None,
                "storage_warning_raw_media_gb": None,
            }
        }
        summary = self._make_summary(raw_media_bytes=raw_bytes)
        warnings = check_storage_health(summary, tmp_path, config=config)
        assert warnings == []

    def test_exact_threshold_triggers(self, tmp_path, monkeypatch):
        """Warning triggers at exactly the threshold (>=, not >)."""
        raw_bytes = int(5.0 * 1024**3)
        config = {
            "retention": {
                "storage_warning_disk_percent": None,
                "storage_warning_raw_media_gb": 5.0,
            }
        }
        summary = self._make_summary(raw_media_bytes=raw_bytes)
        warnings = check_storage_health(summary, tmp_path, config=config)
        assert len(warnings) == 1
        assert warnings[0]["type"] == "raw_media_gb"

    def test_missing_retention_section_uses_defaults(self, tmp_path, monkeypatch):
        """Missing retention section falls back to defaults (80% disk, null raw media)."""
        config = {}
        summary = self._make_summary()
        warnings = check_storage_health(summary, tmp_path, config=config)
        for w in warnings:
            assert w["type"] != "raw_media_gb"

    def test_warning_dict_structure(self, tmp_path, monkeypatch):
        """Each warning has all required keys."""
        config = {
            "retention": {
                "storage_warning_disk_percent": 1,
                "storage_warning_raw_media_gb": 0.001,
            }
        }
        raw_bytes = int(1 * 1024**3)
        summary = self._make_summary(raw_media_bytes=raw_bytes)
        warnings = check_storage_health(summary, tmp_path, config=config)
        for w in warnings:
            assert "level" in w
            assert "type" in w
            assert "message" in w
            assert "current" in w
            assert "threshold" in w


class TestStorageHealthNudge:
    def _make_summary(self):
        return StorageSummary(
            raw_media_bytes=int(10 * 1024**3),
            derived_bytes=0,
            total_segments=10,
            segments_with_raw=5,
            segments_purged=3,
        )

    def _config(self, mode: str) -> dict:
        return {
            "retention": {
                "raw_media": mode,
                "storage_warning_disk_percent": 1,
                "storage_warning_raw_media_gb": 5.0,
            }
        }

    def _force_disk_warning(self, tmp_path, monkeypatch) -> None:
        usage_type = type(shutil.disk_usage(tmp_path))
        monkeypatch.setattr(
            "shutil.disk_usage",
            lambda path: usage_type(1000, 950, 50),
        )

    def test_keep_mode_appends_nudge(self, tmp_path, monkeypatch):
        self._force_disk_warning(tmp_path, monkeypatch)
        warnings = check_storage_health(
            self._make_summary(),
            tmp_path,
            config=self._config("keep"),
        )
        assert {warning["type"] for warning in warnings} == {
            "disk_percent",
            "raw_media_gb",
        }
        assert all(
            "always retain observed media" in warning["message"] for warning in warnings
        )

    def test_days_mode_does_not_append_nudge(self, tmp_path, monkeypatch):
        self._force_disk_warning(tmp_path, monkeypatch)
        warnings = check_storage_health(
            self._make_summary(),
            tmp_path,
            config=self._config("days"),
        )
        assert all(
            "always retain observed media" not in warning["message"]
            for warning in warnings
        )

    def test_processed_mode_does_not_append_nudge(self, tmp_path, monkeypatch):
        self._force_disk_warning(tmp_path, monkeypatch)
        warnings = check_storage_health(
            self._make_summary(),
            tmp_path,
            config=self._config("processed"),
        )
        assert all(
            "always retain observed media" not in warning["message"]
            for warning in warnings
        )


class TestRetentionDerivationRule:
    @staticmethod
    def derive_retention(days_value: str, dont_retain: bool) -> tuple[str, int | None]:
        # Mirrors the JS deriveRetention helper in convey/templates/init.html
        # and apps/settings/workspace.html.
        if dont_retain:
            return ("processed", None)
        try:
            days = int(days_value)
        except (TypeError, ValueError):
            days = None
        if days is not None and days >= 1:
            return ("days", days)
        return ("keep", None)

    def test_empty_days_defaults_to_keep(self):
        assert self.derive_retention("", False) == ("keep", None)

    def test_numeric_days_uses_days_mode(self):
        assert self.derive_retention("30", False) == ("days", 30)

    def test_checkbox_wins_over_numeric_days(self):
        assert self.derive_retention("30", True) == ("processed", None)

    def test_checkbox_wins_when_days_empty(self):
        assert self.derive_retention("", True) == ("processed", None)
