# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for dream preflight skip evaluation."""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def segment_dir(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    seg_dir = journal / "20240115" / "default" / "120000_300"
    seg_dir.mkdir(parents=True)
    (seg_dir / "agents").mkdir()
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))
    return seg_dir


class TestShouldSkipPreflight:
    def test_daily_mode_never_skips(self):
        from think.dream import _should_skip_preflight

        assert _should_skip_preflight("observation", day="20240115", segment=None, stream=None) == (
            False,
            None,
        )

    def test_firstday_checkin_not_complete(self, monkeypatch):
        from think import awareness
        from think.dream import _should_skip_preflight

        monkeypatch.setattr(awareness, "get_onboarding", lambda: {"status": "observing"})
        assert _should_skip_preflight(
            "firstday_checkin", day="20240115", segment="120000_300", stream="default"
        ) == (True, "preflight:not_complete")

    def test_firstday_checkin_already_sent(self, monkeypatch):
        from think import awareness
        from think.dream import _should_skip_preflight

        monkeypatch.setattr(
            awareness,
            "get_onboarding",
            lambda: {"status": "complete", "firstday_checkin_sent": "20260402T10:00:00"},
        )
        assert _should_skip_preflight(
            "firstday_checkin", day="20240115", segment="120000_300", stream="default"
        ) == (True, "preflight:already_sent")

    def test_observation_not_observing(self, monkeypatch):
        from think import awareness
        from think.dream import _should_skip_preflight

        monkeypatch.setattr(awareness, "get_onboarding", lambda: {"status": "complete"})
        assert _should_skip_preflight(
            "observation", day="20240115", segment="120000_300", stream="default"
        ) == (True, "preflight:not_observing")

    def test_speaker_attribution_requires_embeddings(self, segment_dir):
        from think.dream import _should_skip_preflight

        assert _should_skip_preflight(
            "speaker_attribution",
            day="20240115",
            segment="120000_300",
            stream="default",
        ) == (True, "preflight:no_embeddings")

        (segment_dir / "audio.npz").write_bytes(b"x")
        assert _should_skip_preflight(
            "speaker_attribution",
            day="20240115",
            segment="120000_300",
            stream="default",
        ) == (False, None)

    def test_speakers_requires_transcripts(self, segment_dir):
        from think.dream import _should_skip_preflight

        assert _should_skip_preflight(
            "speakers",
            day="20240115",
            segment="120000_300",
            stream="default",
        ) == (True, "preflight:no_transcripts")

    def test_speakers_skips_single_speaker(self, segment_dir):
        from think.dream import _should_skip_preflight

        (segment_dir / "audio.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"raw": "audio.flac"}),
                    json.dumps({"start": "00:00:01", "speaker": 1, "text": "hello"}),
                    json.dumps({"start": "00:00:02", "speaker": 1, "text": "again"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        assert _should_skip_preflight(
            "speakers",
            day="20240115",
            segment="120000_300",
            stream="default",
        ) == (True, "preflight:single_speaker")

    def test_speakers_runs_for_multiple_speakers(self, segment_dir):
        from think.dream import _should_skip_preflight

        (segment_dir / "audio.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"raw": "audio.flac"}),
                    json.dumps({"start": "00:00:01", "speaker": 1, "text": "hello"}),
                    json.dumps({"start": "00:00:02", "speaker": 2, "text": "hi"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        assert _should_skip_preflight(
            "speakers",
            day="20240115",
            segment="120000_300",
            stream="default",
        ) == (False, None)

    def test_activities_requires_previous_segment(self, segment_dir):
        from think.dream import _should_skip_preflight

        assert _should_skip_preflight(
            "activities",
            day="20240115",
            segment="120000_300",
            stream="default",
        ) == (True, "preflight:no_previous_segment")

    def test_activities_requires_previous_activity_state(self, tmp_path, monkeypatch):
        from think.dream import _should_skip_preflight

        journal = tmp_path / "journal"
        prev_dir = journal / "20240115" / "default" / "110000_300"
        curr_dir = journal / "20240115" / "default" / "120000_300"
        prev_dir.mkdir(parents=True)
        curr_dir.mkdir(parents=True)
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))

        assert _should_skip_preflight(
            "activities",
            day="20240115",
            segment="120000_300",
            stream="default",
        ) == (True, "preflight:no_previous_activity_state")

    def test_activities_runs_with_previous_activity_state(self, tmp_path, monkeypatch):
        from think.dream import _should_skip_preflight

        journal = tmp_path / "journal"
        prev_dir = journal / "20240115" / "default" / "110000_300" / "agents" / "work"
        curr_dir = journal / "20240115" / "default" / "120000_300"
        prev_dir.mkdir(parents=True)
        curr_dir.mkdir(parents=True)
        (prev_dir / "activity_state.json").write_text("[]", encoding="utf-8")
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))

        assert _should_skip_preflight(
            "activities",
            day="20240115",
            segment="120000_300",
            stream="default",
        ) == (False, None)
