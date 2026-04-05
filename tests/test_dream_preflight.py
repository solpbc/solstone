# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for dream preflight skip evaluation."""

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

        assert _should_skip_preflight(
            "observation",
            day="20240115",
            segment=None,
            stream=None,
        ) == (False, None)

    def test_firstday_checkin_not_complete(self, monkeypatch):
        from think import awareness
        from think.dream import _should_skip_preflight

        monkeypatch.setattr(
            awareness, "get_onboarding", lambda: {"status": "observing"}
        )
        assert _should_skip_preflight(
            "firstday_checkin",
            day="20240115",
            segment="120000_300",
            stream="default",
        ) == (True, "preflight:not_complete")

    def test_firstday_checkin_already_sent(self, monkeypatch):
        from think import awareness
        from think.dream import _should_skip_preflight

        monkeypatch.setattr(
            awareness,
            "get_onboarding",
            lambda: {
                "status": "complete",
                "firstday_checkin_sent": "20260402T10:00:00",
            },
        )
        assert _should_skip_preflight(
            "firstday_checkin",
            day="20240115",
            segment="120000_300",
            stream="default",
        ) == (True, "preflight:already_sent")

    def test_observation_not_observing(self, monkeypatch):
        from think import awareness
        from think.dream import _should_skip_preflight

        monkeypatch.setattr(awareness, "get_onboarding", lambda: {"status": "complete"})
        assert _should_skip_preflight(
            "observation",
            day="20240115",
            segment="120000_300",
            stream="default",
        ) == (True, "preflight:not_observing")
