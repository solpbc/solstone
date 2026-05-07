# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for updated_days() utility."""

import time

from solstone.think.utils import updated_days


def test_updated_days_fixture(monkeypatch):
    """20250101 has stream.updated but no daily.updated — should be updated."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", "tests/fixtures/journal")
    days = updated_days()
    assert "20250101" in days


def test_updated_days_exclude(monkeypatch):
    """Excluded days should not appear in results."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", "tests/fixtures/journal")
    days = updated_days(exclude={"20250101"})
    assert "20250101" not in days


def test_updated_days_clean(tmp_path, monkeypatch):
    """Day with daily.updated newer than stream.updated is not updated."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    day_dir = tmp_path / "chronicle" / "20260101" / "health"
    day_dir.mkdir(parents=True)
    (day_dir / "stream.updated").touch()
    time.sleep(0.05)
    (day_dir / "daily.updated").touch()
    assert updated_days() == []


def test_updated_days_no_stream(tmp_path, monkeypatch):
    """Day without stream.updated is not updated (no stream data)."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    (tmp_path / "chronicle" / "20260101").mkdir(parents=True)
    assert updated_days() == []
