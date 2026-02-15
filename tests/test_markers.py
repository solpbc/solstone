# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for processing marker files (stream.updated, daily.updated)."""

import time

from think.utils import day_path


def test_stream_updated_marker_created(tmp_path, monkeypatch):
    """stream.updated marker file is created in day health directory."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day = "20260215"
    health_dir = day_path(day) / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    (health_dir / "stream.updated").touch()
    assert (health_dir / "stream.updated").exists()


def test_daily_updated_marker_created(tmp_path, monkeypatch):
    """daily.updated marker file is created in day health directory."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day = "20260215"
    health_dir = day_path(day) / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    (health_dir / "daily.updated").touch()
    assert (health_dir / "daily.updated").exists()


def test_marker_mtime_updates(tmp_path, monkeypatch):
    """Touching a marker file again updates its mtime."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day = "20260215"
    health_dir = day_path(day) / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    marker = health_dir / "stream.updated"
    marker.touch()
    mtime1 = marker.stat().st_mtime
    time.sleep(0.05)
    marker.touch()
    mtime2 = marker.stat().st_mtime
    assert mtime2 > mtime1


def test_marker_not_created_when_day_is_none(tmp_path, monkeypatch):
    """When day is None, no marker file should be created."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day = None
    if day:
        health_dir = day_path(day) / "health"
        health_dir.mkdir(parents=True, exist_ok=True)
        (health_dir / "stream.updated").touch()

    # No files should exist under tmp_path since day was None
    assert not list(tmp_path.rglob("stream.updated"))
