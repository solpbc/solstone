# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from convey.utils import format_date, format_date_short, time_since
from think.utils import day_path


def test_format_date():
    assert "2024" not in format_date("20240102")
    assert format_date("bad") == "bad"


def test_format_date_short(monkeypatch):
    from datetime import datetime

    # Mock today as Nov 29, 2025
    class MockDatetime(datetime):
        @classmethod
        def now(cls):
            return datetime(2025, 11, 29, 12, 0, 0)

    monkeypatch.setattr("convey.utils.datetime", MockDatetime)

    # Test relative dates
    assert format_date_short("20251129") == "Today"
    assert format_date_short("20251128") == "Yesterday"
    assert format_date_short("20251130") == "Tomorrow"

    # Test within past 6 days - should return day name
    assert format_date_short("20251127") == "Thursday"
    assert format_date_short("20251124") == "Monday"

    # Test older date same year - short format without year
    result = format_date_short("20250815")
    assert "Aug" in result
    assert "15" in result
    assert "'" not in result  # No year suffix

    # Test date >6 months ago in different year - should have year suffix
    result = format_date_short("20240301")
    assert "Mar" in result
    assert "'24" in result

    # Test invalid date - should return input unchanged
    assert format_date_short("bad") == "bad"


def test_time_since(monkeypatch):
    monkeypatch.setattr("time.time", lambda: 120)
    assert time_since(60) == "1 minute ago"


def test_list_day_folders(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    from think.utils import day_dirs

    day_path("20240101")
    day_path("20240103")
    days = sorted(day_dirs().keys())
    assert days == ["20240101", "20240103"]
