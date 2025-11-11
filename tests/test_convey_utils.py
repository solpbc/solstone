import json
import os

from convey.utils import format_date, time_since
from think.utils import day_path


def test_format_date():
    assert "2024" not in format_date("20240102")
    assert format_date("bad") == "bad"


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
