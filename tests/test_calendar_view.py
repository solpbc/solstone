import importlib
import os

from think.utils import day_path


def test_calendar_days_api(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_path("20240101")
    day_path("20240102")

    review = importlib.import_module("convey")
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/calendar/api/days"):
        resp = review.calendar_days()
    assert resp.json == ["20240101", "20240102"]


def test_transcript_ranges_api(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    review = importlib.import_module("convey")
    (day_dir / "090101_raw_audio.json").write_text("{}")
    (day_dir / "100101_screen.md").write_text("screen")
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/calendar/api/transcript_ranges/20240101"):
        resp = review.calendar_transcript_ranges("20240101")
    assert resp.json["audio"] == [["09:00", "09:15"]]
    assert resp.json["screen"] == [["10:00", "10:15"]]


def test_transcript_range_api(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    review = importlib.import_module("convey")
    (day_dir / "120000_raw_audio.json").write_text('{"text": "hi"}')
    (day_dir / "120000_screen.md").write_text("screen summary")
    review.journal_root = str(tmp_path)
    with review.app.test_request_context(
        "/calendar/api/transcript/20240101?start=120000&end=120100"
    ):
        resp = review.calendar_transcript_range("20240101")
    assert "Audio Transcript" in resp.json["html"]
    assert "Screen Activity Summary" in resp.json["html"]
