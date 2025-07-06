import importlib
from pathlib import Path


def test_cluster(tmp_path, monkeypatch):
    mod = importlib.import_module("think.cluster")
    day_dir = tmp_path / "20240101"
    day_dir.mkdir()
    (day_dir / "120000_audio.json").write_text('{"text": "hi"}')
    (day_dir / "120500_screen.md").write_text("screen summary")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    result, count = mod.cluster("20240101")
    assert count == 2
    assert "Audio Transcript" in result
    assert "Screen Activity Summary" in result


def test_cluster_range(tmp_path, monkeypatch):
    mod = importlib.import_module("think.cluster")
    day_dir = tmp_path / "20240101"
    day_dir.mkdir()
    (day_dir / "120000_audio.json").write_text('{"text": "hi"}')
    (day_dir / "120000_monitor_1_diff.json").write_text("{}")
    (day_dir / "120000_screen.md").write_text("screen summary")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    md = mod.cluster_range("20240101", "120000", "120100", audio=True, screen="raw")
    assert "Monitor 1" in md
    assert "Audio Transcript" in md
