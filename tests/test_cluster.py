import importlib
from pathlib import Path


def test_cluster(tmp_path):
    mod = importlib.import_module("think.cluster")
    day = tmp_path / "20240101"
    day.mkdir()
    (day / "120000_audio.json").write_text('{"text": "hi"}')
    (day / "120500_screen.md").write_text("screen summary")
    result, count = mod.cluster(str(day))
    assert count == 2
    assert "Audio Transcript" in result
    assert "Screen Activity Summary" in result


def test_cluster_range(tmp_path):
    mod = importlib.import_module("think.cluster")
    day = tmp_path / "20240101"
    day.mkdir()
    (day / "120000_audio.json").write_text('{"text": "hi"}')
    (day / "120000_monitor_1_diff.json").write_text("{}")
    (day / "120000_screen.md").write_text("screen summary")

    md = mod.cluster_range(str(day), "120000", "120100", audio=True, screen="raw")
    assert "Monitor 1" in md
    assert "Audio Transcript" in md
