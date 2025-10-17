import importlib

from think.utils import day_path


def test_cluster(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")
    # Write JSONL format: metadata first, then entry
    (day_dir / "120000_audio.jsonl").write_text('{}\n{"text": "hi"}\n')
    (day_dir / "120500_screen.md").write_text("screen summary")
    result, count = mod.cluster("20240101")
    assert count == 2
    assert "Audio Transcript" in result
    assert "Screen Activity Summary" in result


def test_cluster_range(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")
    # Write JSONL format: metadata first, then entry
    (day_dir / "120000_audio.jsonl").write_text('{}\n{"text": "hi"}\n')
    (day_dir / "120000_monitor_1_diff.json").write_text("{}")
    (day_dir / "120000_screen.md").write_text("screen summary")
    md = mod.cluster_range("20240101", "120000", "120100", audio=True, screen="raw")
    assert "Monitor 1" in md
    assert "Audio Transcript" in md


def test_cluster_scan(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("think.cluster")
    # Audio transcripts at 09:01, 09:05, 09:20 and 11:00 (JSONL format with empty metadata)
    (day_dir / "090101_audio.jsonl").write_text("{}\n")
    (day_dir / "090500_audio.jsonl").write_text("{}\n")
    (day_dir / "092000_audio.jsonl").write_text("{}\n")
    (day_dir / "110000_audio.jsonl").write_text("{}\n")
    # Screen transcripts at 10:01, 10:05, 10:20 and 12:00
    (day_dir / "100101_screen.md").write_text("screen")
    (day_dir / "100500_screen.md").write_text("screen")
    (day_dir / "102000_screen.md").write_text("screen")
    (day_dir / "120000_screen.md").write_text("screen")
    audio_ranges, screen_ranges = mod.cluster_scan("20240101")
    assert audio_ranges == [("09:00", "09:30"), ("11:00", "11:15")]
    assert screen_ranges == [("10:00", "10:30"), ("12:00", "12:15")]
