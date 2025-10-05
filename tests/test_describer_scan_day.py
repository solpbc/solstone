import importlib

from think.utils import day_path


def test_scan_day(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("see.describe")
    seen_dir = day_dir / "seen"
    seen_dir.mkdir(parents=True)
    (day_dir / "120000_monitor_1_diff.png").write_bytes(b"data")
    (day_dir / "120000_monitor_1_diff_box.json").write_text("{}")
    (seen_dir / "110000_monitor_1_diff.png").write_bytes(b"data")
    (seen_dir / "110000_monitor_1_diff_box.json").write_text("{}")

    info = mod.Describer.scan_day(day_dir)
    assert info["raw"] == ["seen/110000_monitor_1_diff.png"]
    assert info["processed"] == []
    assert info["repairable"] == ["120000_monitor_1_diff.png"]
