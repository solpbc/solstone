import importlib


def test_scan_day(tmp_path):
    mod = importlib.import_module("see.describe")
    day_dir = tmp_path / "20240101"
    seen_dir = day_dir / "seen"
    seen_dir.mkdir(parents=True)
    (day_dir / "120000_monitor_1_diff.png").write_bytes(b"data")
    (day_dir / "120000_monitor_1_diff_box.json").write_text("{}")
    (seen_dir / "110000_monitor_1_diff.png").write_bytes(b"data")
    (seen_dir / "110000_monitor_1_diff_box.json").write_text("{}")

    info = mod.Describer.scan_day(day_dir)
    assert info["raw"] == ["120000_monitor_1_diff_box.json"]
    assert info["processed"] == ["seen/110000_monitor_1_diff_box.json"]
    assert info["repairable"] == ["120000_monitor_1_diff_box.json"]
