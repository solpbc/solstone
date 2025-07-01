import importlib


def test_scan_day(tmp_path):
    stats_mod = importlib.import_module("think.journal_stats")
    day = tmp_path
    (day / "123456_audio.flac").write_bytes(b"RIFF")
    (day / "123456_audio.json").write_text("{}")
    (day / "123456_monitor_1_diff.png").write_bytes(b"PNG")
    (day / "123456_monitor_1_diff_box.json").write_text("{}")
    (day / "123456_monitor_1_diff.json").write_text("{}")
    (day / "123456_screen.md").write_text("hi")
    (day / "entities.md").write_text("")
    (day / "day.md").write_text("")
    js = stats_mod.JournalStats()
    js.scan_day("20240101", str(day))
    assert js.days["20240101"]["audio_flac"] == 1
    assert js.totals["diff_png"] == 1


def test_markdown(tmp_path):
    stats_mod = importlib.import_module("think.journal_stats")
    journal = tmp_path
    day = journal / "20240101"
    day.mkdir()
    (day / "123456_audio.flac").write_bytes(b"RIFF")
    (day / "123456_audio.json").write_text("{}")
    js = stats_mod.JournalStats()
    js.scan(str(journal))
    md = js.to_markdown()
    assert "Days scanned: 1" in md
    js.save_markdown(str(journal))
    assert (journal / "summary.md").exists()
