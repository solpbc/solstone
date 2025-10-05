import importlib

from think.utils import day_path


def test_scan_day(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")

    mod = importlib.import_module("hear.transcribe")
    heard_dir = day_dir / "heard"
    heard_dir.mkdir(parents=True)
    (day_dir / "120000_audio.flac").write_bytes(b"data")
    (heard_dir / "110000_audio.flac").write_bytes(b"data")

    info = mod.Transcriber.scan_day(day_dir)
    assert info["raw"] == ["heard/110000_audio.flac"]
    assert info["processed"] == []
    assert info["repairable"] == ["120000_audio.flac"]
