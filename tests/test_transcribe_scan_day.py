import importlib


def test_scan_day(tmp_path):
    mod = importlib.import_module("hear.transcribe")
    day_dir = tmp_path / "20240101"
    heard_dir = day_dir / "heard"
    heard_dir.mkdir(parents=True)
    (day_dir / "120000_audio.flac").write_bytes(b"data")
    (heard_dir / "110000_audio.flac").write_bytes(b"data")

    info = mod.Transcriber.scan_day(day_dir)
    assert info["raw"] == ["120000_audio.flac"]
    assert info["processed"] == ["heard/110000_audio.flac"]
    assert info["repairable"] == ["120000_audio.flac"]
