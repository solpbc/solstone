import importlib

from think.utils import day_path


def test_get_raw_file(tmp_path, monkeypatch):
    utils = importlib.import_module("think.utils")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_dir = day_path("20240101")
    seen = day_dir / "seen"
    seen.mkdir()
    heard = day_dir / "heard"
    heard.mkdir()

    (seen / "123000_monitor_1_diff.png").write_bytes(b"data")
    (day_dir / "123000_monitor_1_diff.json").write_text(
        '{"visual_description": "screen"}'
    )

    (heard / "090000_audio.flac").write_bytes(b"data")
    (day_dir / "090000_audio.json").write_text('[{"text": "hello"}]')

    path, mime, meta = utils.get_raw_file("20240101", "123000_monitor_1_diff.json")
    assert path == "seen/123000_monitor_1_diff.png"
    assert mime == "image/png"
    assert meta["visual_description"] == "screen"

    path, mime, meta = utils.get_raw_file("20240101", "090000_audio.json")
    assert path == "heard/090000_raw.flac"
    assert mime == "audio/flac"
    assert isinstance(meta, list) and meta[0]["text"] == "hello"
