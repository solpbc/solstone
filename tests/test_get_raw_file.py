import importlib
import os


def test_get_raw_file(tmp_path):
    utils = importlib.import_module("think.utils")
    os.environ["JOURNAL_PATH"] = str(tmp_path)
    day_dir = tmp_path / "20240101"
    day_dir.mkdir()
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
    assert path == "heard/090000_audio.flac"
    assert mime == "audio/flac"
    assert isinstance(meta, list) and meta[0]["text"] == "hello"
