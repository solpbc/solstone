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

    (heard / "090000_raw.flac").write_bytes(b"data")
    # Write JSONL format: metadata first, then entry
    (day_dir / "090000_audio.jsonl").write_text('{}\n{"text": "hello"}\n')

    path, mime, meta = utils.get_raw_file("20240101", "123000_monitor_1_diff.json")
    assert path == "seen/123000_monitor_1_diff.png"
    assert mime == "image/png"
    assert meta["visual_description"] == "screen"

    path, mime, meta = utils.get_raw_file("20240101", "090000_audio.jsonl")
    assert path == "heard/090000_raw.flac"
    assert mime == "audio/flac"
    # JSONL format returns a list: [metadata_dict, entry1_dict, ...]
    assert isinstance(meta, list) and len(meta) == 2
    assert meta[1]["text"] == "hello"
