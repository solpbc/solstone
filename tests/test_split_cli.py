import importlib
import types
from pathlib import Path

import numpy as np


def prepare_soundfile(monkeypatch, frames):
    def read(path, dtype="float32"):
        return np.zeros((frames, 2), dtype=dtype), 16000

    def info(path):
        return types.SimpleNamespace(frames=frames, samplerate=16000)

    def write(path, data, samplerate, format=None):
        Path(path).write_bytes(b"fLaCfake")

    sf = types.SimpleNamespace(read=read, info=info, write=write)
    monkeypatch.setitem(importlib.import_module("sys").modules, "soundfile", sf)
    return sf


def test_split_cli(tmp_path, monkeypatch):
    frames = 16000 * 60
    _ = prepare_soundfile(monkeypatch, frames)

    day = tmp_path / "20240101"
    day.mkdir()
    src = day / "120000_audio.flac"
    src.write_bytes(b"data")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    mod = importlib.import_module("hear.split")
    monkeypatch.setattr(
        "sys.argv",
        ["hear-split", "Jan 1 2024 12:00pm to 12:01pm"],
    )
    mod.main()
    assert (day / "120000_mic_audio.flac").exists()
    assert (day / "120000_system_audio.flac").exists()
    assert src.exists()

    monkeypatch.setattr(
        "sys.argv",
        [
            "hear-split",
            "--day",
            "20240101",
            "--start",
            "120000",
            "--length",
            "1",
            "--cleanup",
        ],
    )
    mod.main()
    assert not src.exists()
