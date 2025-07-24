import importlib
import json
from pathlib import Path


def test_importer_text(tmp_path, monkeypatch):
    mod = importlib.import_module("think.importer")

    transcript = "hello\nworld"
    txt = tmp_path / "sample.txt"
    txt.write_text(transcript)

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(
        mod, "detect_created", lambda p: {"day": "20240101", "time": "120000"}
    )
    monkeypatch.setattr(mod, "detect_transcript_segment", lambda t: ["seg1", "seg2"])
    monkeypatch.setattr(mod, "detect_transcript_json", lambda t: [{"text": t}])

    monkeypatch.setattr("sys.argv", ["think-importer", str(txt)])
    mod.main()

    f1 = tmp_path / "20240101" / "120000_imported_audio.json"
    f2 = tmp_path / "20240101" / "120500_imported_audio.json"
    assert json.loads(f1.read_text()) == [{"text": "seg1"}]
    assert json.loads(f2.read_text()) == [{"text": "seg2"}]
