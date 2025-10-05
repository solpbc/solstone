import datetime as dt
import importlib
import json
from pathlib import Path
from unittest.mock import patch

from think.utils import day_path


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

    # Mock gemini_generate to prevent real API calls during summarization
    monkeypatch.setattr(mod, "gemini_generate", lambda **kwargs: "Mocked summary")

    monkeypatch.setattr(
        "sys.argv",
        ["think-importer", str(txt), "--timestamp", "20240101_120000"],
    )
    mod.main()

    day_dir = day_path("20240101")
    f1 = day_dir / "120000_imported_audio.json"
    f2 = day_dir / "120500_imported_audio.json"

    chunk1 = json.loads(f1.read_text())
    chunk2 = json.loads(f2.read_text())

    assert chunk1["entries"] == [{"text": "seg1"}]
    assert chunk1["imported"]["id"] == "20240101_120000"
    assert "domain" not in chunk1["imported"]

    assert chunk2["entries"] == [{"text": "seg2"}]
    assert chunk2["imported"]["id"] == "20240101_120000"
    assert "domain" not in chunk2["imported"]


def test_importer_audio_transcribe(tmp_path, monkeypatch):
    """Test the new audio_transcribe functionality with Rev AI."""
    mod = importlib.import_module("think.importer")

    # Create a test audio file
    audio_file = tmp_path / "test_audio.mp3"
    audio_file.write_bytes(b"fake audio content")

    # Mock Rev AI response
    mock_revai_response = {
        "monologues": [
            {
                "speaker": 0,
                "elements": [
                    {"type": "text", "value": "Hello", "ts": 0.0, "confidence": 0.95},
                    {"type": "text", "value": " ", "ts": 0.5},
                    {"type": "text", "value": "world", "ts": 0.6, "confidence": 0.98},
                    {"type": "punct", "value": "."},
                    {"type": "text", "value": " ", "ts": 1.0},
                    {"type": "text", "value": "This", "ts": 1.1, "confidence": 0.9},
                    {"type": "text", "value": " ", "ts": 1.5},
                    {"type": "text", "value": "is", "ts": 1.6, "confidence": 0.92},
                    {"type": "text", "value": " ", "ts": 1.8},
                    {"type": "text", "value": "a", "ts": 1.9, "confidence": 0.93},
                    {"type": "text", "value": " ", "ts": 2.0},
                    {"type": "text", "value": "test", "ts": 2.1, "confidence": 0.91},
                    {"type": "punct", "value": "."},
                ],
            },
            {
                "speaker": 1,
                "elements": [
                    {
                        "type": "text",
                        "value": "Second",
                        "ts": 310.0,
                        "confidence": 0.88,
                    },  # After 5 minutes
                    {"type": "text", "value": " ", "ts": 310.5},
                    {"type": "text", "value": "chunk", "ts": 310.6, "confidence": 0.89},
                    {"type": "punct", "value": "."},
                ],
            },
        ]
    }

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    # Mock gemini_generate to prevent real API calls during summarization
    monkeypatch.setattr(mod, "gemini_generate", lambda **kwargs: "Mocked summary")

    # Mock the transcribe_file function
    with patch("think.importer.transcribe_file") as mock_transcribe:
        mock_transcribe.return_value = mock_revai_response

        # Run with --hear option
        monkeypatch.setattr(
            "sys.argv",
            [
                "think-importer",
                str(audio_file),
                "--timestamp",
                "20240101_120000",
                "--hear",
                "true",
                "--see",
                "false",
                "--split",
                "false",
            ],
        )
        mod.main()

    # Check that the files were created correctly
    day_dir = day_path("20240101")
    f1 = day_dir / "120000_imported_audio.json"
    f2 = day_dir / "120500_imported_audio.json"

    assert f1.exists()
    assert f2.exists()

    # Check first chunk (0-5 minutes)
    chunk1 = json.loads(f1.read_text())
    assert chunk1["imported"]["id"] == "20240101_120000"
    assert len(chunk1["entries"]) == 2
    assert chunk1["entries"][0]["text"] == "Hello world."
    assert chunk1["entries"][0]["speaker"] == 1  # Rev uses 0-based, we use 1-based
    assert chunk1["entries"][0]["source"] == "import"
    assert chunk1["entries"][0]["start"] == "00:00:00"
    assert chunk1["entries"][1]["text"] == "This is a test."

    # Check second chunk (5-10 minutes)
    chunk2 = json.loads(f2.read_text())
    assert chunk2["imported"]["id"] == "20240101_120000"
    assert len(chunk2["entries"]) == 1
    assert chunk2["entries"][0]["text"] == "Second chunk."
    assert chunk2["entries"][0]["speaker"] == 2
    assert chunk2["entries"][0]["start"] == "00:05:10"


def test_audio_transcribe_sanitizes_entities(tmp_path, monkeypatch):
    mod = importlib.import_module("think.importer")

    audio_file = tmp_path / "test_audio.mp3"
    audio_file.write_bytes(b"fake audio content")

    # Use fixtures journal for domain entities lookup
    from pathlib import Path
    fixtures_journal = Path(__file__).parent.parent / "fixtures" / "journal"
    monkeypatch.setenv("JOURNAL_PATH", str(fixtures_journal))

    captured: list[list[str]] = []

    def fake_transcribe_file(media_path, entities=None):
        captured.append(entities or [])
        return {}

    monkeypatch.setattr(mod, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(mod, "convert_revai_to_sunstone", lambda _: [])

    mod.audio_transcribe(
        str(audio_file),
        str(tmp_path),
        dt.datetime(2024, 1, 1, 12, 0, 0),
        import_id="20240101_120000",
        domain="acme",
    )

    assert captured
    assert captured[0] == [
        "Test Person (TP)",
        "Test Initiative (TI)",
    ]


def test_audio_transcribe_includes_import_metadata(tmp_path, monkeypatch):
    mod = importlib.import_module("think.importer")

    audio_file = tmp_path / "test_audio.mp3"
    audio_file.write_bytes(b"fake audio content")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    monkeypatch.setattr(
        "think.domains.get_domains",
        lambda: {
            "uavionix": {
                "entities": {"Person": ["Ryan Reed (R2)"]},
            }
        },
    )

    monkeypatch.setattr(mod, "transcribe_file", lambda *_, **__: {"ok": True})
    monkeypatch.setattr(
        mod,
        "convert_revai_to_sunstone",
        lambda *_: [
            {
                "text": "Test entry",
                "start": "00:00:00",
                "speaker": 1,
                "source": "import",
            }
        ],
    )

    created_files, _ = mod.audio_transcribe(
        str(audio_file),
        str(tmp_path),
        dt.datetime(2024, 1, 1, 12, 0, 0),
        import_id="20240101_120000",
        domain="uavionix",
    )

    assert created_files

    payload = json.loads(Path(created_files[0]).read_text())
    assert payload["entries"][0]["text"] == "Test entry"
    assert payload["imported"]["id"] == "20240101_120000"
    assert payload["imported"]["domain"] == "uavionix"
