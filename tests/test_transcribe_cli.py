# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unit tests for sol transcribe CLI (M3, M8, M9)."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_main_accepts_journal_relative_path(tmp_path, monkeypatch):
    """main() resolves audio_path relative to JOURNAL_PATH when absolute path fails."""
    seg_dir = tmp_path / "20260201" / "default" / "090000_300"
    seg_dir.mkdir(parents=True)
    audio_file = seg_dir / "audio.wav"
    audio_file.touch()

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv", ["sol transcribe", "20260201/default/090000_300/audio.wav"]
    )

    mock_load = MagicMock(return_value=MagicMock())
    mock_vad_result = MagicMock()
    mock_vad_result.has_speech = False
    mock_vad_result.speech_duration = 0.0
    mock_vad_result.duration = 5.0
    mock_vad = MagicMock(return_value=mock_vad_result)

    with (
        patch("observe.transcribe.main.load_audio", mock_load),
        patch("observe.transcribe.main.run_vad", mock_vad),
        patch("observe.transcribe.main.callosum_send"),
        patch("observe.transcribe.main.get_segment_key", return_value="090000_300"),
        patch("observe.transcribe.main._build_base_event", return_value={}),
        patch("think.entities.load_recent_entity_names", return_value=[]),
    ):
        from observe.transcribe.main import main

        main()

    mock_load.assert_called_once()


def test_main_errors_on_nonexistent_absolute_path(tmp_path, monkeypatch, capsys):
    """main() errors clearly when path doesn't exist as absolute or journal-relative."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["sol transcribe", "/nonexistent/path/audio.wav"])

    from observe.transcribe.main import main

    with pytest.raises(SystemExit):
        main()

    captured = capsys.readouterr()
    assert "Tried absolute" in captured.err or "not found" in captured.err.lower()


def test_setup_cli_prints_message_on_default_journal(tmp_path, monkeypatch, capsys):
    """setup_cli() prints an informational message when JOURNAL_PATH uses the default."""
    monkeypatch.delenv("JOURNAL_PATH", raising=False)

    with (
        patch("think.utils.get_journal_info", return_value=(str(tmp_path), "default")),
        patch("think.utils.get_journal", return_value=str(tmp_path)),
        patch("think.utils.get_config", return_value={}),
    ):
        from think.utils import setup_cli

        parser = argparse.ArgumentParser()
        monkeypatch.setattr("sys.argv", ["test"])
        setup_cli(parser)

    captured = capsys.readouterr()
    assert "docs/INSTALL.md" in captured.err


def test_setup_cli_no_message_when_journal_path_set(tmp_path, monkeypatch, capsys):
    """setup_cli() prints no informational message when JOURNAL_PATH is explicitly set."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    with (
        patch("think.utils.get_journal_info", return_value=(str(tmp_path), "shell")),
        patch("think.utils.get_journal", return_value=str(tmp_path)),
        patch("think.utils.get_config", return_value={}),
    ):
        from think.utils import setup_cli

        parser = argparse.ArgumentParser()
        monkeypatch.setattr("sys.argv", ["test"])
        setup_cli(parser)

    captured = capsys.readouterr()
    assert "docs/INSTALL.md" not in captured.err


def _make_batch_journal(tmp_path: Path) -> Path:
    """Create a minimal temp journal with three segments for batch testing."""
    seg1 = tmp_path / "20260101" / "default" / "090000_300"
    seg1.mkdir(parents=True)
    (seg1 / "audio.flac").touch()

    seg2 = tmp_path / "20260101" / "default" / "140000_300"
    seg2.mkdir(parents=True)
    (seg2 / "audio.flac").touch()
    (seg2 / "audio.jsonl").touch()

    seg3 = tmp_path / "20260101" / "default" / "180000_300"
    seg3.mkdir(parents=True)
    (seg3 / "screen.png").touch()

    return tmp_path


def test_all_batch_processes_unprocessed_skips_transcribed(
    tmp_path, monkeypatch, capsys
):
    """--all processes unprocessed audio, skips already-transcribed, ignores non-audio."""
    journal = _make_batch_journal(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setattr("sys.argv", ["sol transcribe", "--all"])

    mock_process_one = MagicMock()

    with (
        patch("observe.transcribe.main._process_one", mock_process_one),
        patch("think.entities.load_recent_entity_names", return_value=[]),
    ):
        from observe.transcribe.main import main

        main()

    assert mock_process_one.call_count == 1
    called_path = mock_process_one.call_args[0][0]
    assert called_path.name == "audio.flac"
    assert "090000_300" in str(called_path)

    captured = capsys.readouterr()
    assert "1 processed" in captured.out
    assert "1 skipped" in captured.out


def test_all_redo_reprocesses_transcribed(tmp_path, monkeypatch):
    """--all --redo reprocesses even segments that already have .jsonl."""
    journal = _make_batch_journal(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setattr("sys.argv", ["sol transcribe", "--all", "--redo"])

    mock_process_one = MagicMock()

    with (
        patch("observe.transcribe.main._process_one", mock_process_one),
        patch("think.entities.load_recent_entity_names", return_value=[]),
    ):
        from observe.transcribe.main import main

        main()

    assert mock_process_one.call_count == 2


def test_all_and_audio_path_mutually_exclusive(tmp_path, monkeypatch):
    """Providing both --all and audio_path produces a clear error."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["sol transcribe", "--all", "some/audio.wav"])

    with patch("think.entities.load_recent_entity_names", return_value=[]):
        from observe.transcribe.main import main

        with pytest.raises(SystemExit):
            main()


def test_neither_all_nor_audio_path_errors(tmp_path, monkeypatch):
    """Providing neither --all nor audio_path produces a clear error."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["sol transcribe"])

    with patch("think.entities.load_recent_entity_names", return_value=[]):
        from observe.transcribe.main import main

        with pytest.raises(SystemExit):
            main()
