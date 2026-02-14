# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.help_cli."""

import io
import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from think.help_cli import main


@pytest.fixture(autouse=True)
def _set_journal_path(monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", "tests/fixtures/journal")


def _make_popen(stdout_lines, *, returncode=0):
    """Build a mock Popen whose stdout yields *stdout_lines*."""
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdout = io.StringIO("\n".join(stdout_lines) + "\n")
    proc.stderr = MagicMock()
    proc.stderr.read.return_value = ""
    proc.returncode = returncode
    proc.wait.return_value = returncode
    return proc


def test_help_no_question_shows_static_help(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sol help"])

    with patch("sol.print_help") as mock_print_help:
        main()

    mock_print_help.assert_called_once()


def test_help_parses_question(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sol help", "how", "do", "I", "search"])
    mock_proc = _make_popen(
        ['{"event":"finish","result":"Use sol call journal search"}'],
    )

    with patch("think.help_cli.subprocess.Popen", return_value=mock_proc) as mock_cls:
        main()

    call_args = mock_cls.call_args
    assert call_args[0][0] == ["sol", "agents"]
    assert call_args[1]["stdin"] == subprocess.PIPE
    assert call_args[1]["stdout"] == subprocess.PIPE
    assert call_args[1]["text"] is True

    written = mock_proc.stdin.write.call_args[0][0]
    payload = json.loads(written.strip())
    assert payload["prompt"] == "how do I search"
    mock_proc.stdin.close.assert_called_once()


def test_help_ndjson_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sol help", "show", "todo", "commands"])
    mock_proc = _make_popen(['{"event":"finish","result":"ok"}'])

    with patch("think.help_cli.subprocess.Popen", return_value=mock_proc):
        main()

    written = mock_proc.stdin.write.call_args[0][0]
    payload = json.loads(written.strip())
    assert payload == {"name": "help", "prompt": "show todo commands"}


def test_help_parses_finish_event(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sol help", "how", "to", "search"])
    mock_proc = _make_popen([
        '{"event":"start","ts":1}',
        '{"event":"thinking","ts":2,"summary":"..."}',
        '{"event":"finish","ts":3,"result":"Use `sol call journal search`."}',
    ])

    with patch("think.help_cli.subprocess.Popen", return_value=mock_proc):
        main()

    captured = capsys.readouterr()
    assert "Use `sol call journal search`." in captured.out


def test_help_uses_last_finish_event(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sol help", "search"])
    mock_proc = _make_popen([
        '{"event":"finish","ts":1,"result":"old result"}',
        '{"event":"finish","ts":2,"result":"new result"}',
    ])

    with patch("think.help_cli.subprocess.Popen", return_value=mock_proc):
        main()

    captured = capsys.readouterr()
    assert "new result" in captured.out
    assert "old result" not in captured.out


def test_help_handles_error_event(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sol help", "bad", "request"])
    mock_proc = _make_popen(
        ['{"event":"error","error":"provider unavailable"}'],
        returncode=1,
    )

    with patch("think.help_cli.subprocess.Popen", return_value=mock_proc):
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "provider unavailable" in captured.err


def test_help_handles_empty_finish_result(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sol help", "empty"])
    mock_proc = _make_popen(['{"event":"finish","result":""}'])

    with patch("think.help_cli.subprocess.Popen", return_value=mock_proc):
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "empty result" in captured.err.lower()


def test_help_handles_timeout(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sol help", "slow", "question"])
    mock_proc = _make_popen([])
    mock_proc.wait.side_effect = subprocess.TimeoutExpired(
        cmd=["sol", "agents"], timeout=120
    )

    with patch("think.help_cli.subprocess.Popen", return_value=mock_proc):
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 1
    mock_proc.kill.assert_called_once()
    captured = capsys.readouterr()
    assert "timed out" in captured.err.lower()
