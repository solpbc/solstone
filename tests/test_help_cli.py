# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.help_cli."""

import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from think.help_cli import main


@pytest.fixture(autouse=True)
def _set_journal_path(monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", "tests/fixtures/journal")


def test_help_no_question_shows_static_help(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sol help"])

    with patch("sol.print_help") as mock_print_help:
        main()

    mock_print_help.assert_called_once()


def test_help_parses_question(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sol help", "how", "do", "I", "search"])
    mock_result = subprocess.CompletedProcess(
        args=["sol", "agents"],
        returncode=0,
        stdout='{"event":"finish","result":"Use sol call journal search"}\n',
        stderr="",
    )

    with patch("think.help_cli.subprocess.run", return_value=mock_result) as mock_run:
        main()

    call_args = mock_run.call_args
    assert call_args[0][0] == ["sol", "agents"]
    assert call_args[1]["capture_output"] is True
    assert call_args[1]["text"] is True
    assert call_args[1]["timeout"] == 120

    sent = call_args[1]["input"]
    payload = json.loads(sent.strip())
    assert payload["prompt"] == "how do I search"


def test_help_ndjson_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sol help", "show", "todo", "commands"])
    mock_result = subprocess.CompletedProcess(
        args=["sol", "agents"],
        returncode=0,
        stdout='{"event":"finish","result":"ok"}\n',
        stderr="",
    )

    with patch("think.help_cli.subprocess.run", return_value=mock_result) as mock_run:
        main()

    sent = mock_run.call_args[1]["input"]
    payload = json.loads(sent.strip())
    assert payload == {"name": "help", "prompt": "show todo commands"}


def test_help_parses_finish_event(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sol help", "how", "to", "search"])
    stdout = "\n".join(
        [
            '{"event":"start","ts":1}',
            '{"event":"thinking","ts":2,"summary":"..."}',
            '{"event":"finish","ts":3,"result":"Use `sol call journal search`."}',
        ]
    )
    mock_result = subprocess.CompletedProcess(
        args=["sol", "agents"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )

    with patch("think.help_cli.subprocess.run", return_value=mock_result):
        main()

    captured = capsys.readouterr()
    assert "Use `sol call journal search`." in captured.out


def test_help_uses_last_finish_event(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sol help", "search"])
    stdout = "\n".join(
        [
            '{"event":"finish","ts":1,"result":"old result"}',
            '{"event":"finish","ts":2,"result":"new result"}',
        ]
    )
    mock_result = subprocess.CompletedProcess(
        args=["sol", "agents"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )

    with patch("think.help_cli.subprocess.run", return_value=mock_result):
        main()

    captured = capsys.readouterr()
    assert "new result" in captured.out
    assert "old result" not in captured.out


def test_help_handles_error_event(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sol help", "bad", "request"])
    mock_result = subprocess.CompletedProcess(
        args=["sol", "agents"],
        returncode=1,
        stdout='{"event":"error","error":"provider unavailable"}\n',
        stderr="provider failed",
    )

    with patch("think.help_cli.subprocess.run", return_value=mock_result):
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "provider unavailable" in captured.err


def test_help_handles_empty_finish_result(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sol help", "empty"])
    mock_result = subprocess.CompletedProcess(
        args=["sol", "agents"],
        returncode=0,
        stdout='{"event":"finish","result":""}\n',
        stderr="",
    )

    with patch("think.help_cli.subprocess.run", return_value=mock_result):
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "empty result" in captured.err.lower()


def test_help_handles_timeout(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sol help", "slow", "question"])

    with patch(
        "think.help_cli.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["sol", "agents"], timeout=120),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "timed out" in captured.err.lower()
