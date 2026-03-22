# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the handoff CLI command."""

import importlib
from unittest.mock import patch

from typer.testing import CliRunner

runner = CliRunner()


def _call_app():
    call_mod = importlib.reload(importlib.import_module("think.call"))
    return call_mod.call_app


def _invoke_handoff(*args, input_text=""):
    return runner.invoke(_call_app(), ["handoff", *args], input=input_text)


def _assert_handoff_success():
    with patch(
        "think.cortex_client.cortex_request", return_value="agent-123"
    ) as mock_cr:
        result = _invoke_handoff("coder", input_text="fix the bug\n")
    assert result.exit_code == 0
    assert "agent-123" in result.output
    mock_cr.assert_called_once_with(prompt="fix the bug", name="coder")


def _assert_handoff_empty_stdin():
    result = _invoke_handoff("coder", input_text="")
    assert result.exit_code == 1
    assert (
        "no prompt" in result.output.lower()
        or "no prompt" in (result.stderr or "").lower()
    )


def _assert_handoff_cortex_failure():
    with patch("think.cortex_client.cortex_request", return_value=None):
        result = _invoke_handoff("coder", input_text="fix the bug\n")
    assert result.exit_code == 1


class TestHandoff:
    def test_success(self):
        _assert_handoff_success()

    def test_empty_stdin(self):
        _assert_handoff_empty_stdin()

    def test_cortex_failure(self):
        _assert_handoff_cortex_failure()
