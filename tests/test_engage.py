# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the engage CLI command."""

import importlib
from unittest.mock import patch

from typer.testing import CliRunner

runner = CliRunner()


def _engage_app():
    mod = importlib.reload(importlib.import_module("think.engage"))
    return mod.engage_app


def _invoke_engage(*args, input_text=""):
    return runner.invoke(_engage_app(), [*args], input=input_text)


def _call_app():
    call_mod = importlib.reload(importlib.import_module("think.call"))
    return call_mod.call_app


class TestEngage:
    def test_fire_and_forget(self):
        with patch(
            "think.cortex_client.cortex_request", return_value="agent-123"
        ) as mock_cr:
            result = _invoke_engage("coder", input_text="fix the bug\n")

        assert result.exit_code == 0
        assert "agent-123" in result.output
        mock_cr.assert_called_once_with(
            prompt="fix the bug", name="coder", config=None
        )

    def test_empty_stdin(self):
        result = _invoke_engage("coder", input_text="")

        assert result.exit_code == 1
        assert (
            "no prompt" in result.output.lower()
            or "no prompt" in (result.stderr or "").lower()
        )

    def test_cortex_failure(self):
        with patch("think.cortex_client.cortex_request", return_value=None):
            result = _invoke_engage("coder", input_text="fix the bug\n")

        assert result.exit_code == 1

    def test_wait_success(self):
        with patch(
            "think.cortex_client.cortex_request", return_value="agent-123"
        ), patch(
            "think.cortex_client.wait_for_agents",
            return_value=({"agent-123": "finish"}, []),
        ), patch(
            "think.cortex_client.read_agent_events",
            return_value=[{"event": "finish", "result": "All fixed!"}],
        ):
            result = _invoke_engage("coder", "--wait", input_text="fix the bug\n")

        assert result.exit_code == 0
        assert "All fixed!" in result.output

    def test_wait_error(self):
        with patch(
            "think.cortex_client.cortex_request", return_value="agent-123"
        ), patch(
            "think.cortex_client.wait_for_agents",
            return_value=({"agent-123": "error"}, []),
        ):
            result = _invoke_engage("coder", "--wait", input_text="fix the bug\n")

        assert result.exit_code == 1

    def test_wait_timeout(self):
        with patch(
            "think.cortex_client.cortex_request", return_value="agent-123"
        ), patch(
            "think.cortex_client.wait_for_agents",
            return_value=({}, ["agent-123"]),
        ):
            result = _invoke_engage("coder", "--wait", input_text="fix the bug\n")

        assert result.exit_code == 1
        combined_output = result.output
        if result.stderr:
            combined_output += result.stderr
        assert "timed out" in combined_output.lower()

    def test_facet_and_day(self):
        with patch(
            "think.cortex_client.cortex_request", return_value="agent-123"
        ) as mock_cr:
            result = _invoke_engage(
                "coder",
                "--facet",
                "work",
                "--day",
                "20260404",
                input_text="do stuff\n",
            )

        assert result.exit_code == 0
        mock_cr.assert_called_once_with(
            prompt="do stuff",
            name="coder",
            config={"facet": "work", "day": "20260404"},
        )

    def test_facet_only(self):
        with patch(
            "think.cortex_client.cortex_request", return_value="agent-123"
        ) as mock_cr:
            result = _invoke_engage(
                "coder", "--facet", "work", input_text="do stuff\n"
            )

        assert result.exit_code == 0
        mock_cr.assert_called_once_with(
            prompt="do stuff", name="coder", config={"facet": "work"}
        )

    def test_day_only(self):
        with patch(
            "think.cortex_client.cortex_request", return_value="agent-123"
        ) as mock_cr:
            result = _invoke_engage(
                "coder", "--day", "20260404", input_text="do stuff\n"
            )

        assert result.exit_code == 0
        mock_cr.assert_called_once_with(
            prompt="do stuff", name="coder", config={"day": "20260404"}
        )


class TestHandoffDeprecated:
    def test_handoff_still_works(self):
        with patch("think.cortex_client.cortex_request", return_value="agent-123"):
            result = runner.invoke(_call_app(), ["handoff", "coder"], input="fix the bug\n")

        assert result.exit_code == 0
        assert "agent-123" in result.output

    def test_handoff_hidden(self):
        result = runner.invoke(_call_app(), ["--help"])

        assert "handoff" not in result.output
