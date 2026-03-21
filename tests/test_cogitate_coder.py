# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for cogitate coder mode: write flag, handoff command, coder agent."""

import asyncio
import importlib
import io
import sys
from unittest.mock import AsyncMock, patch

import pytest
import typer
from typer.testing import CliRunner

from think.call import call_app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Write flag — Anthropic provider
# ---------------------------------------------------------------------------


class TestAnthropicWriteFlag:
    """Verify --allowedTools is controlled by config write flag."""

    def _provider(self):
        return importlib.import_module("think.providers.anthropic")

    @patch("think.providers.anthropic.check_cli_binary")
    @patch("think.providers.anthropic.CLIRunner")
    def test_no_write_restricts_tools(self, mock_runner_cls, mock_check):
        """Without write flag, --allowedTools restricts to sol call."""
        provider = self._provider()
        mock_instance = AsyncMock()
        mock_instance.run = AsyncMock(return_value="result")
        mock_instance.cli_session_id = None
        mock_runner_cls.return_value = mock_instance

        config = {"prompt": "test", "model": "claude-sonnet-4-20250514"}
        asyncio.run(provider.run_cogitate(config))

        cmd = mock_runner_cls.call_args.kwargs["cmd"]
        assert "--allowedTools" in cmd
        assert "Bash(sol call *)" in cmd

    @patch("think.providers.anthropic.check_cli_binary")
    @patch("think.providers.anthropic.CLIRunner")
    def test_write_true_grants_full_access(self, mock_runner_cls, mock_check):
        """With write=True, --allowedTools is omitted for full tool access."""
        provider = self._provider()
        mock_instance = AsyncMock()
        mock_instance.run = AsyncMock(return_value="result")
        mock_instance.cli_session_id = None
        mock_runner_cls.return_value = mock_instance

        config = {"prompt": "test", "model": "claude-sonnet-4-20250514", "write": True}
        asyncio.run(provider.run_cogitate(config))

        cmd = mock_runner_cls.call_args.kwargs["cmd"]
        assert "--allowedTools" not in cmd

    @patch("think.providers.anthropic.check_cli_binary")
    @patch("think.providers.anthropic.CLIRunner")
    def test_write_false_restricts_tools(self, mock_runner_cls, mock_check):
        """Explicit write=False keeps restriction."""
        provider = self._provider()
        mock_instance = AsyncMock()
        mock_instance.run = AsyncMock(return_value="result")
        mock_instance.cli_session_id = None
        mock_runner_cls.return_value = mock_instance

        config = {"prompt": "test", "model": "claude-sonnet-4-20250514", "write": False}
        asyncio.run(provider.run_cogitate(config))

        cmd = mock_runner_cls.call_args.kwargs["cmd"]
        assert "--allowedTools" in cmd


# ---------------------------------------------------------------------------
# Write flag — OpenAI provider
# ---------------------------------------------------------------------------


class TestOpenAIWriteFlag:
    """Verify sandbox mode is controlled by config write flag."""

    def _provider(self):
        return importlib.import_module("think.providers.openai")

    @patch("think.providers.openai.CLIRunner")
    def test_no_write_uses_readonly_sandbox(self, mock_runner_cls):
        """Without write flag, sandbox is read-only."""
        provider = self._provider()
        mock_instance = AsyncMock()
        mock_instance.run = AsyncMock(return_value="result")
        mock_instance.cli_session_id = None
        mock_runner_cls.return_value = mock_instance

        config = {"prompt": "test", "model": "gpt-5.2"}
        asyncio.run(provider.run_cogitate(config))

        cmd = mock_runner_cls.call_args.kwargs["cmd"]
        # Find the -s flag and its value
        s_idx = cmd.index("-s")
        assert cmd[s_idx + 1] == "read-only"

    @patch("think.providers.openai.CLIRunner")
    def test_write_true_uses_write_sandbox(self, mock_runner_cls):
        """With write=True, sandbox is write."""
        provider = self._provider()
        mock_instance = AsyncMock()
        mock_instance.run = AsyncMock(return_value="result")
        mock_instance.cli_session_id = None
        mock_runner_cls.return_value = mock_instance

        config = {"prompt": "test", "model": "gpt-5.2", "write": True}
        asyncio.run(provider.run_cogitate(config))

        cmd = mock_runner_cls.call_args.kwargs["cmd"]
        s_idx = cmd.index("-s")
        assert cmd[s_idx + 1] == "write"

    @patch("think.providers.openai.CLIRunner")
    def test_write_true_with_session_resume(self, mock_runner_cls):
        """Write flag works correctly with session resume path."""
        provider = self._provider()
        mock_instance = AsyncMock()
        mock_instance.run = AsyncMock(return_value="result")
        mock_instance.cli_session_id = None
        mock_runner_cls.return_value = mock_instance

        config = {
            "prompt": "test",
            "model": "gpt-5.2",
            "write": True,
            "session_id": "sess-123",
        }
        asyncio.run(provider.run_cogitate(config))

        cmd = mock_runner_cls.call_args.kwargs["cmd"]
        s_idx = cmd.index("-s")
        assert cmd[s_idx + 1] == "write"
        assert "resume" in cmd


# ---------------------------------------------------------------------------
# Write flag — Google provider
# ---------------------------------------------------------------------------


class TestGoogleWriteFlag:
    """Verify --allowed-tools is controlled by config write flag."""

    def _provider(self):
        return importlib.import_module("think.providers.google")

    @patch("think.providers.google.CLIRunner")
    def test_no_write_restricts_tools(self, mock_runner_cls):
        """Without write flag, --allowed-tools restricts to sol call."""
        provider = self._provider()
        mock_instance = AsyncMock()
        mock_instance.run = AsyncMock(return_value="result")
        mock_instance.cli_session_id = None
        mock_runner_cls.return_value = mock_instance

        config = {"prompt": "test", "model": "gemini-2.5-flash"}
        asyncio.run(provider.run_cogitate(config))

        cmd = mock_runner_cls.call_args.kwargs["cmd"]
        assert "--allowed-tools" in cmd
        assert "run_shell_command(sol call)" in cmd

    @patch("think.providers.google.CLIRunner")
    def test_write_true_grants_full_access(self, mock_runner_cls):
        """With write=True, --allowed-tools is omitted."""
        provider = self._provider()
        mock_instance = AsyncMock()
        mock_instance.run = AsyncMock(return_value="result")
        mock_instance.cli_session_id = None
        mock_runner_cls.return_value = mock_instance

        config = {"prompt": "test", "model": "gemini-2.5-flash", "write": True}
        asyncio.run(provider.run_cogitate(config))

        cmd = mock_runner_cls.call_args.kwargs["cmd"]
        assert "--allowed-tools" not in cmd


# ---------------------------------------------------------------------------
# sol call handoff command
# ---------------------------------------------------------------------------


class TestHandoffCommand:
    """Tests for sol call handoff subcommand."""

    @patch("think.cortex_client.cortex_request", return_value="1710864123456")
    def test_handoff_success(self, mock_cortex):
        """Handoff reads stdin, calls cortex_request, prints agent_id."""
        result = runner.invoke(call_app, ["handoff", "coder"], input="Fix the bug\n")

        assert result.exit_code == 0
        assert "1710864123456" in result.output
        mock_cortex.assert_called_once_with(prompt="Fix the bug", name="coder")

    def test_handoff_empty_stdin(self):
        """Empty stdin produces error and exit code 1."""
        result = runner.invoke(call_app, ["handoff", "coder"], input="")

        assert result.exit_code == 1
        assert "no prompt" in result.output.lower() or "no prompt" in (
            result.stderr or ""
        ).lower()

    def test_handoff_whitespace_only_stdin(self):
        """Whitespace-only stdin produces error."""
        result = runner.invoke(call_app, ["handoff", "coder"], input="   \n  \n")

        assert result.exit_code == 1

    @patch("think.cortex_client.cortex_request", return_value=None)
    def test_handoff_cortex_failure(self, mock_cortex):
        """When cortex_request returns None, handoff reports error."""
        result = runner.invoke(
            call_app, ["handoff", "coder"], input="Fix the bug\n"
        )

        assert result.exit_code == 1
        assert "failed" in result.output.lower() or "failed" in (
            result.stderr or ""
        ).lower()


# ---------------------------------------------------------------------------
# muse/coder.md existence and frontmatter
# ---------------------------------------------------------------------------


class TestCoderAgent:
    """Verify muse/coder.md exists with correct frontmatter."""

    def test_coder_md_exists(self):
        """muse/coder.md must exist in the repo."""
        from pathlib import Path

        coder_path = Path(__file__).parent.parent / "muse" / "coder.md"
        assert coder_path.exists(), "muse/coder.md not found"

    def test_coder_frontmatter(self):
        """coder.md must have write: true and type: cogitate."""
        import frontmatter
        from pathlib import Path

        coder_path = Path(__file__).parent.parent / "muse" / "coder.md"
        post = frontmatter.load(coder_path)

        assert post.metadata.get("type") == "cogitate"
        assert post.metadata.get("write") is True
        assert post.metadata.get("title") == "Coder"
        assert "description" in post.metadata

    def test_coder_has_developer_instructions(self):
        """coder.md body must contain development guidelines."""
        from pathlib import Path

        coder_path = Path(__file__).parent.parent / "muse" / "coder.md"
        content = coder_path.read_text(encoding="utf-8")

        # Should contain key sections from body skill content
        assert "Development Guidelines" in content
        assert "make test" in content
        assert "Coding Standards" in content
        assert "Project Structure" in content
