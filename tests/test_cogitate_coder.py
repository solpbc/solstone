# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for cogitate coder mode: write flag, coder agent."""

import asyncio
import importlib
from unittest.mock import AsyncMock, patch

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
        """Without write flag, --allowedTools restricts to sol."""
        provider = self._provider()
        mock_instance = AsyncMock()
        mock_instance.run = AsyncMock(return_value="result")
        mock_instance.cli_session_id = None
        mock_runner_cls.return_value = mock_instance

        config = {"prompt": "test", "model": "claude-sonnet-4-20250514"}
        asyncio.run(provider.run_cogitate(config))

        cmd = mock_runner_cls.call_args.kwargs["cmd"]
        assert "--allowedTools" in cmd
        assert "Bash(sol *)" in cmd

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
        assert cmd[s_idx + 1] == "workspace-write"

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
        assert cmd[s_idx + 1] == "workspace-write"
        assert "resume" in cmd


# ---------------------------------------------------------------------------
# Write flag — Google provider
# ---------------------------------------------------------------------------


class TestGoogleWriteFlag:
    """Verify --approval-mode is controlled by config write flag."""

    def _provider(self):
        return importlib.import_module("think.providers.google")

    @patch("think.providers.google.CLIRunner")
    def test_no_write_uses_yolo_with_policy(self, mock_runner_cls):
        """Without write flag, approval-mode is yolo with scoped policy."""
        provider = self._provider()
        mock_instance = AsyncMock()
        mock_instance.run = AsyncMock(return_value="result")
        mock_instance.cli_session_id = None
        mock_runner_cls.return_value = mock_instance

        config = {"prompt": "test", "model": "gemini-2.5-flash"}
        asyncio.run(provider.run_cogitate(config))

        cmd = mock_runner_cls.call_args.kwargs["cmd"]
        idx = cmd.index("--approval-mode")
        assert cmd[idx + 1] == "yolo"
        policy_idx = cmd.index("--policy")
        assert cmd[policy_idx + 1].endswith("policies/cogitate.toml")

    @patch("think.providers.google.CLIRunner")
    def test_write_true_uses_yolo_mode(self, mock_runner_cls):
        """With write=True, approval-mode is yolo (full access)."""
        provider = self._provider()
        mock_instance = AsyncMock()
        mock_instance.run = AsyncMock(return_value="result")
        mock_instance.cli_session_id = None
        mock_runner_cls.return_value = mock_instance

        config = {"prompt": "test", "model": "gemini-2.5-flash", "write": True}
        asyncio.run(provider.run_cogitate(config))

        cmd = mock_runner_cls.call_args.kwargs["cmd"]
        idx = cmd.index("--approval-mode")
        assert cmd[idx + 1] == "yolo"
        assert "--policy" not in cmd

    def test_cogitate_policy_file_exists_on_disk(self):
        """The policy path wired into argv must resolve to a real file."""
        from think.providers.google import _COGITATE_POLICY_PATH

        assert _COGITATE_POLICY_PATH.is_file(), (
            f"Expected policy file at {_COGITATE_POLICY_PATH}"
        )


# ---------------------------------------------------------------------------
# talent/coder.md existence and frontmatter
# ---------------------------------------------------------------------------


class TestCoderAgent:
    """Verify talent/coder.md exists with correct frontmatter."""

    def test_coder_md_exists(self):
        """talent/coder.md must exist in the repo."""
        from pathlib import Path

        coder_path = Path(__file__).parent.parent / "talent" / "coder.md"
        assert coder_path.exists(), "talent/coder.md not found"

    def test_coder_frontmatter(self):
        """coder.md must have write: true and type: cogitate."""
        from pathlib import Path

        import frontmatter

        coder_path = Path(__file__).parent.parent / "talent" / "coder.md"
        post = frontmatter.load(coder_path)

        assert post.metadata.get("type") == "cogitate"
        assert post.metadata.get("write") is True
        assert post.metadata.get("title") == "Coder"
        assert "description" in post.metadata

    def test_coder_references_coding_skill(self):
        """coder.md must reference the developer docs instead of inlining guidelines."""
        from pathlib import Path

        coder_path = Path(__file__).parent.parent / "talent" / "coder.md"
        content = coder_path.read_text(encoding="utf-8")

        # Should reference the developer guide/docs, not inline dev guidelines
        assert "AGENTS.md" in content
        assert "docs/project-structure.md" in content
        assert "single source of truth" in content

        docs_dir = Path(__file__).parent.parent / "docs"
        assert (docs_dir / "coding-standards.md").exists()
        assert (docs_dir / "project-structure.md").exists()
        assert (docs_dir / "testing.md").exists()
        assert (docs_dir / "environment.md").exists()
