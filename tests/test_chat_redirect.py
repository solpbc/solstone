# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for `sol call chat redirect` command behavior."""

from __future__ import annotations

import json
from unittest.mock import patch

from typer.testing import CliRunner

from think.call import call_app
from think.models import resolve_provider

runner = CliRunner()


def _configure_journal(tmp_path, monkeypatch):
    """Set up writable test journal paths."""
    import convey.state
    import think.utils

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    think.utils._journal_path_cache = None
    convey.state.journal_root = str(tmp_path)


class TestChatRedirect:
    """Tests for chat redirection from CLI."""

    def test_redirect_creates_chat_metadata(self, tmp_path, monkeypatch):
        """Metadata file is saved with expected schema for new chat threads."""
        _configure_journal(tmp_path, monkeypatch)
        agent_id = "1234567890123"
        expected_provider, _ = resolve_provider("muse.system.default", "cogitate")

        with (
            patch("think.cortex_client.cortex_request", return_value=agent_id),
            patch("apps.chat.call.callosum_send") as mock_navigate,
            patch("apps.chat.routes.generate_chat_title", return_value="Test Title"),
        ):
            with patch("think.callosum.callosum_send"):
                result = runner.invoke(
                    call_app,
                    [
                        "chat",
                        "redirect",
                        "Plan a sprint",
                        "--app",
                        "todos",
                        "--path",
                        "/app/todos",
                        "--facet",
                        "work",
                    ],
                )

        assert result.exit_code == 0
        assert f"Redirected to chat: {agent_id}" in result.output

        chat_file = tmp_path / "apps" / "chat" / "chats" / f"{agent_id}.json"
        assert chat_file.exists()
        data = json.loads(chat_file.read_text())
        assert data["facet"] == "work"
        assert data["provider"] == expected_provider
        assert data["title"] == "Test Title"
        assert data["agent_ids"] == [agent_id]
        assert isinstance(data["ts"], int)
        mock_navigate.assert_called_once_with(
            "navigate", "request", path=f"/app/chat#{agent_id}"
        )

    def test_redirect_spawns_default_agent(self, tmp_path, monkeypatch):
        """Command resolves provider and forwards full prompt + agent metadata."""
        _configure_journal(tmp_path, monkeypatch)
        agent_id = "1234567890123"
        expected_provider, _ = resolve_provider("muse.system.default", "cogitate")

        with (
            patch(
                "think.cortex_client.cortex_request", return_value=agent_id
            ) as mock_request,
            patch("apps.chat.call.callosum_send"),
            patch("apps.chat.routes.generate_chat_title", return_value="Test Title"),
            patch("think.callosum.callosum_send"),
        ):
            result = runner.invoke(
                call_app,
                [
                    "chat",
                    "redirect",
                    "Read recent logs",
                    "--app",
                    "calendar",
                    "--path",
                    "/app/calendar",
                    "--facet",
                    "work",
                ],
            )

        assert result.exit_code == 0
        mock_request.assert_called_once()
        kwargs = mock_request.call_args.kwargs
        assert kwargs["name"] == "default"
        assert kwargs["provider"] == expected_provider
        assert kwargs["config"] == {"facet": "work"}
        assert (
            kwargs["prompt"]
            == "Current app: calendar\nCurrent path: /app/calendar\nCurrent facet: work\n\nRead recent logs"
        )

    def test_redirect_navigates_to_chat(self, tmp_path, monkeypatch):
        """Successful redirect sends navigate message to the browser."""
        _configure_journal(tmp_path, monkeypatch)
        agent_id = "1234567890123"

        with (
            patch("think.cortex_client.cortex_request", return_value=agent_id),
            patch("apps.chat.call.callosum_send") as mock_navigate,
            patch("apps.chat.routes.generate_chat_title", return_value="Test Title"),
            patch("think.callosum.callosum_send"),
        ):
            result = runner.invoke(
                call_app,
                [
                    "chat",
                    "redirect",
                    "Open the sprint",
                    "--app",
                    "todos",
                    "--path",
                    "/app/todos",
                    "--facet",
                    "work",
                ],
            )

        assert result.exit_code == 0
        mock_navigate.assert_called_once_with(
            "navigate", "request", path=f"/app/chat#{agent_id}"
        )

    def test_redirect_includes_context_in_prompt(self, tmp_path, monkeypatch):
        """Context context should be prepended in the redirect prompt."""
        _configure_journal(tmp_path, monkeypatch)
        agent_id = "1234567890123"

        with (
            patch(
                "think.cortex_client.cortex_request", return_value=agent_id
            ) as mock_request,
            patch("apps.chat.call.callosum_send"),
            patch("apps.chat.routes.generate_chat_title", return_value="Test Title"),
            patch("think.callosum.callosum_send"),
        ):
            result = runner.invoke(
                call_app,
                [
                    "chat",
                    "redirect",
                    "Draft follow-up",
                    "--app",
                    "todos",
                    "--path",
                    "/app/todos",
                    "--facet",
                    "work",
                ],
            )

        assert result.exit_code == 0
        prompt = mock_request.call_args.kwargs["prompt"]
        assert prompt.startswith(
            "Current app: todos\nCurrent path: /app/todos\nCurrent facet: work\n\n"
        )

    def test_redirect_no_facet(self, tmp_path, monkeypatch):
        """Facet is optional and omitted from agent config and metadata when blank."""
        _configure_journal(tmp_path, monkeypatch)
        agent_id = "1234567890123"
        expected_provider, _ = resolve_provider("muse.system.default", "cogitate")

        with (
            patch(
                "think.cortex_client.cortex_request", return_value=agent_id
            ) as mock_request,
            patch("apps.chat.call.callosum_send"),
            patch("apps.chat.routes.generate_chat_title", return_value="Test Title"),
            patch("think.callosum.callosum_send"),
        ):
            result = runner.invoke(
                call_app,
                [
                    "chat",
                    "redirect",
                    "No facet message",
                    "--app",
                    "todos",
                    "--path",
                    "/app/todos",
                ],
            )

        assert result.exit_code == 0
        config = mock_request.call_args.kwargs["config"]
        assert "facet" not in config
        assert config == {}

        chat_file = tmp_path / "apps" / "chat" / "chats" / f"{agent_id}.json"
        data = json.loads(chat_file.read_text())
        assert data["facet"] is None
        assert data["provider"] == expected_provider
