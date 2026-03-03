# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for `sol call chat history` and `sol call chat read` commands."""

from __future__ import annotations

import os
from datetime import datetime

from typer.testing import CliRunner

from think.call import call_app

runner = CliRunner()

CHAT_FIXTURE_PATH = os.path.join(os.getcwd(), "tests", "fixtures", "journal")

CHAT_TS = {
    "1764019444672": 1764019444672,
    "1764019602551": 1764019602551,
    "1764019955580": 1764019955580,
}


def _configure_journal(monkeypatch):
    """Configure CHAT test commands to use fixture journal paths."""
    import convey.state
    import think.utils

    monkeypatch.setenv("JOURNAL_PATH", CHAT_FIXTURE_PATH)
    think.utils._journal_path_cache = None
    convey.state.journal_root = CHAT_FIXTURE_PATH


class TestChatHistory:
    """Tests for chat history and read CLI commands."""

    def test_history_lists_chats(self, monkeypatch):
        """All fixture chats appear in output."""
        _configure_journal(monkeypatch)

        result = runner.invoke(call_app, ["chat", "history"])

        assert result.exit_code == 0
        assert "1764019444672" in result.output
        assert "1764019602551" in result.output
        assert "1764019955580" in result.output

    def test_history_filter_facet(self, monkeypatch):
        """--facet work returns only the work chat."""
        _configure_journal(monkeypatch)

        result = runner.invoke(call_app, ["chat", "history", "--facet", "work"])

        assert result.exit_code == 0
        assert "1764019444672" in result.output
        assert "1764019955580" not in result.output

    def test_history_filter_facet_personal(self, monkeypatch):
        """--facet personal returns only the personal chat."""
        _configure_journal(monkeypatch)

        result = runner.invoke(call_app, ["chat", "history", "--facet", "personal"])

        assert result.exit_code == 0
        assert "1764019955580" in result.output
        assert "1764019444672" not in result.output

    def test_history_filter_muse(self, monkeypatch):
        """--muse default returns all chats (all are default)."""
        _configure_journal(monkeypatch)

        result = runner.invoke(call_app, ["chat", "history", "--muse", "default"])

        assert result.exit_code == 0
        assert "1764019444672" in result.output
        assert "1764019602551" in result.output
        assert "1764019955580" in result.output

    def test_history_empty_filter(self, monkeypatch):
        """--facet nonexistent returns 'No chats found.'."""
        _configure_journal(monkeypatch)

        result = runner.invoke(call_app, ["chat", "history", "--facet", "nonexistent"])

        assert result.exit_code == 0
        assert "No chats found." in result.output

    def test_history_limit(self, monkeypatch):
        """--limit 1 returns only 1 chat."""
        _configure_journal(monkeypatch)

        result = runner.invoke(call_app, ["chat", "history", "--limit", "1"])

        assert result.exit_code == 0
        lines = [line for line in result.output.strip().splitlines() if line.strip()]
        assert len(lines) == 1

    def test_history_filter_day(self, monkeypatch):
        """--day filters by the day derived from ts."""
        _configure_journal(monkeypatch)
        chat_day = datetime.fromtimestamp(CHAT_TS["1764019444672"] / 1000).strftime(
            "%Y%m%d"
        )

        result = runner.invoke(call_app, ["chat", "history", "--day", chat_day])

        assert result.exit_code == 0
        assert "1764019444672" in result.output

    def test_read_displays_events(self, monkeypatch):
        """read shows events from agent log."""
        _configure_journal(monkeypatch)

        result = runner.invoke(call_app, ["chat", "read", "1764019444672"])

        assert result.exit_code == 0
        assert "[request]" in result.output
        assert "[finish]" in result.output
        assert "project updates" in result.output

    def test_read_summary(self, monkeypatch):
        """--summary shows only request and finish."""
        _configure_journal(monkeypatch)

        result = runner.invoke(call_app, ["chat", "read", "1764019444672", "--summary"])

        assert result.exit_code == 0
        assert "[request]" in result.output
        assert "[finish]" in result.output
        assert "[thinking]" not in result.output
        assert "[tool_start]" not in result.output

    def test_read_nonexistent(self, monkeypatch):
        """read with bad chat_id exits 1."""
        _configure_journal(monkeypatch)

        result = runner.invoke(call_app, ["chat", "read", "9999999999999"])

        assert result.exit_code == 1
        assert "Chat not found" in result.output

    def test_read_no_agent_logs(self, monkeypatch):
        """read chat with missing agent log shows no events gracefully."""
        _configure_journal(monkeypatch)

        result = runner.invoke(call_app, ["chat", "read", "1764019602551"])

        assert result.exit_code == 0
        assert "No events found." in result.output
