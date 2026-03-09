# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for onboarding routing logic."""

import argparse
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask


@pytest.fixture(autouse=True)
def _temp_journal(monkeypatch, tmp_path):
    """Ensure journaling defaults remain isolated from developer data."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))


class _ImmediateEvent:
    """Event object that never blocks in waits."""

    def set(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> bool:
        return True


def _run_chat_cli_main(
    args: argparse.Namespace,
    facets: dict,
    onboarding: dict | None = None,
) -> "MagicMock":
    with (
        patch("think.chat_cli.setup_cli", return_value=args),
        patch("think.facets.get_enabled_facets", return_value=facets),
        patch("think.awareness.get_onboarding", return_value=onboarding or {}),
        patch("think.chat_cli.cortex_request", return_value="agent-1") as mock_request,
        patch(
            "think.chat_cli.read_agent_events",
            return_value=[{"event": "finish", "result": "ok"}],
        ),
        patch("think.chat_cli.threading.Event", return_value=_ImmediateEvent()),
        patch("think.chat_cli.CallosumConnection") as mock_connection,
    ):
        mock_conn = MagicMock()
        mock_connection.return_value = mock_conn

        import think.chat_cli as chat_cli

        chat_cli.main()

    return mock_request


def _run_triage(
    facets: dict,
    onboarding: dict | None = None,
) -> "MagicMock":
    """Run the triage endpoint with mocked state."""
    app = Flask(__name__)
    with (
        patch("think.facets.get_enabled_facets", return_value=facets),
        patch("think.awareness.get_onboarding", return_value=onboarding or {}),
        patch("convey.utils.spawn_agent", return_value="agent-1") as mock_spawn,
        patch("think.cortex_client.wait_for_agents", return_value=({}, [])),
        patch(
            "think.cortex_client.read_agent_events",
            return_value=[{"event": "finish", "result": "ok"}],
        ),
    ):
        from convey.triage import triage

        with app.test_request_context("/", method="POST", json={"message": "hello"}):
            response = triage()

    assert response.status_code == 200
    return mock_spawn


# --- Triage endpoint routing ---


def test_triage_new_user_gets_onboarding():
    """No facets, no awareness state → onboarding agent."""
    mock = _run_triage(facets={})
    assert mock.call_args.kwargs["name"] == "onboarding"


def test_triage_established_user_gets_triage():
    """Has facets → triage agent."""
    mock = _run_triage(facets={"work": {}})
    assert mock.call_args.kwargs["name"] == "triage"


def test_triage_path_a_observing_gets_triage():
    """Path A active → triage (not onboarding again)."""
    mock = _run_triage(facets={}, onboarding={"status": "observing"})
    assert mock.call_args.kwargs["name"] == "triage"


def test_triage_path_a_ready_gets_triage():
    """Path A recommendations ready → triage."""
    mock = _run_triage(facets={}, onboarding={"status": "ready"})
    assert mock.call_args.kwargs["name"] == "triage"


def test_triage_skipped_gets_triage():
    """Onboarding skipped, no facets → triage (not onboarding again)."""
    mock = _run_triage(facets={}, onboarding={"status": "skipped"})
    assert mock.call_args.kwargs["name"] == "triage"


def test_triage_complete_gets_triage():
    """Onboarding complete, no facets → triage."""
    mock = _run_triage(facets={}, onboarding={"status": "complete"})
    assert mock.call_args.kwargs["name"] == "triage"


# --- Chat CLI routing ---


def test_chat_cli_routes_to_onboarding_when_default_and_no_facets():
    args = argparse.Namespace(
        message=["Hi there"],
        muse="default",
        facet=None,
        provider=None,
        verbose=False,
    )
    mock_request = _run_chat_cli_main(args, facets={})
    assert mock_request.call_args.kwargs["name"] == "onboarding"


def test_chat_cli_keeps_explicit_muse_when_no_facets():
    args = argparse.Namespace(
        message=["Hi there"],
        muse="entities",
        facet=None,
        provider=None,
        verbose=False,
    )
    mock_request = _run_chat_cli_main(args, facets={})
    assert mock_request.call_args.kwargs["name"] == "entities"


def test_chat_cli_path_a_observing_stays_default():
    """During Path A observation, chat CLI uses default muse, not onboarding."""
    args = argparse.Namespace(
        message=["What have you noticed?"],
        muse="default",
        facet=None,
        provider=None,
        verbose=False,
    )
    mock_request = _run_chat_cli_main(
        args, facets={}, onboarding={"status": "observing"}
    )
    assert mock_request.call_args.kwargs["name"] == "default"


def test_chat_cli_skipped_stays_default():
    """After skipping onboarding, chat CLI uses default muse."""
    args = argparse.Namespace(
        message=["Hello"],
        muse="default",
        facet=None,
        provider=None,
        verbose=False,
    )
    mock_request = _run_chat_cli_main(args, facets={}, onboarding={"status": "skipped"})
    assert mock_request.call_args.kwargs["name"] == "default"


# --- Placeholder resolution ---


class TestPlaceholderResolution:
    def test_observing(self):
        from convey.apps import _resolve_placeholder

        result = _resolve_placeholder("observing", {}, 0)
        assert "learning how you work" in result

    def test_ready(self):
        from convey.apps import _resolve_placeholder

        result = _resolve_placeholder("ready", {}, 0)
        assert "suggestions" in result

    def test_interviewing(self):
        from convey.apps import _resolve_placeholder

        result = _resolve_placeholder("interviewing", {}, 0)
        assert "Tell me about" in result

    def test_complete_no_daily(self):
        from convey.apps import _resolve_placeholder

        result = _resolve_placeholder("complete", {}, 0)
        assert "Capture is running" in result

    def test_complete_first_daily_young(self):
        from convey.apps import _resolve_placeholder

        current = {"journal": {"first_daily_ready": True}}
        result = _resolve_placeholder("complete", current, 1)
        assert "first daily analysis is ready" in result

    def test_complete_first_daily_mid(self):
        from convey.apps import _resolve_placeholder

        current = {"journal": {"first_daily_ready": True}}
        result = _resolve_placeholder("complete", current, 3)
        assert "daily analysis is ready" in result
        assert "first" not in result

    def test_complete_first_daily_mature(self):
        from convey.apps import _resolve_placeholder

        current = {"journal": {"first_daily_ready": True}}
        result = _resolve_placeholder("complete", current, 10)
        assert "Ask me about your day" in result

    def test_skipped_no_daily(self):
        from convey.apps import _resolve_placeholder

        result = _resolve_placeholder("skipped", {}, 0)
        assert "Capture is running" in result

    def test_skipped_with_daily_mature(self):
        from convey.apps import _resolve_placeholder

        current = {"journal": {"first_daily_ready": True}}
        result = _resolve_placeholder("skipped", current, 10)
        assert "Ask me about your day" in result

    def test_unknown_status_fallback(self):
        from convey.apps import _resolve_placeholder

        result = _resolve_placeholder("", {}, 0)
        assert result == "Send a message..."

    def test_no_status_fallback(self):
        from convey.apps import _resolve_placeholder

        result = _resolve_placeholder("", {}, 5)
        assert result == "Send a message..."


# --- Triage daily output context ---


class TestTriageDailyContext:
    def test_triage_complete_injects_daily_context(self, tmp_path):
        """When agent outputs exist, the prompt includes daily analysis context."""
        from datetime import datetime

        today = datetime.now().strftime("%Y%m%d")
        agents_dir = tmp_path / today / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "flow.md").write_text("# Flow")
        (agents_dir / "meetings.md").write_text("# Meetings")

        mock = _run_triage(
            facets={"work": {}},
            onboarding={"status": "complete"},
        )
        prompt = mock.call_args.kwargs["prompt"]
        assert "Daily analysis available" in prompt
        assert "flow" in prompt
        assert "meetings" in prompt

    def test_triage_complete_no_outputs_no_extra_context(self):
        """When no agent outputs exist, no daily analysis context is added."""
        mock = _run_triage(
            facets={"work": {}},
            onboarding={"status": "complete"},
        )
        prompt = mock.call_args.kwargs["prompt"]
        assert "Daily analysis" not in prompt

    def test_triage_complete_falls_back_to_yesterday(self, tmp_path):
        """When today has no outputs but yesterday does, use yesterday's."""
        from datetime import datetime, timedelta

        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        agents_dir = tmp_path / yesterday / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "flow.md").write_text("# Flow")

        mock = _run_triage(
            facets={"work": {}},
            onboarding={"status": "complete"},
        )
        prompt = mock.call_args.kwargs["prompt"]
        assert "Daily analysis available" in prompt
        assert "flow" in prompt
        assert yesterday in prompt

    def test_triage_skipped_injects_daily_context(self, tmp_path):
        """Skipped onboarding also gets daily analysis context."""
        from datetime import datetime

        today = datetime.now().strftime("%Y%m%d")
        agents_dir = tmp_path / today / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "knowledge_graph.md").write_text("# KG")

        mock = _run_triage(
            facets={},
            onboarding={"status": "skipped"},
        )
        prompt = mock.call_args.kwargs["prompt"]
        assert "Daily analysis available" in prompt
        assert "knowledge_graph" in prompt
