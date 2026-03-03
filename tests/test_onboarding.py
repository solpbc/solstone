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
) -> "MagicMock":
    with (
        patch("think.chat_cli.setup_cli", return_value=args),
        patch("think.facets.get_enabled_facets", return_value=facets),
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


@pytest.mark.parametrize(
    "facets, expected_name",
    [({}, "onboarding"), ({"work": {}}, "triage")],
)
def test_triage_route_name_selection(facets, expected_name):
    from convey.triage import triage

    app = Flask(__name__)
    with (
        patch("think.facets.get_enabled_facets", return_value=facets),
        patch("convey.utils.spawn_agent", return_value="agent-1") as mock_spawn,
        patch("think.cortex_client.wait_for_agents", return_value=({}, [])),
        patch(
            "think.cortex_client.read_agent_events",
            return_value=[{"event": "finish", "result": "ok"}],
        ),
    ):
        with app.test_request_context("/", method="POST", json={"message": "hello"}):
            response = triage()

    assert response.status_code == 200
    assert response.get_json() == {"response": "ok"}
    assert mock_spawn.call_args.kwargs["name"] == expected_name


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
