# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for apps/home/events.py — conversation exchange recording."""

from unittest.mock import patch

import pytest

from apps.events import EventContext, clear_handlers, stop_dispatcher
from apps.home.events import TRIAGE_AGENT_NAMES, record_triage_exchange


@pytest.fixture(autouse=True)
def clean_handlers():
    clear_handlers()
    yield
    clear_handlers()
    stop_dispatcher()


class TestRecordTriageExchange:
    """Tests for record_triage_exchange handler."""

    def _make_ctx(self, msg):
        return EventContext(msg=msg, app="home", tract="cortex", event="finish")

    def test_ignores_non_triage_agent(self):
        """Handler returns early for non-triage agent names."""
        ctx = self._make_ctx(
            {
                "tract": "cortex",
                "event": "finish",
                "name": "reviewer",
                "agent_id": "123",
                "result": "hello",
            }
        )
        with patch("apps.home.events.record_exchange") as mock_record:
            record_triage_exchange(ctx)
            mock_record.assert_not_called()

    def test_ignores_missing_agent_id(self):
        """Handler returns early if agent_id is missing."""
        ctx = self._make_ctx(
            {
                "tract": "cortex",
                "event": "finish",
                "name": "unified",
                "result": "hello",
            }
        )
        with patch("apps.home.events.record_exchange") as mock_record:
            record_triage_exchange(ctx)
            mock_record.assert_not_called()

    @pytest.mark.parametrize("agent_name", sorted(TRIAGE_AGENT_NAMES))
    def test_records_exchange_for_triage_agents(self, agent_name):
        """Handler calls record_exchange with correct fields for each triage agent name."""
        events = [
            {
                "event": "request",
                "ts": 1700000000000,
                "agent_id": "abc123",
                "facet": "work",
                "app": "home",
                "path": "/home",
                "user_message": "hello world",
            },
            {
                "event": "finish",
                "ts": 1700000001000,
                "agent_id": "abc123",
                "result": "hi there",
            },
        ]
        ctx = self._make_ctx(
            {
                "tract": "cortex",
                "event": "finish",
                "name": agent_name,
                "agent_id": "abc123",
                "result": "hi there",
            }
        )
        with patch("apps.home.events.read_agent_events", return_value=events):
            with patch("apps.home.events.record_exchange") as mock_record:
                record_triage_exchange(ctx)
                mock_record.assert_called_once_with(
                    facet="work",
                    app="home",
                    path="/home",
                    user_message="hello world",
                    agent_response="hi there",
                    talent=agent_name,
                    agent_id="abc123",
                )

    def test_handles_missing_request_event(self):
        """Handler uses empty strings for metadata if request event not found."""
        events = [
            {"event": "finish", "agent_id": "abc123", "result": "done"},
        ]
        ctx = self._make_ctx(
            {
                "tract": "cortex",
                "event": "finish",
                "name": "unified",
                "agent_id": "abc123",
                "result": "done",
            }
        )
        with patch("apps.home.events.read_agent_events", return_value=events):
            with patch("apps.home.events.record_exchange") as mock_record:
                record_triage_exchange(ctx)
                mock_record.assert_called_once_with(
                    facet="",
                    app="",
                    path="",
                    user_message="",
                    agent_response="done",
                    talent="unified",
                    agent_id="abc123",
                )

    def test_handles_read_error_gracefully(self):
        """Handler logs and swallows exceptions from read_agent_events."""
        ctx = self._make_ctx(
            {
                "tract": "cortex",
                "event": "finish",
                "name": "unified",
                "agent_id": "abc123",
                "result": "done",
            }
        )
        with patch(
            "apps.home.events.read_agent_events",
            side_effect=FileNotFoundError("not found"),
        ):
            with patch("apps.home.events.record_exchange") as mock_record:
                record_triage_exchange(ctx)  # should not raise
                mock_record.assert_not_called()
