# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the observation generator hooks and related features."""

import json

import pytest


@pytest.fixture(autouse=True)
def _temp_journal(monkeypatch, tmp_path):
    """Isolate all tests to a temporary journal."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))


class TestPreHook:
    def test_skips_when_not_observing(self):
        from talent.observation import pre_process

        result = pre_process({"day": "20260306", "segment": "120000_300"})
        assert result == {"skip_reason": "not_observing"}

    def test_skips_when_status_is_ready(self):
        from talent.observation import pre_process
        from think.awareness import update_state

        update_state("onboarding", {"status": "ready"})
        result = pre_process({"day": "20260306", "segment": "120000_300"})
        assert result == {"skip_reason": "not_observing"}

    def test_skips_when_status_is_complete(self):
        from talent.observation import pre_process
        from think.awareness import update_state

        update_state("onboarding", {"status": "complete"})
        result = pre_process({"day": "20260306", "segment": "120000_300"})
        assert result == {"skip_reason": "not_observing"}

    def test_skips_when_status_is_skipped(self):
        from talent.observation import pre_process
        from think.awareness import update_state

        update_state("onboarding", {"status": "skipped"})
        result = pre_process({"day": "20260306", "segment": "120000_300"})
        assert result == {"skip_reason": "not_observing"}

    def test_proceeds_when_observing(self):
        from talent.observation import pre_process
        from think.awareness import start_onboarding

        start_onboarding("a")
        result = pre_process({"day": "20260306", "segment": "120000_300"})
        assert result is None  # No modifications — proceed with LLM


class TestPostHook:
    @pytest.fixture(autouse=True)
    def _start_observation(self):
        from think.awareness import start_onboarding

        start_onboarding("a")

    def test_writes_observation_to_log(self):
        from talent.observation import post_process
        from think.awareness import read_log

        findings = json.dumps(
            {
                "has_meeting": False,
                "speaker_count": 1,
                "apps": ["VS Code"],
                "people": [],
                "companies": [],
                "projects": ["auth-service"],
                "tools": ["Git"],
                "topics": ["coding"],
                "summary": "Solo coding session",
            }
        )

        post_process(findings, {"day": "20260306", "segment": "120000_300"})

        entries = read_log("20260306")
        # Filter to observation entries (start_onboarding also writes a log entry)
        obs = [e for e in entries if e["kind"] == "observation"]
        assert len(obs) == 1
        assert obs[0]["data"]["apps"] == ["VS Code"]
        assert obs[0]["message"] == "Solo coding session"

    def test_increments_observation_count(self):
        from talent.observation import post_process
        from think.awareness import get_onboarding

        findings = json.dumps({"summary": "test", "has_meeting": False})

        post_process(findings, {"day": "20260306", "segment": "120000_300"})
        assert get_onboarding()["observation_count"] == 1

        post_process(findings, {"day": "20260306", "segment": "120500_300"})
        assert get_onboarding()["observation_count"] == 2

    def test_handles_invalid_json(self):
        from talent.observation import post_process

        result = post_process("not json", {"day": "20260306", "segment": "120000_300"})
        assert result == "not json"  # Returns result unchanged

    def test_handles_non_dict_json(self):
        from talent.observation import post_process

        result = post_process("[1,2,3]", {"day": "20260306", "segment": "120000_300"})
        assert result == "[1,2,3]"  # Returns result unchanged

    def test_returns_result_unchanged(self):
        from talent.observation import post_process

        findings = json.dumps({"summary": "test", "has_meeting": False})
        result = post_process(findings, {"day": "20260306", "segment": "120000_300"})
        assert result == findings


class TestNudgeLogic:
    def test_first_meeting_triggers_nudge(self):
        from talent.observation import _check_nudge

        findings = {
            "has_meeting": True,
            "speaker_count": 3,
            "meeting_topic": "sprint planning",
        }
        nudge = _check_nudge(findings, 1, 0, {})
        assert nudge is not None
        assert "Meeting detected" in nudge["title"]
        assert "3 people" in nudge["message"]

    def test_no_meeting_no_first_nudge(self):
        from talent.observation import _check_nudge

        findings = {"has_meeting": False}
        nudge = _check_nudge(findings, 1, 0, {})
        assert nudge is None

    def test_entity_cluster_triggers_nudge(self):
        from talent.observation import _check_nudge

        findings = {
            "has_meeting": False,
            "people": ["Alice", "Bob", "Charlie"],
        }
        nudge = _check_nudge(findings, 2, 1, {})
        assert nudge is not None
        assert "network" in nudge["title"].lower()

    def test_progress_update_at_5_segments(self):
        from talent.observation import _check_nudge

        findings = {"has_meeting": False, "people": []}
        nudge = _check_nudge(findings, 5, 2, {})
        assert nudge is not None
        assert "Still learning" in nudge["title"]

    def test_no_nudge_when_max_reached(self):
        from talent.observation import MAX_NUDGES, _check_nudge

        findings = {"has_meeting": True, "speaker_count": 5}
        # nudges_sent == MAX_NUDGES means all nudges used
        nudge = _check_nudge(findings, 1, MAX_NUDGES, {})
        # MAX_NUDGES is checked in post_process, not _check_nudge
        # But _check_nudge with nudges_sent=4 won't match any trigger
        assert nudge is None


class TestThreshold:
    def test_not_met_with_few_segments(self):
        from talent.observation import _threshold_met

        onboarding = {"started": "20260306T08:00:00"}
        assert _threshold_met(onboarding, 5) is False

    def test_not_met_with_short_time(self):
        # Just started — not enough time elapsed
        from datetime import datetime

        from talent.observation import MIN_SEGMENTS, _threshold_met

        now = datetime.now().strftime("%Y%m%dT%H:%M:%S")
        onboarding = {"started": now}
        assert _threshold_met(onboarding, MIN_SEGMENTS) is False

    def test_met_with_enough_segments_and_time(self):
        from talent.observation import MIN_SEGMENTS, _threshold_met

        # Started 5 hours ago
        onboarding = {"started": "20260101T03:00:00"}
        assert _threshold_met(onboarding, MIN_SEGMENTS) is True

    def test_not_met_with_no_started(self):
        from talent.observation import MIN_SEGMENTS, _threshold_met

        onboarding = {}
        assert _threshold_met(onboarding, MIN_SEGMENTS) is False


class TestElapsedHours:
    def test_valid_iso(self):
        from talent.observation import _elapsed_hours

        # A date far in the past should give many hours
        hours = _elapsed_hours("20200101T00:00:00")
        assert hours > 24

    def test_empty_string(self):
        from talent.observation import _elapsed_hours

        assert _elapsed_hours("") == 0.0

    def test_invalid_format(self):
        from talent.observation import _elapsed_hours

        assert _elapsed_hours("not-a-date") == 0.0


class TestAwarenessLogReadCLI:
    def test_log_read_empty(self):
        from typer.testing import CliRunner

        from apps.awareness.call import app

        result = CliRunner().invoke(app, ["log-read"])
        assert result.exit_code == 0
        assert "No entries found" in result.output

    def test_log_read_with_entries(self):
        from typer.testing import CliRunner

        from apps.awareness.call import app
        from think.awareness import append_log

        append_log("observation", message="test finding")
        append_log("state", key="test.key")

        result = CliRunner().invoke(app, ["log-read"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2

    def test_log_read_filter_by_kind(self):
        from typer.testing import CliRunner

        from apps.awareness.call import app
        from think.awareness import append_log

        append_log("observation", message="finding 1")
        append_log("state", key="transition")
        append_log("observation", message="finding 2")

        result = CliRunner().invoke(app, ["log-read", "--kind", "observation"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2
        assert all(e["kind"] == "observation" for e in data)

    def test_log_read_with_limit(self):
        from typer.testing import CliRunner

        from apps.awareness.call import app
        from think.awareness import append_log

        for i in range(5):
            append_log("observation", message=f"finding {i}")

        result = CliRunner().invoke(app, ["log-read", "--limit", "2"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2
        # Should return the LAST 2 entries
        assert data[0]["message"] == "finding 3"
        assert data[1]["message"] == "finding 4"


class TestChatBarPlaceholder:
    def _get_placeholder(self):
        """Extract chat bar placeholder from the context processor."""

        from flask import Flask

        app = Flask(__name__)
        app.config["TESTING"] = True

        from apps import AppRegistry
        from convey.apps import register_app_context

        registry = AppRegistry()
        register_app_context(app, registry)

        with app.test_request_context("/"):
            # Get context from context processors
            ctx = {}
            for func in app.template_context_processors[None]:
                ctx.update(func())
            return ctx.get("chat_bar_placeholder", "")

    def test_default_placeholder(self):
        assert "Bring in past conversations" in self._get_placeholder()

    def test_observing_placeholder(self):
        from think.awareness import start_onboarding

        start_onboarding("a")
        placeholder = self._get_placeholder()
        assert "Bring in past conversations" in placeholder

    def test_ready_placeholder(self):
        from think.awareness import start_onboarding, update_state

        start_onboarding("a")
        update_state("onboarding", {"status": "ready"})
        placeholder = self._get_placeholder()
        assert "Bring in past conversations" in placeholder

    def test_interviewing_placeholder(self):
        from think.awareness import start_onboarding

        start_onboarding("b")
        placeholder = self._get_placeholder()
        assert "Bring in past conversations" in placeholder

    def test_complete_placeholder(self):
        from think.awareness import start_onboarding, update_state

        start_onboarding("a")
        update_state("onboarding", {"status": "complete"})
        update_state("imports", {"has_imported": True})
        placeholder = self._get_placeholder()
        assert "Capture is running" in placeholder

    def test_skipped_placeholder(self):
        from think.awareness import skip_onboarding, update_state

        skip_onboarding()
        update_state("imports", {"has_imported": True})
        placeholder = self._get_placeholder()
        assert "Capture is running" in placeholder
