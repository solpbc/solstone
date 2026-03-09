# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the awareness system."""

import json

import pytest


@pytest.fixture(autouse=True)
def _temp_journal(monkeypatch, tmp_path):
    """Isolate all tests to a temporary journal."""
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))


class TestCurrentState:
    def test_empty_state_returns_empty_dict(self):
        from think.awareness import get_current

        assert get_current() == {}

    def test_update_state_creates_section(self):
        from think.awareness import get_current, update_state

        update_state("onboarding", {"path": "a", "status": "observing"})

        state = get_current()
        assert state["onboarding"]["path"] == "a"
        assert state["onboarding"]["status"] == "observing"

    def test_update_state_merges_into_existing(self):
        from think.awareness import get_current, update_state

        update_state("onboarding", {"path": "a", "status": "observing"})
        update_state("onboarding", {"observation_count": 5})

        state = get_current()
        assert state["onboarding"]["path"] == "a"
        assert state["onboarding"]["observation_count"] == 5

    def test_update_state_multiple_sections(self):
        from think.awareness import get_current, update_state

        update_state("onboarding", {"status": "complete"})
        update_state("preferences", {"nudge_frequency": "low"})

        state = get_current()
        assert state["onboarding"]["status"] == "complete"
        assert state["preferences"]["nudge_frequency"] == "low"

    def test_current_json_written_atomically(self, tmp_path):
        from think.awareness import _awareness_dir, update_state

        update_state("test", {"key": "value"})

        path = _awareness_dir() / "current.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["test"]["key"] == "value"


class TestDailyLog:
    def test_append_log_creates_file(self, tmp_path):
        from think.awareness import _awareness_dir, _today, append_log

        entry = append_log("state", key="test.started", message="hello")

        log_path = _awareness_dir() / f"{_today()}.jsonl"
        assert log_path.exists()
        assert entry["kind"] == "state"
        assert entry["key"] == "test.started"
        assert entry["message"] == "hello"
        assert "ts" in entry

    def test_append_log_appends_multiple(self):
        from think.awareness import _today, append_log, read_log

        append_log("state", key="a")
        append_log("observation", message="saw something")
        append_log("nudge", message="hey")

        entries = read_log(_today())
        assert len(entries) == 3
        assert entries[0]["kind"] == "state"
        assert entries[1]["kind"] == "observation"
        assert entries[2]["kind"] == "nudge"

    def test_read_log_empty_returns_empty_list(self):
        from think.awareness import read_log

        assert read_log("20990101") == []

    def test_append_log_with_data(self):
        from think.awareness import _today, append_log, read_log

        append_log("observation", data={"meetings": 2, "entities": ["Alice"]})

        entries = read_log(_today())
        assert entries[0]["data"]["meetings"] == 2

    def test_append_log_with_extra_fields(self):
        from think.awareness import _today, append_log, read_log

        append_log("observation", segment="123456_300", detail="meeting detected")

        entries = read_log(_today())
        assert entries[0]["segment"] == "123456_300"
        assert entries[0]["detail"] == "meeting detected"


class TestOnboarding:
    def test_get_onboarding_empty(self):
        from think.awareness import get_onboarding

        assert get_onboarding() == {}

    def test_start_onboarding_path_a(self):
        from think.awareness import get_onboarding, start_onboarding

        state = start_onboarding("a")

        assert state["path"] == "a"
        assert state["status"] == "observing"
        assert state["observation_count"] == 0
        assert state["nudges_sent"] == 0
        assert "started" in state

        # Verify persisted
        assert get_onboarding()["status"] == "observing"

    def test_start_onboarding_path_b(self):
        from think.awareness import start_onboarding

        state = start_onboarding("b")
        assert state["path"] == "b"
        assert state["status"] == "interviewing"

    def test_skip_onboarding(self):
        from think.awareness import get_onboarding, skip_onboarding

        skip_onboarding()
        assert get_onboarding()["status"] == "skipped"

    def test_complete_onboarding(self):
        from think.awareness import complete_onboarding, start_onboarding

        start_onboarding("a")
        complete_onboarding()

        from think.awareness import get_onboarding

        state = get_onboarding()
        assert state["status"] == "complete"
        assert state["path"] == "a"  # Preserved from start

    def test_start_onboarding_writes_log(self):
        from think.awareness import _today, read_log, start_onboarding

        start_onboarding("a")

        entries = read_log(_today())
        assert len(entries) == 1
        assert entries[0]["kind"] == "state"
        assert entries[0]["key"] == "onboarding.started"
        assert entries[0]["data"]["path"] == "a"

    def test_skip_writes_log(self):
        from think.awareness import _today, read_log, skip_onboarding

        skip_onboarding()

        entries = read_log(_today())
        assert entries[0]["key"] == "onboarding.skipped"

    def test_complete_writes_log(self):
        from think.awareness import _today, complete_onboarding, read_log

        complete_onboarding()

        entries = read_log(_today())
        assert entries[0]["key"] == "onboarding.complete"


class TestAwarenessCLI:
    def test_status_empty(self):
        from typer.testing import CliRunner

        from apps.awareness.call import app

        result = CliRunner().invoke(app, ["status"])
        assert result.exit_code == 0
        assert "No awareness state" in result.output

    def test_status_with_data(self):
        from typer.testing import CliRunner

        from apps.awareness.call import app
        from think.awareness import update_state

        update_state("onboarding", {"status": "observing"})

        result = CliRunner().invoke(app, ["status"])
        assert result.exit_code == 0
        assert "observing" in result.output

    def test_status_section(self):
        from typer.testing import CliRunner

        from apps.awareness.call import app
        from think.awareness import update_state

        update_state("onboarding", {"status": "observing"})

        result = CliRunner().invoke(app, ["status", "onboarding"])
        assert result.exit_code == 0
        assert "observing" in result.output

    def test_onboarding_read_empty(self):
        from typer.testing import CliRunner

        from apps.awareness.call import app

        result = CliRunner().invoke(app, ["onboarding"])
        assert result.exit_code == 0
        assert "No onboarding state" in result.output

    def test_onboarding_set_path_a(self):
        from typer.testing import CliRunner

        from apps.awareness.call import app

        result = CliRunner().invoke(app, ["onboarding", "--path", "a"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["path"] == "a"
        assert data["status"] == "observing"

    def test_onboarding_set_path_b(self):
        from typer.testing import CliRunner

        from apps.awareness.call import app

        result = CliRunner().invoke(app, ["onboarding", "--path", "b"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["path"] == "b"
        assert data["status"] == "interviewing"

    def test_onboarding_skip(self):
        from typer.testing import CliRunner

        from apps.awareness.call import app

        result = CliRunner().invoke(app, ["onboarding", "--skip"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "skipped"

    def test_onboarding_complete(self):
        from typer.testing import CliRunner

        from apps.awareness.call import app

        result = CliRunner().invoke(app, ["onboarding", "--complete"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "complete"

    def test_onboarding_invalid_path(self):
        from typer.testing import CliRunner

        from apps.awareness.call import app

        result = CliRunner().invoke(app, ["onboarding", "--path", "c"])
        assert result.exit_code == 1

    def test_log_cmd(self):
        from typer.testing import CliRunner

        from apps.awareness.call import app

        result = CliRunner().invoke(
            app, ["log", "observation", "saw a meeting", "--key", "test"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["kind"] == "observation"
        assert data["message"] == "saw a meeting"
        assert data["key"] == "test"

    def test_log_cmd_with_data(self):
        from typer.testing import CliRunner

        from apps.awareness.call import app

        result = CliRunner().invoke(
            app,
            ["log", "observation", "--data", '{"meetings": 2}'],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["data"]["meetings"] == 2


class TestJournalState:
    def test_first_daily_ready_via_update_state(self):
        from think.awareness import get_current, update_state

        update_state(
            "journal",
            {"first_daily_ready": True, "first_daily_ready_at": "20260308T14:00:00"},
        )

        state = get_current()
        assert state["journal"]["first_daily_ready"] is True
        assert state["journal"]["first_daily_ready_at"] == "20260308T14:00:00"

    def test_first_daily_ready_preserves_onboarding(self):
        from think.awareness import get_current, update_state

        update_state("onboarding", {"status": "complete", "path": "b"})
        update_state(
            "journal",
            {"first_daily_ready": True, "first_daily_ready_at": "20260308T14:00:00"},
        )

        state = get_current()
        assert state["onboarding"]["status"] == "complete"
        assert state["onboarding"]["path"] == "b"
        assert state["journal"]["first_daily_ready"] is True
