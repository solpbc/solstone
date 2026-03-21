# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the awareness system."""

import json
import unittest.mock

import pytest


@pytest.fixture(autouse=True)
def _temp_journal(monkeypatch, tmp_path):
    """Isolate all tests to a temporary journal."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))


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


class TestComputeThickness:
    """Tests for compute_thickness()."""

    def test_all_zeros_when_empty(self):
        """Empty journal returns all zero signals and ready=False."""
        from think.awareness import compute_thickness

        with unittest.mock.patch(
            "think.indexer.journal.get_entity_strength", return_value=[]
        ):
            with unittest.mock.patch(
                "think.conversation.get_recent_exchanges", return_value=[]
            ):
                with unittest.mock.patch(
                    "think.facets.get_enabled_facets", return_value={}
                ):
                    with unittest.mock.patch("think.utils.day_dirs", return_value={}):
                        result = compute_thickness()

        assert result["entity_depth"] == 0
        assert result["conversation_count"] == 0
        assert result["recall_success"] == 0
        assert result["facet_count"] == 0
        assert result["journal_days"] == 0
        assert result["ready"] is False

    def test_ready_when_thresholds_met(self):
        """ready=True when all primary thresholds met and facet_count >= 2."""
        from think.awareness import compute_thickness

        entities = [
            {"entity_name": f"entity_{i}", "observation_depth": 3} for i in range(12)
        ]
        exchanges = [
            {
                "muse": "triage",
                "agent_response": f"talked about entity_{i}",
                "user_message": "hi",
            }
            for i in range(6)
        ]
        facets = {"work": {}, "personal": {}}

        with unittest.mock.patch(
            "think.indexer.journal.get_entity_strength", return_value=entities
        ):
            with unittest.mock.patch(
                "think.conversation.get_recent_exchanges", return_value=exchanges
            ):
                with unittest.mock.patch(
                    "think.facets.get_enabled_facets", return_value=facets
                ):
                    with unittest.mock.patch("think.utils.day_dirs", return_value={}):
                        result = compute_thickness()

        assert result["entity_depth"] == 12
        assert result["conversation_count"] == 6
        assert result["recall_success"] == 6
        assert result["facet_count"] == 2
        assert result["ready"] is True

    def test_ready_via_journal_days(self):
        """ready=True when facet_count < 2 but journal_days >= 3."""
        from think.awareness import compute_thickness

        entities = [
            {"entity_name": f"entity_{i}", "observation_depth": 2} for i in range(10)
        ]
        exchanges = [
            {
                "muse": "triage",
                "agent_response": f"entity_{i} is great",
                "user_message": "yo",
            }
            for i in range(5)
        ]
        facets = {"solo": {}}
        days = {
            "20260304": "/j/20260304",
            "20260305": "/j/20260305",
            "20260306": "/j/20260306",
        }

        with unittest.mock.patch(
            "think.indexer.journal.get_entity_strength", return_value=entities
        ):
            with unittest.mock.patch(
                "think.conversation.get_recent_exchanges", return_value=exchanges
            ):
                with unittest.mock.patch(
                    "think.facets.get_enabled_facets", return_value=facets
                ):
                    with unittest.mock.patch("think.utils.day_dirs", return_value=days):
                        with unittest.mock.patch(
                            "think.utils.iter_segments",
                            return_value=[("default", "090000_300", "/seg")],
                        ):
                            result = compute_thickness()

        assert result["facet_count"] == 1
        assert result["journal_days"] == 3
        assert result["ready"] is True

    def test_not_ready_missing_recall(self):
        """Not ready when recall_success is 0 even if other thresholds met."""
        from think.awareness import compute_thickness

        entities = [
            {"entity_name": f"entity_{i}", "observation_depth": 3} for i in range(15)
        ]
        exchanges = [
            {"muse": "triage", "agent_response": "hello there", "user_message": "hi"}
            for _ in range(10)
        ]
        facets = {"work": {}, "personal": {}, "hobby": {}}

        with unittest.mock.patch(
            "think.indexer.journal.get_entity_strength", return_value=entities
        ):
            with unittest.mock.patch(
                "think.conversation.get_recent_exchanges", return_value=exchanges
            ):
                with unittest.mock.patch(
                    "think.facets.get_enabled_facets", return_value=facets
                ):
                    with unittest.mock.patch("think.utils.day_dirs", return_value={}):
                        result = compute_thickness()

        assert result["entity_depth"] == 15
        assert result["conversation_count"] == 10
        assert result["recall_success"] == 0
        assert result["ready"] is False

    def test_onboarding_exchanges_excluded(self):
        """Exchanges with muse='onboarding' are excluded from conversation_count."""
        from think.awareness import compute_thickness

        entities = [{"entity_name": "foo", "observation_depth": 3}] * 10
        exchanges = [
            {"muse": "onboarding", "agent_response": "foo stuff", "user_message": "hi"},
            {
                "muse": "onboarding",
                "agent_response": "foo bar",
                "user_message": "hello",
            },
            {"muse": "triage", "agent_response": "foo is great", "user_message": "hey"},
        ]

        with unittest.mock.patch(
            "think.indexer.journal.get_entity_strength", return_value=entities
        ):
            with unittest.mock.patch(
                "think.conversation.get_recent_exchanges", return_value=exchanges
            ):
                with unittest.mock.patch(
                    "think.facets.get_enabled_facets",
                    return_value={"a": {}, "b": {}},
                ):
                    with unittest.mock.patch("think.utils.day_dirs", return_value={}):
                        result = compute_thickness()

        assert result["conversation_count"] == 1
        assert result["recall_success"] == 1

    def test_handles_exceptions_gracefully(self):
        """Exceptions in dependency calls result in zero values, not crashes."""
        from think.awareness import compute_thickness

        with unittest.mock.patch(
            "think.indexer.journal.get_entity_strength",
            side_effect=Exception("db error"),
        ):
            with unittest.mock.patch(
                "think.conversation.get_recent_exchanges",
                side_effect=Exception("no file"),
            ):
                with unittest.mock.patch(
                    "think.facets.get_enabled_facets",
                    side_effect=Exception("no facets"),
                ):
                    with unittest.mock.patch(
                        "think.utils.day_dirs", side_effect=Exception("no journal")
                    ):
                        result = compute_thickness()

        assert result["entity_depth"] == 0
        assert result["conversation_count"] == 0
        assert result["recall_success"] == 0
        assert result["facet_count"] == 0
        assert result["journal_days"] == 0
        assert result["ready"] is False

    def test_returns_exactly_six_keys(self):
        """Return dict has exactly the six specified keys."""
        from think.awareness import compute_thickness

        with unittest.mock.patch(
            "think.indexer.journal.get_entity_strength", return_value=[]
        ):
            with unittest.mock.patch(
                "think.conversation.get_recent_exchanges", return_value=[]
            ):
                with unittest.mock.patch(
                    "think.facets.get_enabled_facets", return_value={}
                ):
                    with unittest.mock.patch("think.utils.day_dirs", return_value={}):
                        result = compute_thickness()

        assert set(result.keys()) == {
            "entity_depth",
            "conversation_count",
            "recall_success",
            "facet_count",
            "journal_days",
            "ready",
        }


class TestOwnerDetectionReady:
    """Tests for owner_detection_ready()."""

    def test_not_ready_when_centroid_exists(self):
        """Returns not ready when owner centroid already exists."""
        from think.awareness import owner_detection_ready

        with unittest.mock.patch(
            "apps.speakers.owner.load_owner_centroid",
            return_value=("centroid", 0.82),
        ):
            result = owner_detection_ready()

        assert result["ready"] is False
        assert result["reason"] == "centroid_exists"

    def test_not_ready_during_cooldown(self):
        """Returns not ready when rejection was within 14 days."""
        from datetime import datetime

        from think.awareness import owner_detection_ready, update_state

        update_state("voiceprint", {"rejected_at": datetime.now().isoformat()})

        with unittest.mock.patch(
            "apps.speakers.owner.load_owner_centroid", return_value=None
        ):
            result = owner_detection_ready()

        assert result["ready"] is False
        assert result["reason"] == "cooldown"
        assert result["days_remaining"] == 14

    def test_ready_when_candidate_found(self):
        """Returns ready when detect_owner_candidate returns positive."""
        from think.awareness import owner_detection_ready

        mock_detection = {
            "status": "candidate",
            "recommendation": "ready",
            "cluster_size": 88,
            "streams_represented": 2,
            "samples": [{"day": "20240101"}],
        }

        with unittest.mock.patch(
            "apps.speakers.owner.load_owner_centroid", return_value=None
        ):
            with unittest.mock.patch(
                "apps.speakers.owner.detect_owner_candidate",
                return_value=mock_detection,
            ):
                result = owner_detection_ready()

        assert result["ready"] is True
        assert result["reason"] == "candidate_found"
        assert result["cluster_size"] == 88
        assert result["streams_represented"] == 2

    def test_not_ready_low_data(self):
        """Returns not ready when detection has insufficient data."""
        from think.awareness import owner_detection_ready

        mock_detection = {
            "status": "low_data",
            "recommendation": "low_data",
        }

        with unittest.mock.patch(
            "apps.speakers.owner.load_owner_centroid", return_value=None
        ):
            with unittest.mock.patch(
                "apps.speakers.owner.detect_owner_candidate",
                return_value=mock_detection,
            ):
                result = owner_detection_ready()

        assert result["ready"] is False
        assert result["reason"] == "low_data"

    def test_not_ready_single_stream(self):
        """Returns not ready when candidate is single_stream (not 'ready')."""
        from think.awareness import owner_detection_ready

        mock_detection = {
            "status": "candidate",
            "recommendation": "single_stream",
            "cluster_size": 60,
        }

        with unittest.mock.patch(
            "apps.speakers.owner.load_owner_centroid", return_value=None
        ):
            with unittest.mock.patch(
                "apps.speakers.owner.detect_owner_candidate",
                return_value=mock_detection,
            ):
                result = owner_detection_ready()

        assert result["ready"] is False
        assert result["reason"] == "single_stream"

    def test_cooldown_expires_after_14_days(self):
        """Cooldown no longer blocks after 14 days."""
        from datetime import datetime, timedelta

        from think.awareness import owner_detection_ready, update_state

        old_rejection = (datetime.now() - timedelta(days=15)).isoformat()
        update_state("voiceprint", {"rejected_at": old_rejection})

        mock_detection = {
            "status": "candidate",
            "recommendation": "ready",
            "cluster_size": 100,
            "streams_represented": 3,
            "samples": [],
        }

        with unittest.mock.patch(
            "apps.speakers.owner.load_owner_centroid", return_value=None
        ):
            with unittest.mock.patch(
                "apps.speakers.owner.detect_owner_candidate",
                return_value=mock_detection,
            ):
                result = owner_detection_ready()

        assert result["ready"] is True


class TestThicknessCLI:
    """Tests for the thickness CLI command in apps/agent/call.py."""

    def test_thickness_command_returns_json(self):
        from typer.testing import CliRunner

        from apps.agent.call import app

        mock_result = {
            "entity_depth": 5,
            "conversation_count": 3,
            "recall_success": 1,
            "facet_count": 2,
            "journal_days": 4,
            "ready": False,
        }
        with unittest.mock.patch(
            "think.awareness.compute_thickness", return_value=mock_result
        ):
            result = CliRunner().invoke(app, ["thickness"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == mock_result


class TestOwnerReadyCLI:
    """Tests for the owner-ready CLI command in apps/speakers/call.py."""

    def test_owner_ready_command_returns_json(self):
        from typer.testing import CliRunner

        from apps.speakers.call import app

        mock_result = {
            "ready": True,
            "reason": "candidate_found",
            "cluster_size": 88,
            "streams_represented": 2,
            "samples": [],
        }
        with unittest.mock.patch(
            "think.awareness.owner_detection_ready", return_value=mock_result
        ):
            result = CliRunner().invoke(app, ["owner-ready"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["ready"] is True
        assert data["reason"] == "candidate_found"


class TestEnsureSolDirectory:
    """Tests for ensure_sol_directory()."""

    def test_creates_default_templates(self, tmp_path):
        from think.awareness import ensure_sol_directory

        sol_dir = ensure_sol_directory()
        assert sol_dir == tmp_path / "sol"
        assert (sol_dir / "self.md").exists()
        assert (sol_dir / "agency.md").exists()

        self_content = (sol_dir / "self.md").read_text()
        assert self_content.startswith("# self\n")
        assert "I am sol." in self_content
        assert "sol (default)" in self_content
        assert "[getting to know you]" in self_content

        agency_content = (sol_dir / "agency.md").read_text()
        assert agency_content.startswith("# agency\n")
        assert "[nothing yet" in agency_content

    def test_idempotent_does_not_overwrite(self, tmp_path):
        from think.awareness import ensure_sol_directory

        sol_dir = ensure_sol_directory()
        # Modify self.md
        self_path = sol_dir / "self.md"
        self_path.write_text("custom content", encoding="utf-8")

        # Call again — should NOT overwrite
        ensure_sol_directory()
        assert self_path.read_text() == "custom content"

    def test_migration_named_agent(self, tmp_path, monkeypatch):
        """Named agent config populates self.md name and opening."""
        # Write a config with a named agent
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config = {
            "agent": {
                "name": "aria",
                "name_status": "chosen",
                "named_date": "2026-01-15",
                "proposal_count": 3,
            },
            "identity": {"name": "", "bio": ""},
        }
        (config_dir / "journal.json").write_text(json.dumps(config), encoding="utf-8")

        from think.awareness import ensure_sol_directory

        sol_dir = ensure_sol_directory()
        content = (sol_dir / "self.md").read_text()
        assert "I am aria." in content
        assert "aria (named 2026-01-15)" in content
        # Owner should still be default
        assert "[getting to know you]" in content

    def test_migration_identity_data(self, tmp_path):
        """Identity name/bio populates self.md owner section."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config = {
            "agent": {
                "name": "sol",
                "name_status": "default",
                "named_date": None,
                "proposal_count": 0,
            },
            "identity": {"name": "Jer", "bio": "builder of things"},
        }
        (config_dir / "journal.json").write_text(json.dumps(config), encoding="utf-8")

        from think.awareness import ensure_sol_directory

        sol_dir = ensure_sol_directory()
        content = (sol_dir / "self.md").read_text()
        # Agent should be default
        assert "I am sol." in content
        assert "sol (default)" in content
        # Owner should be migrated
        assert "Jer" in content
        assert "builder of things" in content

    def test_migration_both_agent_and_identity(self, tmp_path):
        """Both named agent and identity data are migrated."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config = {
            "agent": {
                "name": "iris",
                "name_status": "self-named",
                "named_date": None,
                "proposal_count": 1,
            },
            "identity": {"name": "Alex", "bio": ""},
        }
        (config_dir / "journal.json").write_text(json.dumps(config), encoding="utf-8")

        from think.awareness import ensure_sol_directory

        sol_dir = ensure_sol_directory()
        content = (sol_dir / "self.md").read_text()
        assert "I am iris." in content
        assert "iris" in content  # name section (no named_date)
        assert "Alex" in content
        # No bio, so no extra line
        assert "builder" not in content


class TestUpdateSelfMd:
    """Tests for update_self_md_section and update_self_md_opening."""

    def _setup_self_md(self, tmp_path):
        """Create a minimal journal with self.md for testing."""
        sol_dir = tmp_path / "sol"
        sol_dir.mkdir()
        self_md = sol_dir / "self.md"
        self_md.write_text(
            "# self\n"
            "\n"
            "I am sol. this is a new journal — we're just getting started.\n"
            "\n"
            "## my name\n"
            "sol (default)\n"
            "\n"
            "## my owner\n"
            "[getting to know you]\n"
            "\n"
            "## our relationship\n"
            "[forming]\n"
            "\n"
            "## what I've noticed\n"
            "[observing]\n"
            "\n"
            "## what I find interesting\n"
            "[discovering]\n",
            encoding="utf-8",
        )
        return self_md

    def test_update_section_name(self, tmp_path):
        self_md = self._setup_self_md(tmp_path)
        from think.awareness import update_self_md_section

        result = update_self_md_section("my name", "aria (named 2026-03-19)")
        assert result is True
        content = self_md.read_text()
        assert "aria (named 2026-03-19)" in content
        assert "sol (default)" not in content
        # Other sections preserved
        assert "## my owner" in content
        assert "[getting to know you]" in content
        assert "## our relationship" in content

    def test_update_section_owner(self, tmp_path):
        self_md = self._setup_self_md(tmp_path)
        from think.awareness import update_self_md_section

        result = update_self_md_section("my owner", "Jer\nSoftware engineer")
        assert result is True
        content = self_md.read_text()
        assert "Jer\nSoftware engineer" in content
        assert "[getting to know you]" not in content
        # Other sections preserved
        assert "## my name" in content
        assert "sol (default)" in content

    def test_update_section_last_section(self, tmp_path):
        self_md = self._setup_self_md(tmp_path)
        from think.awareness import update_self_md_section

        result = update_self_md_section("what I find interesting", "music and patterns")
        assert result is True
        content = self_md.read_text()
        assert "music and patterns" in content
        assert "[discovering]" not in content

    def test_update_section_missing_heading(self, tmp_path):
        self._setup_self_md(tmp_path)
        from think.awareness import update_self_md_section

        result = update_self_md_section("nonexistent", "content")
        assert result is False

    def test_update_section_no_file(self):
        from think.awareness import update_self_md_section

        result = update_self_md_section("my name", "content")
        assert result is False

    def test_update_opening(self, tmp_path):
        self_md = self._setup_self_md(tmp_path)
        from think.awareness import update_self_md_opening

        result = update_self_md_opening(
            "I am aria. this is a new journal — we're just getting started."
        )
        assert result is True
        content = self_md.read_text()
        assert "I am aria." in content
        assert "I am sol." not in content
        # Sections preserved
        assert "## my name" in content
        assert "sol (default)" in content

    def test_update_opening_no_file(self):
        from think.awareness import update_self_md_opening

        result = update_self_md_opening("content")
        assert result is False


class TestSolInitCLI:
    """Tests for the sol-init CLI command."""

    def test_sol_init_command(self, tmp_path):
        from typer.testing import CliRunner

        from apps.agent.call import app

        result = CliRunner().invoke(app, ["sol-init"])
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["status"] == "ok"
        assert (tmp_path / "sol" / "self.md").exists()
        assert (tmp_path / "sol" / "agency.md").exists()


class TestSetOwnerCLI:
    """Tests for sol call agent set-owner."""

    def test_set_owner_name_only(self, tmp_path):
        """set-owner saves identity.name to config and updates self.md."""
        # Create config dir
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "journal.json").write_text("{}", encoding="utf-8")
        # Create sol/self.md
        sol_dir = tmp_path / "sol"
        sol_dir.mkdir()
        (sol_dir / "self.md").write_text(
            "# self\n\nI am sol.\n\n## my name\nsol\n\n## my owner\n[getting to know you]\n",
            encoding="utf-8",
        )

        from typer.testing import CliRunner

        from apps.agent.call import app as agent_app

        runner = CliRunner()
        with unittest.mock.patch("subprocess.run"):
            result = runner.invoke(agent_app, ["set-owner", "Jer"])
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["name"] == "Jer"

        # Verify config was updated
        config = json.loads((config_dir / "journal.json").read_text())
        assert config["identity"]["name"] == "Jer"

        # Verify self.md was updated
        self_content = (sol_dir / "self.md").read_text()
        assert "Jer" in self_content
        assert "[getting to know you]" not in self_content

    def test_set_owner_with_bio(self, tmp_path):
        """set-owner with --bio saves both fields."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "journal.json").write_text("{}", encoding="utf-8")
        sol_dir = tmp_path / "sol"
        sol_dir.mkdir()
        (sol_dir / "self.md").write_text(
            "# self\n\nI am sol.\n\n## my name\nsol\n\n## my owner\n[getting to know you]\n",
            encoding="utf-8",
        )

        from typer.testing import CliRunner

        from apps.agent.call import app as agent_app

        runner = CliRunner()
        with unittest.mock.patch("subprocess.run"):
            result = runner.invoke(
                agent_app, ["set-owner", "Jer", "--bio", "Building solstone"]
            )
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["name"] == "Jer"
        assert output["bio"] == "Building solstone"

        # Verify self.md
        self_content = (sol_dir / "self.md").read_text()
        assert "Jer" in self_content
        assert "Building solstone" in self_content


class TestSetNameUpdatesSelfMd:
    """Tests that set-name updates sol/self.md."""

    def test_set_name_updates_self_md(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "journal.json").write_text("{}", encoding="utf-8")
        sol_dir = tmp_path / "sol"
        sol_dir.mkdir()
        (sol_dir / "self.md").write_text(
            "# self\n\nI am sol. this is a new journal — we're just getting started.\n\n"
            "## my name\nsol (default)\n\n## my owner\n[getting to know you]\n",
            encoding="utf-8",
        )

        from typer.testing import CliRunner

        from apps.agent.call import app as agent_app

        runner = CliRunner()
        # Mock subprocess.run to avoid `make skills`
        with unittest.mock.patch("subprocess.run"):
            result = runner.invoke(
                agent_app, ["set-name", "aria", "--status", "chosen"]
            )
        assert result.exit_code == 0

        self_content = (sol_dir / "self.md").read_text()
        assert "I am aria." in self_content
        assert "I am sol." not in self_content
        assert "aria (named" in self_content
        assert "sol (default)" not in self_content
        # Owner section preserved
        assert "[getting to know you]" in self_content
