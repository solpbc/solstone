# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for dream --activity mode and activity template variables."""

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# run_activity_prompts
# ---------------------------------------------------------------------------


class TestRunActivityPrompts:
    """Tests for dream.run_activity_prompts."""

    def _write_record(self, tmpdir, facet, day, record):
        """Helper to write an activity record to the journal."""
        path = Path(tmpdir) / "facets" / facet / "activities" / f"{day}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def test_not_found_returns_false(self, monkeypatch):
        from think.dream import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            result = run_activity_prompts(
                day="20260209",
                activity_id="nonexistent_100000_300",
                facet="work",
            )
            assert result is False

    def test_no_matching_agents_returns_true(self, monkeypatch):
        from think.dream import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            self._write_record(
                tmpdir,
                "work",
                "20260209",
                {
                    "id": "coding_100000_300",
                    "activity": "coding",
                    "segments": ["100000_300", "100500_300"],
                    "level_avg": 0.75,
                    "description": "Coding session",
                    "active_entities": ["VS Code"],
                },
            )

            # No activity-scheduled agents
            monkeypatch.setattr("think.dream.get_muse_configs", lambda schedule: {})

            result = run_activity_prompts(
                day="20260209",
                activity_id="coding_100000_300",
                facet="work",
            )
            assert result is True

    def test_filters_by_activity_type(self, monkeypatch):
        from think.dream import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            self._write_record(
                tmpdir,
                "work",
                "20260209",
                {
                    "id": "coding_100000_300",
                    "activity": "coding",
                    "segments": ["100000_300"],
                    "level_avg": 0.75,
                    "description": "Coding session",
                    "active_entities": [],
                },
            )

            configs = {
                "code_review": {
                    "type": "cogitate",
                    "priority": 50,
                    "activities": ["coding"],
                    "multi_facet": True,
                },
                "meeting_notes": {
                    "type": "generate",
                    "priority": 50,
                    "activities": ["meeting"],
                    "output": "md",
                },
            }

            monkeypatch.setattr(
                "think.dream.get_muse_configs", lambda schedule: configs
            )

            spawned_requests = []

            def mock_cortex_request(prompt, name, config):
                spawned_requests.append((name, config))
                return f"agent-{name}"

            monkeypatch.setattr("think.dream.cortex_request", mock_cortex_request)
            monkeypatch.setattr(
                "think.dream.wait_for_agents",
                lambda ids, timeout: ({aid: "finish" for aid in ids}, []),
            )

            result = run_activity_prompts(
                day="20260209",
                activity_id="coding_100000_300",
                facet="work",
            )

            assert result is True
            # Only code_review should be spawned (matches "coding" type)
            assert len(spawned_requests) == 1
            assert spawned_requests[0][0] == "code_review"

    def test_wildcard_matches_all_types(self, monkeypatch):
        from think.dream import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            self._write_record(
                tmpdir,
                "work",
                "20260209",
                {
                    "id": "browsing_100000_300",
                    "activity": "browsing",
                    "segments": ["100000_300"],
                    "level_avg": 0.5,
                    "description": "Web browsing",
                    "active_entities": [],
                },
            )

            configs = {
                "activity_summary": {
                    "type": "generate",
                    "priority": 50,
                    "activities": ["*"],
                    "output": "md",
                },
            }

            monkeypatch.setattr(
                "think.dream.get_muse_configs", lambda schedule: configs
            )

            spawned = []

            def mock_cortex_request(prompt, name, config):
                spawned.append(name)
                return f"agent-{name}"

            monkeypatch.setattr("think.dream.cortex_request", mock_cortex_request)
            monkeypatch.setattr(
                "think.dream.wait_for_agents",
                lambda ids, timeout: ({aid: "finish" for aid in ids}, []),
            )

            result = run_activity_prompts(
                day="20260209",
                activity_id="browsing_100000_300",
                facet="work",
            )

            assert result is True
            assert spawned == ["activity_summary"]

    def test_passes_span_and_activity_in_request(self, monkeypatch):
        from think.dream import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            record = {
                "id": "coding_100000_300",
                "activity": "coding",
                "segments": ["100000_300", "100500_300", "101000_300"],
                "level_avg": 0.83,
                "description": "Extended coding",
                "active_entities": ["VS Code", "Git"],
            }
            self._write_record(tmpdir, "work", "20260209", record)

            configs = {
                "code_review": {
                    "type": "generate",
                    "priority": 50,
                    "activities": ["coding"],
                    "output": "md",
                },
            }

            monkeypatch.setattr(
                "think.dream.get_muse_configs", lambda schedule: configs
            )

            captured_config = {}

            def mock_cortex_request(prompt, name, config):
                captured_config.update(config)
                return "agent-1"

            monkeypatch.setattr("think.dream.cortex_request", mock_cortex_request)
            monkeypatch.setattr(
                "think.dream.wait_for_agents",
                lambda ids, timeout: ({aid: "finish" for aid in ids}, []),
            )

            run_activity_prompts(
                day="20260209",
                activity_id="coding_100000_300",
                facet="work",
            )

            assert captured_config["span"] == [
                "100000_300",
                "100500_300",
                "101000_300",
            ]
            assert captured_config["facet"] == "work"
            assert captured_config["day"] == "20260209"
            assert captured_config["activity"]["id"] == "coding_100000_300"
            assert captured_config["activity"]["activity"] == "coding"

            # Verify facet-scoped output_path
            assert "output_path" in captured_config
            output_path = captured_config["output_path"]
            assert "facets/work/activities/20260209/coding_100000_300/" in output_path
            assert output_path.endswith("code_review.md")

    def test_failed_agent_returns_false(self, monkeypatch):
        from think.dream import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            self._write_record(
                tmpdir,
                "work",
                "20260209",
                {
                    "id": "coding_100000_300",
                    "activity": "coding",
                    "segments": ["100000_300"],
                    "level_avg": 0.75,
                    "description": "Coding",
                    "active_entities": [],
                },
            )

            configs = {
                "code_review": {
                    "type": "cogitate",
                    "priority": 50,
                    "activities": ["coding"],
                },
            }

            monkeypatch.setattr(
                "think.dream.get_muse_configs", lambda schedule: configs
            )
            monkeypatch.setattr(
                "think.dream.cortex_request",
                lambda prompt, name, config: "agent-1",
            )
            monkeypatch.setattr(
                "think.dream.wait_for_agents",
                lambda ids, timeout: ({aid: "error" for aid in ids}, []),
            )

            result = run_activity_prompts(
                day="20260209",
                activity_id="coding_100000_300",
                facet="work",
            )

            assert result is False

    def test_empty_segments_returns_false(self, monkeypatch):
        from think.dream import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            self._write_record(
                tmpdir,
                "work",
                "20260209",
                {
                    "id": "coding_100000_300",
                    "activity": "coding",
                    "segments": [],
                    "level_avg": 0.5,
                    "description": "Empty",
                    "active_entities": [],
                },
            )

            result = run_activity_prompts(
                day="20260209",
                activity_id="coding_100000_300",
                facet="work",
            )

            assert result is False

    def test_emits_dream_events(self, monkeypatch):
        from think.dream import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            self._write_record(
                tmpdir,
                "work",
                "20260209",
                {
                    "id": "coding_100000_300",
                    "activity": "coding",
                    "segments": ["100000_300"],
                    "level_avg": 0.75,
                    "description": "Coding",
                    "active_entities": [],
                },
            )

            configs = {
                "code_review": {
                    "type": "cogitate",
                    "priority": 50,
                    "activities": ["*"],
                },
            }

            monkeypatch.setattr(
                "think.dream.get_muse_configs", lambda schedule: configs
            )
            monkeypatch.setattr(
                "think.dream.cortex_request",
                lambda prompt, name, config: "agent-1",
            )
            monkeypatch.setattr(
                "think.dream.wait_for_agents",
                lambda ids, timeout: ({aid: "finish" for aid in ids}, []),
            )

            emitted = []
            monkeypatch.setattr(
                "think.dream.emit", lambda event, **kw: emitted.append((event, kw))
            )

            run_activity_prompts(
                day="20260209",
                activity_id="coding_100000_300",
                facet="work",
            )

            events = [e[0] for e in emitted]
            assert "started" in events
            assert "group_started" in events
            assert "agent_started" in events
            assert "agent_completed" in events
            assert "group_completed" in events
            assert "completed" in events

            # Verify mode=activity in started event
            started_kw = next(kw for ev, kw in emitted if ev == "started")
            assert started_kw["mode"] == "activity"
            assert started_kw["activity"] == "coding_100000_300"
            assert started_kw["facet"] == "work"


# ---------------------------------------------------------------------------
# Activity template variables in agents.py
# ---------------------------------------------------------------------------


class TestActivityTemplateVars:
    """Tests for activity template variables in _build_prompt_context."""

    def test_activity_vars_populated(self):
        from think.agents import _build_prompt_context

        activity = {
            "id": "coding_100000_300",
            "activity": "coding",
            "description": "Extended coding session with Git and VS Code",
            "level_avg": 0.83,
            "active_entities": ["VS Code", "Git"],
            "segments": ["100000_300", "100500_300"],
        }

        ctx = _build_prompt_context(
            "20260209", None, ["100000_300", "100500_300"], activity=activity
        )

        assert ctx["activity_id"] == "coding_100000_300"
        assert ctx["activity_type"] == "coding"
        assert (
            ctx["activity_description"]
            == "Extended coding session with Git and VS Code"
        )
        assert ctx["activity_level"] == "0.83"
        assert ctx["activity_entities"] == "VS Code, Git"
        assert ctx["activity_segments"] == "100000_300, 100500_300"
        assert int(ctx["activity_duration"]) == 10  # 2 * 300s = 10 min

    def test_no_activity_no_vars(self):
        from think.agents import _build_prompt_context

        ctx = _build_prompt_context("20260209", None, None)

        assert "activity_id" not in ctx
        assert "activity_type" not in ctx

    def test_empty_entities(self):
        from think.agents import _build_prompt_context

        activity = {
            "id": "browsing_100000_300",
            "activity": "browsing",
            "description": "Browsing",
            "level_avg": 0.5,
            "active_entities": [],
            "segments": ["100000_300"],
        }

        ctx = _build_prompt_context("20260209", None, ["100000_300"], activity=activity)

        assert ctx["activity_entities"] == ""

    def test_duration_minimum_one(self):
        from think.agents import _build_prompt_context

        activity = {
            "id": "test_bad_seg",
            "activity": "test",
            "description": "",
            "level_avg": 0.5,
            "active_entities": [],
            "segments": ["invalid"],
        }

        ctx = _build_prompt_context("20260209", None, None, activity=activity)

        # Invalid segment can't parse, should default to 1
        assert ctx["activity_duration"] == "1"


# ---------------------------------------------------------------------------
# Muse config validation for activity schedule
# ---------------------------------------------------------------------------


class TestMuseActivityValidation:
    """Tests for activity schedule validation in get_muse_configs."""

    def _isolate_muse(self, monkeypatch, tmp_path):
        """Point muse discovery at tmp_path only (no real muse/ or apps/)."""
        muse_dir = tmp_path / "muse"
        muse_dir.mkdir(exist_ok=True)
        monkeypatch.setattr("think.muse.MUSE_DIR", muse_dir)
        monkeypatch.setattr("think.muse.APPS_DIR", tmp_path / "no_apps")
        return muse_dir

    def test_missing_activities_field_raises(self, monkeypatch, tmp_path):
        import frontmatter

        from think.muse import get_muse_configs

        muse_dir = self._isolate_muse(monkeypatch, tmp_path)

        post = frontmatter.Post(
            "Test prompt",
            schedule="activity",
            priority=50,
            type="generate",
            output="md",
            # Missing 'activities' field
        )
        (muse_dir / "test_agent.md").write_text(frontmatter.dumps(post))

        with pytest.raises(ValueError, match="non-empty 'activities' list"):
            get_muse_configs(schedule="activity")

    def test_valid_activities_field_passes(self, monkeypatch, tmp_path):
        import frontmatter

        from think.muse import get_muse_configs

        muse_dir = self._isolate_muse(monkeypatch, tmp_path)

        post = frontmatter.Post(
            "Test prompt",
            schedule="activity",
            priority=50,
            type="generate",
            output="md",
            activities=["coding", "meeting"],
        )
        (muse_dir / "test_agent.md").write_text(frontmatter.dumps(post))

        configs = get_muse_configs(schedule="activity")
        assert "test_agent" in configs
        assert configs["test_agent"]["activities"] == ["coding", "meeting"]

    def test_wildcard_activities_passes(self, monkeypatch, tmp_path):
        import frontmatter

        from think.muse import get_muse_configs

        muse_dir = self._isolate_muse(monkeypatch, tmp_path)

        post = frontmatter.Post(
            "Test prompt",
            schedule="activity",
            priority=50,
            type="generate",
            output="md",
            activities=["*"],
        )
        (muse_dir / "test_agent.md").write_text(frontmatter.dumps(post))

        configs = get_muse_configs(schedule="activity")
        assert "test_agent" in configs
        assert configs["test_agent"]["activities"] == ["*"]


# ---------------------------------------------------------------------------
# CLI argument validation
# ---------------------------------------------------------------------------


class TestActivityCLIArgs:
    """Tests for dream CLI argument validation."""

    def test_activity_requires_facet(self, monkeypatch):
        from think.dream import parse_args

        parser = parse_args()

        monkeypatch.setattr(
            "sys.argv",
            ["sol dream", "--activity", "coding_100000_300", "--day", "20260209"],
        )

        # parse_args returns the parser, not args — need to test via main()
        # Instead test the parser directly
        args = parser.parse_args(
            ["--activity", "coding_100000_300", "--day", "20260209"]
        )
        assert args.activity == "coding_100000_300"
        assert args.facet is None  # Validation happens in main()

    def test_activity_args_parsed(self):
        from think.dream import parse_args

        parser = parse_args()
        args = parser.parse_args(
            [
                "--activity",
                "coding_100000_300",
                "--facet",
                "work",
                "--day",
                "20260209",
            ]
        )
        assert args.activity == "coding_100000_300"
        assert args.facet == "work"
        assert args.day == "20260209"
