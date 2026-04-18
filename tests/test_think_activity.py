# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think --activity mode and activity template variables."""

import json
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# run_activity_prompts
# ---------------------------------------------------------------------------


class TestRunActivityPrompts:
    """Tests for think.run_activity_prompts."""

    def _write_record(self, tmpdir, facet, day, record):
        """Helper to write an activity record to the journal."""
        path = Path(tmpdir) / "facets" / facet / "activities" / f"{day}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def test_not_found_returns_false(self, monkeypatch):
        from think.thinking import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

            result = run_activity_prompts(
                day="20260209",
                activity_id="nonexistent_100000_300",
                facet="work",
            )
            assert result is False

    def test_no_matching_agents_returns_true(self, monkeypatch):
        from think.thinking import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

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
            monkeypatch.setattr(
                "think.thinking.get_talent_configs", lambda schedule: {}
            )

            result = run_activity_prompts(
                day="20260209",
                activity_id="coding_100000_300",
                facet="work",
            )
            assert result is True

    def test_filters_by_activity_type(self, monkeypatch):
        from think.thinking import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

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
                "think.thinking.get_talent_configs", lambda schedule: configs
            )

            spawned_requests = []

            def mock_cortex_request(prompt, name, config):
                spawned_requests.append((name, config))
                return f"agent-{name}"

            monkeypatch.setattr("think.thinking.cortex_request", mock_cortex_request)
            monkeypatch.setattr(
                "think.thinking.wait_for_uses",
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
        from think.thinking import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

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
                "think.thinking.get_talent_configs", lambda schedule: configs
            )

            spawned = []

            def mock_cortex_request(prompt, name, config):
                spawned.append(name)
                return f"agent-{name}"

            monkeypatch.setattr("think.thinking.cortex_request", mock_cortex_request)
            monkeypatch.setattr(
                "think.thinking.wait_for_uses",
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
        from think.thinking import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

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
                "think.thinking.get_talent_configs", lambda schedule: configs
            )

            captured_config = {}

            def mock_cortex_request(prompt, name, config):
                captured_config.update(config)
                return "agent-1"

            monkeypatch.setattr("think.thinking.cortex_request", mock_cortex_request)
            monkeypatch.setattr(
                "think.thinking.wait_for_uses",
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
        from think.thinking import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

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
                "think.thinking.get_talent_configs", lambda schedule: configs
            )
            monkeypatch.setattr(
                "think.thinking.cortex_request",
                lambda prompt, name, config: "agent-1",
            )
            monkeypatch.setattr(
                "think.thinking.wait_for_uses",
                lambda ids, timeout: ({aid: "error" for aid in ids}, []),
            )

            result = run_activity_prompts(
                day="20260209",
                activity_id="coding_100000_300",
                facet="work",
            )

            assert result is False

    def test_empty_segments_returns_true_for_synthetic_record(self, monkeypatch):
        from think.thinking import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

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

            assert result is True

    def test_cogitate_source_returns_true_without_running_agents(self, monkeypatch):
        from think.thinking import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

            self._write_record(
                tmpdir,
                "work",
                "20260209",
                {
                    "id": "coding_100000_300",
                    "activity": "coding",
                    "source": "cogitate",
                    "segments": ["100000_300"],
                    "level_avg": 0.5,
                    "description": "Synthetic",
                    "active_entities": [],
                },
            )

            monkeypatch.setattr(
                "think.thinking.get_talent_configs",
                lambda schedule: {"session_review": {"activities": ["*"]}},
            )

            result = run_activity_prompts(
                day="20260209",
                activity_id="coding_100000_300",
                facet="work",
            )

            assert result is True

    def test_emits_think_events(self, monkeypatch):
        from think.thinking import run_activity_prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

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
                "think.thinking.get_talent_configs", lambda schedule: configs
            )
            monkeypatch.setattr(
                "think.thinking.cortex_request",
                lambda prompt, name, config: "agent-1",
            )
            monkeypatch.setattr(
                "think.thinking.wait_for_uses",
                lambda ids, timeout: ({aid: "finish" for aid in ids}, []),
            )

            emitted = []
            monkeypatch.setattr(
                "think.thinking.emit", lambda event, **kw: emitted.append((event, kw))
            )

            run_activity_prompts(
                day="20260209",
                activity_id="coding_100000_300",
                facet="work",
            )

            events = [e[0] for e in emitted]
            assert "started" in events
            assert "group_started" in events
            assert "talent_started" in events
            assert "talent_completed" in events

            assert "group_completed" in events
            assert "completed" in events

            # Verify mode=activity in started event
            started_kw = next(kw for ev, kw in emitted if ev == "started")
            assert started_kw["mode"] == "activity"
            assert started_kw["activity"] == "coding_100000_300"
            assert started_kw["facet"] == "work"


class TestActivityPersistence:
    """Verify state machine completed records persist and load correctly."""

    def test_completed_record_persisted_and_found(self, monkeypatch):
        import tempfile

        from think.activities import append_activity_record, load_activity_records
        from think.activity_state_machine import ActivityStateMachine

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

            sm = ActivityStateMachine()
            sm.update(
                {
                    "density": "active",
                    "content_type": "coding",
                    "activity_summary": "Writing tests",
                    "entities": [],
                    "facets": [
                        {"facet": "work", "activity": "coding", "level": "high"}
                    ],
                    "meeting_detected": False,
                    "speakers": [],
                    "recommend": {},
                },
                "090000_300",
                "20260304",
            )
            changes = sm.update(
                {
                    "density": "active",
                    "content_type": "meeting",
                    "activity_summary": "Stand-up",
                    "entities": [],
                    "facets": [
                        {"facet": "work", "activity": "meeting", "level": "medium"}
                    ],
                    "meeting_detected": True,
                    "speakers": [],
                    "recommend": {},
                },
                "090500_300",
                "20260304",
            )

            # Find the ended change
            ended = [c for c in changes if c.get("state") == "ended"]
            assert len(ended) == 1
            facet = ended[0]["_facet"]

            # Persist completed record (what thinking.py now does)
            completed = sm.get_completed_activities()
            assert len(completed) == 1
            rec = completed[0]
            assert isinstance(rec["created_at"], int)
            append_activity_record(facet, "20260304", rec)

            # Verify load finds it (what run_activity_prompts does)
            records = load_activity_records(facet, "20260304")
            assert len(records) == 1
            assert records[0]["id"] == rec["id"]
            assert records[0]["activity"] == "coding"
            assert isinstance(records[0]["created_at"], int)


class TestActivityPersistenceRoundTrip:
    """Full round-trip: state machine → append → load → field verification."""

    def _sense(
        self,
        content_type="coding",
        density="active",
        facets=None,
        summary="Working.",
        entities=None,
    ):
        if facets is None:
            facets = [{"facet": "work", "activity": content_type, "level": "high"}]
        return {
            "density": density,
            "content_type": content_type,
            "activity_summary": summary,
            "entities": entities or [],
            "facets": facets,
            "meeting_detected": False,
            "speakers": [],
            "recommend": {},
        }

    def test_multi_segment_round_trip(self, monkeypatch):
        """Multi-segment activity persists and loads with all segments intact."""
        from think.activities import append_activity_record, load_activity_records
        from think.activity_state_machine import ActivityStateMachine

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

            sm = ActivityStateMachine()
            sm.update(self._sense(content_type="coding"), "090000_300", "20260304")
            sm.update(self._sense(content_type="coding"), "090500_300", "20260304")
            sm.update(self._sense(content_type="coding"), "091000_300", "20260304")
            # End via type change
            changes = sm.update(
                self._sense(content_type="meeting"), "091500_300", "20260304"
            )

            ended = [c for c in changes if c.get("state") == "ended"]
            assert len(ended) == 1

            # Simulate thinking.py facet_by_id logic
            facet_by_id = {
                c["id"]: c.get("_facet", "__")
                for c in changes
                if c.get("state") == "ended"
            }
            for rec in sm.get_completed_activities():
                if rec["id"] in facet_by_id:
                    append_activity_record(facet_by_id[rec["id"]], "20260304", rec)

            # Verify loaded record matches
            records = load_activity_records("work", "20260304")
            assert len(records) == 1
            r = records[0]
            assert r["segments"] == ["090000_300", "090500_300", "091000_300"]
            assert r["activity"] == "coding"
            assert isinstance(r["created_at"], int)
            assert r["created_at"] > 0

    def test_idle_ending_round_trip(self, monkeypatch):
        """Idle transition persists records correctly."""
        from think.activities import append_activity_record, load_activity_records
        from think.activity_state_machine import ActivityStateMachine

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

            sm = ActivityStateMachine()
            sm.update(self._sense(content_type="coding"), "090000_300", "20260304")
            sm.update(self._sense(content_type="coding"), "090500_300", "20260304")
            changes = sm.update(self._sense(density="idle"), "091000_300", "20260304")

            facet_by_id = {
                c["id"]: c.get("_facet", "__")
                for c in changes
                if c.get("state") == "ended"
            }
            for rec in sm.get_completed_activities():
                if rec["id"] in facet_by_id:
                    append_activity_record(facet_by_id[rec["id"]], "20260304", rec)

            records = load_activity_records("work", "20260304")
            assert len(records) == 1
            assert records[0]["segments"] == ["090000_300", "090500_300"]

    def test_deduplication_prevents_double_write(self, monkeypatch):
        """Same record written twice only appears once in the file."""
        from think.activities import append_activity_record, load_activity_records
        from think.activity_state_machine import ActivityStateMachine

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

            sm = ActivityStateMachine()
            sm.update(self._sense(content_type="coding"), "090000_300", "20260304")
            sm.update(self._sense(content_type="meeting"), "090500_300", "20260304")

            rec = sm.get_completed_activities()[0]

            # Write once
            result1 = append_activity_record("work", "20260304", rec)
            assert result1 is True
            # Write again — should be rejected
            result2 = append_activity_record("work", "20260304", rec)
            assert result2 is False

            records = load_activity_records("work", "20260304")
            assert len(records) == 1

    def test_facet_by_id_only_persists_current_update(self, monkeypatch):
        """Only records from the current update() are persisted, not cumulative."""
        from think.activities import append_activity_record, load_activity_records
        from think.activity_state_machine import ActivityStateMachine

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

            sm = ActivityStateMachine()
            # Activity 1 ends
            sm.update(self._sense(content_type="coding"), "090000_300", "20260304")
            changes1 = sm.update(
                self._sense(content_type="meeting"), "090500_300", "20260304"
            )
            facet_by_id = {
                c["id"]: c.get("_facet", "__")
                for c in changes1
                if c.get("state") == "ended"
            }
            for rec in sm.get_completed_activities():
                if rec["id"] in facet_by_id:
                    append_activity_record(facet_by_id[rec["id"]], "20260304", rec)

            # Activity 2 continues (no ending)
            changes2 = sm.update(
                self._sense(content_type="meeting"), "091000_300", "20260304"
            )
            # No ended changes in this update
            facet_by_id2 = {
                c["id"]: c.get("_facet", "__")
                for c in changes2
                if c.get("state") == "ended"
            }
            # get_completed_activities() still returns activity 1, but
            # facet_by_id2 is empty so nothing should be re-written
            for rec in sm.get_completed_activities():
                if rec["id"] in facet_by_id2:
                    append_activity_record(facet_by_id2[rec["id"]], "20260304", rec)

            records = load_activity_records("work", "20260304")
            assert len(records) == 1  # Only activity 1, not duplicated

    def test_jsonl_field_fidelity(self, monkeypatch):
        """All fields survive JSON serialization round-trip exactly."""
        from think.activities import append_activity_record, load_activity_records
        from think.activity_state_machine import ActivityStateMachine

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

            entities = [
                {"type": "Person", "name": "Alice", "context": "dev"},
                {"type": "Tool", "name": "VS Code", "context": "editor"},
            ]
            sm = ActivityStateMachine()
            sm.update(
                self._sense(
                    content_type="coding",
                    entities=entities,
                    summary="Pair programming with Alice",
                ),
                "090000_300",
                "20260304",
            )
            sm.update(self._sense(content_type="meeting"), "090500_300", "20260304")

            rec = sm.get_completed_activities()[0]
            append_activity_record("work", "20260304", rec)
            loaded = load_activity_records("work", "20260304")[0]

            # Every field must round-trip exactly
            assert loaded["id"] == rec["id"]
            assert loaded["activity"] == rec["activity"]
            assert loaded["segments"] == rec["segments"]
            assert loaded["level_avg"] == rec["level_avg"]
            assert loaded["description"] == rec["description"]
            assert loaded["active_entities"] == rec["active_entities"]
            assert loaded["created_at"] == rec["created_at"]

    def test_pseudo_facet_persistence(self, monkeypatch):
        """Activities with no facets use '__' pseudo-facet for persistence."""
        from think.activities import append_activity_record, load_activity_records
        from think.activity_state_machine import ActivityStateMachine

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

            sm = ActivityStateMachine()
            sm.update(
                self._sense(content_type="coding", facets=[]), "090000_300", "20260304"
            )
            changes = sm.update(self._sense(density="idle"), "090500_300", "20260304")

            facet_by_id = {
                c["id"]: c.get("_facet", "__")
                for c in changes
                if c.get("state") == "ended"
            }
            for rec in sm.get_completed_activities():
                if rec["id"] in facet_by_id:
                    append_activity_record(facet_by_id[rec["id"]], "20260304", rec)

            # Record lands under the "__" pseudo-facet
            records = load_activity_records("__", "20260304")
            assert len(records) == 1
            assert records[0]["activity"] == "coding"

    def test_multi_facet_ending_persists_both(self, monkeypatch):
        """Multiple facets ending simultaneously all persist correctly.

        This tests the ended_pairs fix: the old facet_by_id dict would overwrite
        duplicate IDs, dropping all but one facet. The list-based approach preserves
        all (id, facet) pairs.
        """
        from think.activities import append_activity_record, load_activity_records
        from think.activity_state_machine import ActivityStateMachine

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

            two = [
                {"facet": "work", "activity": "coding", "level": "high"},
                {"facet": "personal", "activity": "browsing", "level": "low"},
            ]
            sm = ActivityStateMachine()
            sm.update(self._sense(facets=two), "090000_300", "20260304")
            sm.update(self._sense(facets=two), "090500_300", "20260304")
            # Both end via idle
            changes = sm.update(self._sense(density="idle"), "091000_300", "20260304")

            # Use the fixed ended_pairs approach (matches thinking.py)
            ended_pairs = [
                (c["id"], c.get("_facet", "__"))
                for c in changes
                if c.get("state") == "ended"
            ]
            completed_lookup = {}
            for rec in sm.get_completed_activities():
                completed_lookup.setdefault(rec["id"], rec)
            for activity_id, facet in ended_pairs:
                rec = completed_lookup.get(activity_id)
                if rec:
                    append_activity_record(facet, "20260304", rec)

            work_records = load_activity_records("work", "20260304")
            personal_records = load_activity_records("personal", "20260304")
            assert len(work_records) == 1
            assert len(personal_records) == 1
            # Both facets use top-level content_type as activity
            assert work_records[0]["activity"] == "coding"
            assert personal_records[0]["activity"] == "coding"
            # Both have 2 segments
            assert work_records[0]["segments"] == ["090000_300", "090500_300"]
            assert personal_records[0]["segments"] == ["090000_300", "090500_300"]


class TestCreatedAtRoutesCompat:
    """Verify created_at works with the routes.py comparison and display logic."""

    def test_created_at_passes_cutoff_filter(self, monkeypatch):
        """Simulate routes.py filtering — recent records pass, old records don't."""
        from datetime import datetime, timedelta

        from think.activities import append_activity_record, load_activity_records
        from think.activity_state_machine import ActivityStateMachine

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

            sm = ActivityStateMachine()
            sm.update(
                {
                    "density": "active",
                    "content_type": "coding",
                    "activity_summary": "test",
                    "entities": [],
                    "facets": [
                        {"facet": "work", "activity": "coding", "level": "high"}
                    ],
                    "meeting_detected": False,
                    "speakers": [],
                    "recommend": {},
                },
                "090000_300",
                "20260304",
            )
            sm.update(
                {
                    "density": "active",
                    "content_type": "meeting",
                    "activity_summary": "standup",
                    "entities": [],
                    "facets": [
                        {"facet": "work", "activity": "meeting", "level": "medium"}
                    ],
                    "meeting_detected": True,
                    "speakers": [],
                    "recommend": {},
                },
                "090500_300",
                "20260304",
            )
            rec = sm.get_completed_activities()[0]
            append_activity_record("work", "20260304", rec)

            records = load_activity_records("work", "20260304")
            record = records[0]

            # Simulate routes.py line 273 and 279
            now = datetime.now()
            cutoff_ts = (now - timedelta(hours=4)).timestamp() * 1000
            created_at = record.get("created_at", 0)

            # This comparison must not raise TypeError
            recent = created_at >= cutoff_ts
            assert recent is True  # Just created, must be recent

            # Simulate routes.py line 283-284
            dt = datetime.fromtimestamp(created_at / 1000)
            display_time = dt.strftime("%H:%M")
            assert len(display_time) == 5  # "HH:MM"

    def test_sort_by_created_at(self, monkeypatch):
        """Multiple records can be sorted by created_at (routes.py line 290)."""
        import time

        from think.activities import append_activity_record, load_activity_records
        from think.activity_state_machine import ActivityStateMachine

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", tmpdir)

            sm = ActivityStateMachine()
            sm.update(
                {
                    "density": "active",
                    "content_type": "coding",
                    "activity_summary": "first",
                    "entities": [],
                    "facets": [
                        {"facet": "work", "activity": "coding", "level": "high"}
                    ],
                    "meeting_detected": False,
                    "speakers": [],
                    "recommend": {},
                },
                "090000_300",
                "20260304",
            )
            changes1 = sm.update(
                {
                    "density": "active",
                    "content_type": "meeting",
                    "activity_summary": "second",
                    "entities": [],
                    "facets": [
                        {"facet": "work", "activity": "meeting", "level": "medium"}
                    ],
                    "meeting_detected": True,
                    "speakers": [],
                    "recommend": {},
                },
                "090500_300",
                "20260304",
            )
            # Persist first completed
            facet_by_id = {
                c["id"]: c.get("_facet", "__")
                for c in changes1
                if c.get("state") == "ended"
            }
            for rec in sm.get_completed_activities():
                if rec["id"] in facet_by_id:
                    append_activity_record(facet_by_id[rec["id"]], "20260304", rec)

            # Small delay so created_at differs
            time.sleep(0.01)

            changes2 = sm.update(
                {
                    "density": "active",
                    "content_type": "coding",
                    "activity_summary": "third",
                    "entities": [],
                    "facets": [
                        {"facet": "work", "activity": "coding", "level": "high"}
                    ],
                    "meeting_detected": False,
                    "speakers": [],
                    "recommend": {},
                },
                "091000_300",
                "20260304",
            )
            facet_by_id2 = {
                c["id"]: c.get("_facet", "__")
                for c in changes2
                if c.get("state") == "ended"
            }
            for rec in sm.get_completed_activities():
                if rec["id"] in facet_by_id2:
                    append_activity_record(facet_by_id2[rec["id"]], "20260304", rec)

            records = load_activity_records("work", "20260304")
            assert len(records) == 2
            # Simulate routes.py sort
            sorted_records = sorted(
                records, key=lambda a: a.get("created_at", 0), reverse=True
            )
            assert sorted_records[0]["created_at"] >= sorted_records[1]["created_at"]


# ---------------------------------------------------------------------------
# Activity template variables in agents.py
# ---------------------------------------------------------------------------


class TestActivityTemplateVars:
    """Tests for activity template variables in _build_prompt_context."""

    def test_activity_vars_populated(self):
        from think.talents import _build_prompt_context

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
        from think.talents import _build_prompt_context

        ctx = _build_prompt_context("20260209", None, None)

        assert "activity_id" not in ctx
        assert "activity_type" not in ctx

    def test_empty_entities(self):
        from think.talents import _build_prompt_context

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
        from think.talents import _build_prompt_context

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

    def test_facet_and_activity_md_dir_populated(self, monkeypatch, tmp_path):
        from think.talents import _build_prompt_context

        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

        ctx = _build_prompt_context(
            day="20260418",
            segment=None,
            span=None,
            facet="work",
        )

        assert ctx["facet"] == "work"
        assert ctx["activity_md_dir"] == f"{tmp_path}/facets/work/activities/20260418/"

    def test_activity_md_dir_omitted_without_facet(self):
        from think.talents import _build_prompt_context

        ctx = _build_prompt_context(day="20260418", segment=None, span=None)

        assert "facet" not in ctx
        assert "activity_md_dir" not in ctx


def test_prepare_config_substitutes_facet_and_activity_md_dir_for_daily_cogitate(
    tmp_path, monkeypatch
):
    import importlib

    import think.talent
    from think.utils import day_path

    mod = importlib.import_module("think.talents")

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    day_dir = day_path("20260418")
    day_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(think.talent, "TALENT_DIR", tmp_path)

    test_agent = tmp_path / "test_activities_cogitate.md"
    test_agent.write_text(
        "{\n"
        '  "type": "cogitate",\n'
        '  "schedule": "daily",\n'
        '  "priority": 30,\n'
        '  "multi_facet": true\n'
        "}\n\n"
        "Facet: $facet\n"
        "Dir: $activity_md_dir\n"
    )

    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    config = mod.prepare_config(
        {
            "name": "test_activities_cogitate",
            "day": "20260418",
            "facet": "work",
        }
    )

    assert "Facet: work" in config["user_instruction"]
    assert "facets/work/activities/20260418/" in config["user_instruction"]
    assert "$facet" not in config["user_instruction"]
    assert "$activity_md_dir" not in config["user_instruction"]


# ---------------------------------------------------------------------------
# Talent config validation for activity schedule
# ---------------------------------------------------------------------------


class TestTalentActivityValidation:
    """Tests for activity schedule validation in get_talent_configs."""

    def _isolate_talent(self, monkeypatch, tmp_path):
        """Point talent discovery at tmp_path only (no real talent/ or apps/)."""
        talent_dir = tmp_path / "talent"
        talent_dir.mkdir(exist_ok=True)
        monkeypatch.setattr("think.talent.TALENT_DIR", talent_dir)
        monkeypatch.setattr("think.talent.APPS_DIR", tmp_path / "no_apps")
        return talent_dir

    def test_missing_activities_field_raises(self, monkeypatch, tmp_path):
        import frontmatter

        from think.talent import get_talent_configs

        talent_dir = self._isolate_talent(monkeypatch, tmp_path)

        post = frontmatter.Post(
            "Test prompt",
            schedule="activity",
            priority=50,
            type="generate",
            output="md",
            # Missing 'activities' field
        )
        (talent_dir / "test_agent.md").write_text(frontmatter.dumps(post))

        with pytest.raises(ValueError, match="non-empty 'activities' list"):
            get_talent_configs(schedule="activity")

    def test_valid_activities_field_passes(self, monkeypatch, tmp_path):
        import frontmatter

        from think.talent import get_talent_configs

        talent_dir = self._isolate_talent(monkeypatch, tmp_path)

        post = frontmatter.Post(
            "Test prompt",
            schedule="activity",
            priority=50,
            type="generate",
            output="md",
            activities=["coding", "meeting"],
        )
        (talent_dir / "test_agent.md").write_text(frontmatter.dumps(post))

        configs = get_talent_configs(schedule="activity")
        assert "test_agent" in configs
        assert configs["test_agent"]["activities"] == ["coding", "meeting"]

    def test_wildcard_activities_passes(self, monkeypatch, tmp_path):
        import frontmatter

        from think.talent import get_talent_configs

        talent_dir = self._isolate_talent(monkeypatch, tmp_path)

        post = frontmatter.Post(
            "Test prompt",
            schedule="activity",
            priority=50,
            type="generate",
            output="md",
            activities=["*"],
        )
        (talent_dir / "test_agent.md").write_text(frontmatter.dumps(post))

        configs = get_talent_configs(schedule="activity")
        assert "test_agent" in configs
        assert configs["test_agent"]["activities"] == ["*"]


# ---------------------------------------------------------------------------
# CLI argument validation
# ---------------------------------------------------------------------------


class TestActivityCLIArgs:
    """Tests for think CLI argument validation."""

    def test_activity_requires_facet(self, monkeypatch):
        from think.thinking import parse_args

        parser = parse_args()

        monkeypatch.setattr(
            "sys.argv",
            ["sol think", "--activity", "coding_100000_300", "--day", "20260209"],
        )

        # parse_args returns the parser, not args — need to test via main()
        # Instead test the parser directly
        args = parser.parse_args(
            ["--activity", "coding_100000_300", "--day", "20260209"]
        )
        assert args.activity == "coding_100000_300"
        assert args.facet is None  # Validation happens in main()

    def test_activity_args_parsed(self):
        from think.thinking import parse_args

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
