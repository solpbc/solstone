# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
from pathlib import Path

from think import dream
from think.activities import load_activity_records, make_activity_id
from think.activity_state_machine import ActivityStateMachine

DAY = "20260304"
STREAM = "default"


SEGMENTS = [
    (
        "090000_300",
        {
            "density": "active",
            "content_type": "coding",
            "activity_summary": "Implementing auth module",
            "facets": [{"facet": "work", "level": "high"}],
            "meeting_detected": False,
            "recommend": {},
            "entities": [{"name": "Acme"}],
        },
    ),
    (
        "090500_300",
        {
            "density": "active",
            "content_type": "coding",
            "activity_summary": "Implementing auth module",
            "facets": [{"facet": "work", "level": "high"}],
            "meeting_detected": False,
            "recommend": {},
            "entities": [{"name": "Acme"}],
        },
    ),
    (
        "091000_300",
        {
            "density": "active",
            "content_type": "meeting",
            "activity_summary": "Sprint planning standup",
            "facets": [{"facet": "work", "level": "medium"}],
            "meeting_detected": True,
            "speakers": ["Alice", "Bob"],
            "recommend": {},
            "entities": [{"name": "Acme"}],
        },
    ),
    (
        "091500_300",
        {
            "density": "active",
            "content_type": "meeting",
            "activity_summary": "Sprint planning standup",
            "facets": [{"facet": "work", "level": "medium"}],
            "meeting_detected": True,
            "speakers": ["Alice", "Bob"],
            "recommend": {},
            "entities": [{"name": "Acme"}],
        },
    ),
    (
        "092000_300",
        {
            "density": "idle",
            "content_type": "idle",
            "activity_summary": "",
            "facets": [],
            "meeting_detected": False,
            "recommend": {},
            "entities": [],
        },
    ),
    (
        "100000_300",
        {
            "density": "active",
            "content_type": "coding",
            "activity_summary": "Reviewing PR feedback",
            "facets": [{"facet": "work", "level": "low"}],
            "meeting_detected": False,
            "recommend": {},
            "entities": [{"name": "Acme"}],
        },
    ),
]


class TestPipelineSmokeTest:
    def test_full_pipeline_smoke(self, tmp_path: Path, monkeypatch):
        journal = tmp_path / "journal"
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))

        state_machine = ActivityStateMachine()
        activity_calls = []

        def _segment_configs():
            return {
                "sense": {
                    "priority": 10,
                    "type": "generate",
                    "output": "json",
                    "schedule": "segment",
                },
                "entities": {
                    "priority": 20,
                    "type": "cogitate",
                    "schedule": "segment",
                },
                "documents": {
                    "priority": 20,
                    "type": "cogitate",
                    "schedule": "segment",
                },
            }

        monkeypatch.setattr(
            dream,
            "cortex_request",
            lambda prompt, name, config=None: f"agent-{name}",
        )
        monkeypatch.setattr(
            dream,
            "wait_for_agents",
            lambda agent_ids, timeout=600: ({aid: "finish" for aid in agent_ids}, []),
        )
        monkeypatch.setattr(dream, "_callosum", None)
        monkeypatch.setattr(
            dream,
            "run_activity_prompts",
            lambda **kwargs: activity_calls.append(kwargs) or True,
        )
        monkeypatch.setattr(
            dream,
            "get_talent_configs",
            lambda schedule=None, **kwargs: _segment_configs(),
        )

        for segment_key, sense_dict in SEGMENTS:
            seg_dir = journal / "chronicle" / DAY / STREAM / segment_key
            (seg_dir / "agents").mkdir(parents=True, exist_ok=True)
            (seg_dir / "agents" / "sense.json").write_text(json.dumps(sense_dict))

            dream.run_segment_sense(
                day=DAY,
                segment=segment_key,
                refresh=False,
                verbose=False,
                stream=STREAM,
                state_machine=state_machine,
            )

        for seg_key in [
            "090000_300",
            "090500_300",
            "091000_300",
            "091500_300",
            "100000_300",
        ]:
            seg_agents = journal / "chronicle" / DAY / STREAM / seg_key / "agents"
            assert (seg_agents / "sense.json").exists()
            assert (seg_agents / "activity.md").exists()
            assert (seg_agents / "density.json").exists()
            density = json.loads((seg_agents / "density.json").read_text())
            assert density["classification"] == "active"
            assert (seg_agents / "facets.json").exists()
            facets = json.loads((seg_agents / "facets.json").read_text())
            assert isinstance(facets, list)

        for seg_key in ["091000_300", "091500_300"]:
            speakers = json.loads(
                (
                    journal
                    / "chronicle"
                    / DAY
                    / STREAM
                    / seg_key
                    / "agents"
                    / "speakers.json"
                ).read_text()
            )
            assert speakers == ["Alice", "Bob"]

        for seg_key in ["090000_300", "090500_300", "100000_300"]:
            assert not (
                journal
                / "chronicle"
                / DAY
                / STREAM
                / seg_key
                / "agents"
                / "speakers.json"
            ).exists()

        idle_density = json.loads(
            (
                journal
                / "chronicle"
                / DAY
                / STREAM
                / "092000_300"
                / "agents"
                / "density.json"
            ).read_text()
        )
        assert idle_density["classification"] == "idle"

        records = load_activity_records("work", DAY)
        assert len(records) == 2

        coding_rec = records[0]
        assert coding_rec["activity"] == "coding"
        assert coding_rec["segments"] == ["090000_300", "090500_300"]
        assert coding_rec["id"] == make_activity_id("coding", "090000_300")
        assert isinstance(coding_rec["created_at"], int)
        assert isinstance(coding_rec["level_avg"], float)
        assert coding_rec["level_avg"] == 1.0
        assert isinstance(coding_rec["active_entities"], list)
        assert isinstance(coding_rec["description"], str)
        assert len(coding_rec["description"]) > 0

        meeting_rec = records[1]
        assert meeting_rec["activity"] == "meeting"
        assert meeting_rec["segments"] == ["091000_300", "091500_300"]
        assert meeting_rec["id"] == make_activity_id("meeting", "091000_300")
        assert isinstance(meeting_rec["created_at"], int)
        assert isinstance(meeting_rec["level_avg"], float)
        assert meeting_rec["level_avg"] == 0.5
        assert isinstance(meeting_rec["active_entities"], list)
        assert isinstance(meeting_rec["description"], str)
        assert len(meeting_rec["description"]) > 0

        # Activity agents fire for BOTH endings:
        # 1. Coding ended by type change (segment 3, active path)
        # 2. Meeting ended by idle (segment 5, idle path)
        assert len(activity_calls) == 2

        assert activity_calls[0]["facet"] == "work"
        assert activity_calls[0]["activity_id"] == make_activity_id(
            "coding", "090000_300"
        )
        assert activity_calls[0]["day"] == DAY

        assert activity_calls[1]["facet"] == "work"
        assert activity_calls[1]["activity_id"] == make_activity_id(
            "meeting", "091000_300"
        )
        assert activity_calls[1]["day"] == DAY

        state_path = journal / "awareness" / "activity_state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert len(state) == 1
        assert state[0]["activity"] == "coding"
        assert state[0]["state"] == "active"
        assert state[0]["id"] == make_activity_id("coding", "100000_300")
