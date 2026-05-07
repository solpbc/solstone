# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("entity_name", "entity_type", "source"),
    [
        ("Claude Code", "Tool", "screen"),
        ("Mozilla Ventures", "Company", "transcript"),
        ("Google Meet", "Tool", "screen"),
    ],
)
def test_sense_entity_role_source_survive_splitter_without_changing_active_entity_flatten(
    tmp_path, entity_name: str, entity_type: str, source: str
):
    from solstone.think.activity_state_machine import ActivityStateMachine
    from solstone.think.sense_splitter import write_sense_outputs

    # This is a schema/plumbing regression test, not an LLM-behavioral test.
    day = "20260418"
    segment = "090000_300"
    sense_json = {
        "density": "active",
        "content_type": "coding",
        "activity_summary": "Worked through captured activity.",
        "entities": [
            {
                "type": entity_type,
                "name": entity_name,
                "role": "mentioned",
                "source": source,
                "context": "Detected by fixture-driven test input",
            }
        ],
        "facets": [{"facet": "work", "activity": "coding", "level": "high"}],
        "meeting_detected": False,
        "speakers": [],
        "recommend": {},
    }

    seg_dir = Path(tmp_path) / day / "default" / segment
    write_sense_outputs(sense_json, seg_dir)

    sense_md = (seg_dir / "talents" / "sense.md").read_text(encoding="utf-8")
    assert f"{entity_name} (role=mentioned, source={source})" in sense_md

    changes = ActivityStateMachine().update(sense_json, segment, day)
    assert changes[0]["active_entities"] == [entity_name]
