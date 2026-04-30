# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Hysteresis tests for the deterministic activity state machine."""

import json
from pathlib import Path

from think.activity_state_machine import ActivityStateMachine
from think.thinking import _write_json_atomic

DAY = "20260304"


def _sense(
    content_type: str = "coding",
    density: str = "active",
    facets: list[dict] | None = None,
    summary: str = "Working on code.",
    entities: list[dict] | None = None,
) -> dict:
    if facets is None:
        facets = [{"facet": "work", "activity": content_type, "level": "high"}]
    return {
        "density": density,
        "content_type": content_type,
        "activity_summary": summary,
        "entities": entities or [],
        "facets": facets,
        "meeting_detected": content_type == "meeting",
        "speakers": [],
        "recommend": {},
    }


def _persist_snapshot(journal_root: Path, state_machine: ActivityStateMachine) -> None:
    snapshot = {
        "last_segment_key": state_machine.last_segment_key,
        "last_segment_day": state_machine.last_segment_day,
        "active": {
            facet: {k: v for k, v in entry.items() if k != "_change"}
            for facet, entry in state_machine.state.items()
        },
    }
    _write_json_atomic(journal_root / "awareness" / "activity_state.json", snapshot)


def test_single_segment_facet_wobble_does_not_end():
    sm = ActivityStateMachine()
    sm.update(_sense(), "090000_300", DAY)

    missing = sm.update(_sense(facets=[]), "090500_300", DAY)
    assert not any(change.get("_change") == "ended_facet_gone" for change in missing)
    assert missing[0]["_change"] == "facet_gone_pending"
    assert sm.state["work"]["segments"] == ["090000_300", "090500_300"]

    returned = sm.update(_sense(), "091000_300", DAY)
    assert not any(change.get("_change") == "ended_facet_gone" for change in returned)
    assert sm.state["work"]["state"] == "active"
    assert sm.state["work"]["segments"] == [
        "090000_300",
        "090500_300",
        "091000_300",
    ]
    assert sm.state["work"]["_pending_facet_misses"] == 0


def test_two_segment_facet_gone_ends_at_k():
    sm = ActivityStateMachine()
    sm.update(_sense(), "090000_300", DAY)
    sm.update(_sense(facets=[]), "090500_300", DAY)

    changes = sm.update(_sense(facets=[]), "091000_300", DAY)
    ended = [
        change for change in changes if change.get("_change") == "ended_facet_gone"
    ]

    assert len(ended) == 1
    assert ended[0]["state"] == "ended"
    assert sm.state == {}
    completed = sm.get_completed_activities()
    assert completed[0]["segments"] == ["090000_300", "090500_300"]


def test_single_segment_type_wobble_does_not_end():
    sm = ActivityStateMachine()
    sm.update(_sense(summary="Writing code."), "090000_300", DAY)

    wobble = sm.update(
        _sense(content_type="meeting", summary="Stand-up."), "090500_300", DAY
    )
    assert not any(change.get("_change") == "ended_type_change" for change in wobble)
    assert wobble[0]["_change"] == "type_change_pending"
    assert sm.state["work"]["description"] == "Writing code."
    assert sm.state["work"]["segments"] == ["090000_300", "090500_300"]

    returned = sm.update(_sense(summary="Still coding."), "091000_300", DAY)
    assert not any(change.get("_change") == "ended_type_change" for change in returned)
    assert sm.get_completed_activities() == []
    assert sm.state["work"]["activity"] == "coding"
    assert sm.state["work"]["description"] == "Still coding."
    assert sm.state["work"]["_pending_type"] is None
    assert sm.state["work"]["_pending_type_count"] == 0
    assert sm.state["work"]["segments"] == [
        "090000_300",
        "090500_300",
        "091000_300",
    ]


def test_two_segment_type_change_ends_at_k():
    sm = ActivityStateMachine()
    sm.update(_sense(content_type="coding"), "090000_300", DAY)
    sm.update(_sense(content_type="meeting"), "090500_300", DAY)

    changes = sm.update(_sense(content_type="meeting"), "091000_300", DAY)
    ended = [
        change for change in changes if change.get("_change") == "ended_type_change"
    ]
    active = [change for change in changes if change.get("_change") == "new"]

    assert len(ended) == 1
    assert ended[0]["activity"] == "coding"
    assert len(active) == 1
    assert active[0]["activity"] == "meeting"
    assert active[0]["since"] == "091000_300"
    assert sm.state["work"]["segments"] == ["091000_300"]
    completed = sm.get_completed_activities()
    assert completed[0]["segments"] == ["090000_300", "090500_300"]


def test_idle_still_ends_immediately():
    sm = ActivityStateMachine()
    sm.update(_sense(content_type="coding"), "090000_300", DAY)
    sm.update(_sense(content_type="meeting"), "090500_300", DAY)

    changes = sm.update(_sense(density="idle"), "091000_300", DAY)
    ended = [change for change in changes if change.get("_change") == "ended_idle"]

    assert len(ended) == 1
    assert sm.state == {}
    assert sm.get_completed_activities()[0]["segments"] == [
        "090000_300",
        "090500_300",
    ]


def test_gap_still_ends_immediately():
    sm = ActivityStateMachine()
    sm.update(_sense(content_type="coding"), "090000_300", DAY)
    sm.update(_sense(content_type="meeting"), "090500_300", DAY)

    changes = sm.update(_sense(content_type="coding"), "100600_300", DAY)
    ended_gap = [change for change in changes if change.get("_change") == "ended_gap"]
    new = [change for change in changes if change.get("_change") == "new"]

    assert len(ended_gap) == 1
    assert len(new) == 1
    assert sm.get_completed_activities()[0]["segments"] == [
        "090000_300",
        "090500_300",
    ]


def test_pending_counters_reset_on_continuing_segment():
    sm = ActivityStateMachine()
    sm.update(_sense(content_type="coding"), "090000_300", DAY)
    sm.update(_sense(content_type="meeting"), "090500_300", DAY)

    sm.update(_sense(content_type="coding"), "091000_300", DAY)
    entry = sm.state["work"]

    assert entry["_pending_facet_misses"] == 0
    assert entry["_pending_type"] is None
    assert entry["_pending_type_count"] == 0


def test_pending_counters_round_trip_via_awareness_json(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    sm1 = ActivityStateMachine(journal_root=tmp_path)
    sm1.update(_sense(), "090000_300", DAY)
    sm1.update(_sense(facets=[]), "090500_300", DAY)
    _persist_snapshot(tmp_path, sm1)

    state_path = tmp_path / "awareness" / "activity_state.json"
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["active"]["work"]["_pending_facet_misses"] == 1

    sm2 = ActivityStateMachine(journal_root=tmp_path)
    changes = sm2.update(_sense(facets=[]), "091000_300", DAY)
    ended = [
        change for change in changes if change.get("_change") == "ended_facet_gone"
    ]

    assert len(ended) == 1
    assert sm2.state == {}
    assert sm2.get_completed_activities()[0]["segments"] == [
        "090000_300",
        "090500_300",
    ]


def test_pending_type_counters_round_trip_via_awareness_json(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    sm1 = ActivityStateMachine(journal_root=tmp_path)
    sm1.update(_sense(content_type="coding"), "090000_300", DAY)
    sm1.update(_sense(content_type="meeting"), "090500_300", DAY)
    _persist_snapshot(tmp_path, sm1)

    state_path = tmp_path / "awareness" / "activity_state.json"
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["active"]["work"]["_pending_type"] == "meeting"
    assert data["active"]["work"]["_pending_type_count"] == 1

    sm2 = ActivityStateMachine(journal_root=tmp_path)
    changes = sm2.update(_sense(content_type="meeting"), "091000_300", DAY)
    ended = [
        change for change in changes if change.get("_change") == "ended_type_change"
    ]

    assert len(ended) == 1
    assert ended[0]["state"] == "ended"


def test_facet_recovery_resets_pending_counter():
    sm = ActivityStateMachine()
    sm.update(_sense(), "090000_300", DAY)
    sm.update(_sense(facets=[]), "090500_300", DAY)
    sm.update(_sense(), "091000_300", DAY)

    changes = sm.update(_sense(facets=[]), "091500_300", DAY)

    assert not any(change.get("_change") == "ended_facet_gone" for change in changes)
    assert sm.state["work"]["_pending_facet_misses"] == 1
    assert sm.state["work"]["segments"] == [
        "090000_300",
        "090500_300",
        "091000_300",
        "091500_300",
    ]


def test_alternating_type_wobble_does_not_accumulate():
    sm = ActivityStateMachine()
    sm.update(_sense(content_type="coding"), "090000_300", DAY)
    sm.update(_sense(content_type="meeting"), "090500_300", DAY)
    sm.update(_sense(content_type="coding"), "091000_300", DAY)
    sm.update(_sense(content_type="meeting"), "091500_300", DAY)
    changes = sm.update(_sense(content_type="coding"), "092000_300", DAY)

    assert not any(change.get("_change") == "ended_type_change" for change in changes)
    assert sm.get_completed_activities() == []
    assert sm.state["work"]["activity"] == "coding"
    assert sm.state["work"]["_pending_type"] is None
    assert sm.state["work"]["_pending_type_count"] == 0
    assert sm.state["work"]["segments"] == [
        "090000_300",
        "090500_300",
        "091000_300",
        "091500_300",
        "092000_300",
    ]
