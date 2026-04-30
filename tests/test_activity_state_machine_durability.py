# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Durability tests for ActivityStateMachine snapshots."""

import json
from pathlib import Path

from think.activities import (
    append_activity_record,
    load_activity_records,
    make_activity_id,
)
from think.activity_state_machine import ActivityStateMachine
from think.thinking import _write_json_atomic


def _sense(content_type: str = "coding", density: str = "active", facet: str = "test"):
    return {
        "density": density,
        "content_type": content_type,
        "activity_summary": f"{content_type} work",
        "entities": [],
        "facets": [{"facet": facet, "activity": content_type, "level": "high"}],
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


def _append_ended_records(
    state_machine: ActivityStateMachine, changes: list[dict], day: str
) -> None:
    completed_lookup = {}
    for record in state_machine.get_completed_activities():
        completed_lookup.setdefault(record["id"], record)
    for change in changes:
        if change.get("state") != "ended":
            continue
        record = completed_lookup.get(change["id"])
        if record:
            append_activity_record(change["facet"], day, record)


def test_state_survives_subprocess_boundary(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    sm1 = ActivityStateMachine(journal_root=tmp_path)
    sm1.update(_sense(), "090000_300", "20260427")
    assert set(sm1.state) == {"test"}
    _persist_snapshot(tmp_path, sm1)

    sm2 = ActivityStateMachine(journal_root=tmp_path)
    assert sm2.last_segment_key == "090000_300"
    assert sm2.last_segment_day == "20260427"
    assert sm2.state["test"]["segments"] == ["090000_300"]

    changes = sm2.update(_sense(density="idle"), "090500_300", "20260427")
    ended = [change for change in changes if change.get("state") == "ended"]
    assert len(ended) == 1
    _append_ended_records(sm2, changes, "20260427")
    _persist_snapshot(tmp_path, sm2)

    records = load_activity_records("test", "20260427")
    assert len(records) == 1
    assert records[0]["id"] == make_activity_id("coding", "090000_300")
    assert records[0]["segments"] == ["090000_300"]

    sm3 = ActivityStateMachine(journal_root=tmp_path)
    assert sm3.state == {}
    assert sm3.last_segment_key == "090500_300"
    assert sm3.last_segment_day == "20260427"


def test_day_boundary_routes_ended_record_to_prior_day(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    sm1 = ActivityStateMachine(journal_root=tmp_path)
    sm1.update(_sense(), "233000_300", "20260304")
    _persist_snapshot(tmp_path, sm1)

    sm2 = ActivityStateMachine(journal_root=tmp_path)
    routing_day = sm2.last_segment_day or "20260305"
    changes = sm2.update(_sense(), "001500_300", "20260305")

    ended = [change for change in changes if change.get("state") == "ended"]
    active = [change for change in changes if change.get("state") == "active"]
    assert len(ended) == 1
    assert len(active) == 1
    _append_ended_records(sm2, changes, routing_day)
    _persist_snapshot(tmp_path, sm2)

    prior_day_records = load_activity_records("test", "20260304")
    current_day_records = load_activity_records("test", "20260305")
    assert len(prior_day_records) == 1
    assert current_day_records == []
    assert prior_day_records[0]["segments"] == ["233000_300"]
    assert sm2.state["test"]["since"] == "001500_300"
    assert sm2.last_segment_day == "20260305"


def test_three_active_segments_then_idle_writes_one_record(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    day = "20260427"

    for segment in ("090000_300", "090505_300", "091010_300"):
        state_machine = ActivityStateMachine(journal_root=tmp_path)
        state_machine.update(_sense(facet="work"), segment, day)
        _persist_snapshot(tmp_path, state_machine)

    state_machine = ActivityStateMachine(journal_root=tmp_path)
    routing_day = state_machine.last_segment_day or day
    changes = state_machine.update(
        _sense(density="idle", facet="work"), "091515_300", day
    )
    ended = [change for change in changes if change.get("state") == "ended"]
    assert len(ended) == 1
    _append_ended_records(state_machine, changes, routing_day)
    _persist_snapshot(tmp_path, state_machine)

    records = load_activity_records("work", day)
    assert len(records) == 1
    assert records[0]["segments"] == ["090000_300", "090505_300", "091010_300"]
    assert len(records[0]["segments"]) == 3
    assert state_machine.state == {}


def test_crash_between_append_and_snapshot_is_idempotent(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    day = "20260427"

    sm1 = ActivityStateMachine(journal_root=tmp_path)
    sm1.update(_sense(facet="work"), "090000_300", day)
    _persist_snapshot(tmp_path, sm1)

    sm2 = ActivityStateMachine(journal_root=tmp_path)
    routing_day = sm2.last_segment_day or day
    changes = sm2.update(_sense(density="idle", facet="work"), "090505_300", day)
    _append_ended_records(sm2, changes, routing_day)
    assert len(load_activity_records("work", day)) == 1

    sm3 = ActivityStateMachine(journal_root=tmp_path)
    routing_day = sm3.last_segment_day or day
    retry_changes = sm3.update(_sense(density="idle", facet="work"), "090505_300", day)
    _append_ended_records(sm3, retry_changes, routing_day)
    records = load_activity_records("work", day)
    assert len(records) == 1
    assert records[0]["id"] == make_activity_id("coding", "090000_300")

    _persist_snapshot(tmp_path, sm3)
    sm4 = ActivityStateMachine(journal_root=tmp_path)
    assert sm4.state == {}
    assert sm4.last_segment_key == "090505_300"
    assert sm4.last_segment_day == day


def test_batch_construction_has_no_journal_root_and_skips_snapshot(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    marker = {
        "last_segment_key": "marker",
        "last_segment_day": "20990101",
        "active": {},
    }
    state_path = tmp_path / "awareness" / "activity_state.json"
    _write_json_atomic(state_path, marker)
    mtime_before = state_path.stat().st_mtime_ns

    state_machine = ActivityStateMachine()
    assert state_machine.journal_root is None
    state_machine.update(_sense(facet="work"), "090000_300", "20260427")
    state_machine.update(_sense(facet="work"), "090505_300", "20260427")
    state_machine.update(_sense(density="idle", facet="work"), "091010_300", "20260427")

    if state_machine.journal_root is not None:
        _persist_snapshot(state_machine.journal_root, state_machine)

    assert state_path.stat().st_mtime_ns == mtime_before
    assert json.loads(state_path.read_text(encoding="utf-8")) == marker


def test_run_segment_sense_emits_activity_events(tmp_path: Path, monkeypatch):
    from think import thinking as think

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    day = "20260427"
    stream = "default"
    segment = "090505_300"

    sm1 = ActivityStateMachine(journal_root=tmp_path)
    sm1.update(_sense(facet="work"), "090000_300", day)
    _persist_snapshot(tmp_path, sm1)

    talents_dir = tmp_path / "chronicle" / day / stream / segment / "talents"
    talents_dir.mkdir(parents=True)
    (talents_dir / "sense.json").write_text(
        json.dumps(_sense(density="idle", facet="work")),
        encoding="utf-8",
    )

    events = []
    monkeypatch.setattr(
        think,
        "get_talent_configs",
        lambda schedule=None, **kwargs: {
            "sense": {"priority": 10, "type": "generate", "output": "json"}
        },
    )
    monkeypatch.setattr(think, "_cortex_request_with_retry", lambda **kwargs: "sense-1")
    monkeypatch.setattr(
        think, "_drain_priority_batch", lambda *args, **kwargs: (1, 0, [])
    )
    monkeypatch.setattr(
        think, "_jsonl_log", lambda event, **fields: events.append((event, fields))
    )
    monkeypatch.setattr(think, "run_activity_prompts", lambda **kwargs: True)
    monkeypatch.setattr(think, "_callosum", None)

    success, failed, failed_names = think.run_segment_sense(
        day,
        segment,
        refresh=False,
        verbose=False,
        stream=stream,
        state_machine=ActivityStateMachine(journal_root=tmp_path),
    )

    assert (success, failed, failed_names) == (1, 0, [])
    event_names = [event for event, _fields in events]
    assert "activity.detected" in event_names
    assert "activity.persisted" in event_names
    detected = [fields for event, fields in events if event == "activity.detected"]
    persisted = [fields for event, fields in events if event == "activity.persisted"]
    assert detected[0]["change"] == "ended_idle"
    assert persisted[0]["change"] == "ended_idle"
