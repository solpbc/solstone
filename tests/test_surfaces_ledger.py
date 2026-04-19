# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

_DAY_MS = 86_400_000
_RUNNER = CliRunner()


def _utc_ms(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def _ledger_close_edits(record: dict) -> list[dict]:
    return [
        edit
        for edit in record.get("edits", [])
        if isinstance(edit, dict) and edit.get("fields") == ["ledger_close"]
    ]


def _minimal_facet_tree(tmp_path, facets=("work",), *, muted_facets=()) -> None:
    for facet in facets:
        facet_dir = tmp_path / "facets" / facet
        facet_dir.mkdir(parents=True, exist_ok=True)
        (facet_dir / "activities").mkdir(exist_ok=True)
        (facet_dir / "facet.json").write_text(
            json.dumps(
                {
                    "title": facet.title(),
                    "description": "",
                    "color": "",
                    "emoji": "",
                    "muted": facet in set(muted_facets),
                }
            ),
            encoding="utf-8",
        )


def _activity_record(
    record_id: str,
    created_at: int,
    *,
    activity: str = "meeting",
    hidden: bool = False,
) -> dict:
    return {
        "id": record_id,
        "activity": activity,
        "description": f"{record_id} description",
        "segments": [f"{record_id.split('_', 1)[-1]}"],
        "created_at": created_at,
        "hidden": hidden,
    }


def _commitment(
    *,
    owner: str = "Mina",
    owner_entity_id: str | None = "mina",
    action: str = "send proposal",
    counterparty: str = "Ravi",
    counterparty_entity_id: str | None = "ravi",
    when: str = "tomorrow",
    context: str = "Commitment context.",
) -> dict:
    return {
        "owner": owner,
        "owner_entity_id": owner_entity_id,
        "action": action,
        "counterparty": counterparty,
        "counterparty_entity_id": counterparty_entity_id,
        "when": when,
        "context": context,
    }


def _closure(
    *,
    owner: str = "Mina",
    owner_entity_id: str | None = "mina",
    action: str = "send proposal",
    counterparty: str = "Ravi",
    counterparty_entity_id: str | None = "ravi",
    resolution: str = "sent",
    context: str = "Closure context.",
) -> dict:
    return {
        "owner": owner,
        "owner_entity_id": owner_entity_id,
        "action": action,
        "counterparty": counterparty,
        "counterparty_entity_id": counterparty_entity_id,
        "resolution": resolution,
        "context": context,
    }


def _decision(
    *,
    owner: str = "Mina",
    owner_entity_id: str | None = "mina",
    action: str = "move launch review",
    context: str = "Decision context.",
) -> dict:
    return {
        "owner": owner,
        "owner_entity_id": owner_entity_id,
        "action": action,
        "context": context,
    }


def _write_story_activity(
    facet: str,
    day: str,
    record_id: str,
    created_at: int,
    *,
    commitments: list[dict] | None = None,
    closures: list[dict] | None = None,
    decisions: list[dict] | None = None,
    hidden: bool = False,
) -> None:
    from think.activities import append_activity_record, merge_story_fields

    append_activity_record(
        facet,
        day,
        _activity_record(record_id, created_at, hidden=hidden),
    )
    merge_story_fields(
        facet,
        day,
        record_id,
        story={
            "talent": "story",
            "body": f"{record_id} summary",
            "topics": ["ledger"],
            "confidence": 0.9,
        },
        commitments=commitments or [],
        closures=closures or [],
        decisions=decisions or [],
        actor="story",
    )


def test_append_edit_payload_validation():
    from think.activities import append_edit

    merged = append_edit(
        {"id": "coding_090000_300"},
        actor="cli:update",
        fields=["details"],
        note="updated",
        payload={"foo": "bar"},
    )
    assert merged["edits"][-1]["foo"] == "bar"

    with pytest.raises(
        ValueError, match="payload cannot overwrite canonical edit fields"
    ):
        append_edit(
            {"id": "coding_090000_300"},
            actor="cli:update",
            fields=["details"],
            note="updated",
            payload={"timestamp": "x"},
        )

    with pytest.raises(TypeError, match="payload must be dict\\[str, Any\\]"):
        append_edit(
            {"id": "coding_090000_300"},
            actor="cli:update",
            fields=["details"],
            note="updated",
            payload="not a dict",
        )


def test_pairing_happy_path(tmp_path, monkeypatch):
    from think.surfaces import ledger as ledger_surface

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _minimal_facet_tree(tmp_path)
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment()],
    )
    _write_story_activity(
        "work",
        "20260412",
        "meeting_110000_300",
        _utc_ms("2026-04-12T11:00:00Z"),
        closures=[_closure()],
    )

    items = ledger_surface.list(state="closed")

    assert len(items) == 1
    assert items[0].state == "closed"
    assert items[0].summary == "send proposal"
    assert len(items[0].sources) == 2


def test_action_fuzzy_just_above_threshold(tmp_path, monkeypatch):
    from think.surfaces import ledger as ledger_surface

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _minimal_facet_tree(tmp_path)
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment(action="send proposal")],
    )
    _write_story_activity(
        "work",
        "20260411",
        "meeting_100000_300",
        _utc_ms("2026-04-11T10:00:00Z"),
        closures=[_closure(action="sent the proposal")],
    )

    assert ledger_surface._actions_match(
        ledger_surface._normalize_action("send proposal"),
        ledger_surface._normalize_action("sent the proposal"),
    )
    assert len(ledger_surface.list(state="closed")) == 1


def test_action_fuzzy_just_below_threshold(tmp_path, monkeypatch):
    from think.surfaces import ledger as ledger_surface

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _minimal_facet_tree(tmp_path)
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment(action="send proposal")],
    )
    _write_story_activity(
        "work",
        "20260411",
        "meeting_100000_300",
        _utc_ms("2026-04-11T10:00:00Z"),
        commitments=[_commitment(action="share the proposal")],
    )

    assert not ledger_surface._actions_match(
        ledger_surface._normalize_action("send proposal"),
        ledger_surface._normalize_action("share the proposal"),
    )
    actions = [item.action for item in ledger_surface.list(state="open")]
    assert actions == ["send proposal", "share the proposal"]


def test_cross_facet_dedup(tmp_path, monkeypatch):
    from think.surfaces import ledger as ledger_surface

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _minimal_facet_tree(tmp_path, facets=("work", "personal"))
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment()],
    )
    _write_story_activity(
        "personal",
        "20260412",
        "meeting_100000_300",
        _utc_ms("2026-04-12T10:00:00Z"),
        closures=[_closure()],
    )

    items = ledger_surface.list(state="closed")

    assert len(items) == 1
    assert {source.facet for source in items[0].sources} == {"work", "personal"}


def test_manual_close_round_trip(tmp_path, monkeypatch):
    from think.activities import load_activity_records
    from think.surfaces import ledger as ledger_surface

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _minimal_facet_tree(tmp_path)
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment()],
    )

    item = ledger_surface.list(state="open")[0]
    closed = ledger_surface.close(item.id, note="done")
    record = load_activity_records("work", "20260410", include_hidden=True)[0]
    manual_edit = _ledger_close_edits(record)[0]

    assert closed.state == "closed"
    assert closed.closed_at == _utc_ms(manual_edit["timestamp"])
    refreshed = ledger_surface.get(item.id)
    assert refreshed is not None
    assert refreshed.state == "closed"
    assert refreshed.closed_at == closed.closed_at
    assert any(source.field == "edits" for source in refreshed.sources)


def test_idempotent_reclose(tmp_path, monkeypatch):
    from think.activities import load_activity_records
    from think.surfaces import ledger as ledger_surface

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _minimal_facet_tree(tmp_path)
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment()],
    )

    item = ledger_surface.list(state="open")[0]
    ledger_surface.close(item.id, note="first")
    ledger_surface.close(item.id, note="second")

    record = load_activity_records("work", "20260410", include_hidden=True)[0]
    closes = [
        edit for edit in record["edits"] if edit.get("fields") == ["ledger_close"]
    ]
    assert len(closes) == 1


def test_close_as_dropped(tmp_path, monkeypatch):
    from think.surfaces import ledger as ledger_surface

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _minimal_facet_tree(tmp_path)
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment()],
    )

    item = ledger_surface.list(state="open")[0]
    dropped = ledger_surface.close(item.id, note="not needed", as_state="dropped")
    assert dropped.state == "dropped"


def test_close_with_new_as_state_appends_and_latest_manual_wins(tmp_path, monkeypatch):
    from think.activities import load_activity_records
    from think.surfaces import ledger as ledger_surface

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _minimal_facet_tree(tmp_path)
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment()],
    )

    item = ledger_surface.list(state="open")[0]
    ledger_surface.close(item.id, note="done", as_state="closed")
    dropped = ledger_surface.close(item.id, note="actually dropped", as_state="dropped")

    assert dropped.state == "dropped"
    record = load_activity_records("work", "20260410", include_hidden=True)[0]
    closes = _ledger_close_edits(record)
    assert [edit["ledger_close"]["as_state"] for edit in closes] == [
        "closed",
        "dropped",
    ]
    assert dropped.closed_at == _utc_ms(closes[-1]["timestamp"])


def test_decisions_dedup(tmp_path, monkeypatch):
    from think.surfaces import ledger as ledger_surface

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _minimal_facet_tree(tmp_path)
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        decisions=[_decision()],
    )
    _write_story_activity(
        "work",
        "20260410",
        "meeting_100000_300",
        _utc_ms("2026-04-10T10:00:00Z"),
        decisions=[_decision(context="Later duplicate.")],
    )

    results = ledger_surface.decisions()

    assert len(results) == 1
    assert results[0].created_at == _utc_ms("2026-04-10T09:00:00Z")
    assert results[0].source.activity_id == "meeting_090000_300"


def test_missing_entity_id_pairing(tmp_path, monkeypatch):
    from think.surfaces import ledger as ledger_surface

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _minimal_facet_tree(tmp_path, facets=("work", "personal"))

    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment(owner_entity_id=None)],
    )
    _write_story_activity(
        "personal",
        "20260411",
        "meeting_100000_300",
        _utc_ms("2026-04-11T10:00:00Z"),
        closures=[_closure(owner_entity_id=None, action="sent the proposal")],
    )
    matched = ledger_surface.list(state="closed")
    assert len(matched) == 1

    _write_story_activity(
        "work",
        "20260412",
        "meeting_110000_300",
        _utc_ms("2026-04-12T11:00:00Z"),
        commitments=[_commitment(owner_entity_id=None, action="draft status update")],
    )
    _write_story_activity(
        "personal",
        "20260413",
        "meeting_120000_300",
        _utc_ms("2026-04-13T12:00:00Z"),
        closures=[
            _closure(
                owner_entity_id="mina",
                action="draft status update",
                counterparty_entity_id="ravi",
            )
        ],
    )
    open_items = ledger_surface.list(state="open")
    assert any(item.action == "draft status update" for item in open_items)


def test_missing_counterparty_id_pairing_falls_back_to_text(tmp_path, monkeypatch):
    from think.surfaces import ledger as ledger_surface

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _minimal_facet_tree(tmp_path, facets=("work", "personal"))
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[
            _commitment(
                counterparty="Finance Team",
                counterparty_entity_id=None,
                owner_entity_id=None,
            )
        ],
    )
    _write_story_activity(
        "personal",
        "20260411",
        "meeting_100000_300",
        _utc_ms("2026-04-11T10:00:00Z"),
        closures=[
            _closure(
                action="sent the proposal",
                counterparty="Finance Team",
                counterparty_entity_id=None,
                owner_entity_id=None,
            )
        ],
    )

    items = ledger_surface.list(state="closed")
    assert len(items) == 1


def test_explicit_facets_include_muted_facet(tmp_path, monkeypatch):
    from think.surfaces import ledger as ledger_surface

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _minimal_facet_tree(tmp_path, facets=("work", "quiet"), muted_facets=("quiet",))
    _write_story_activity(
        "quiet",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment(action="muted facet item")],
    )

    assert ledger_surface.list(state="open") == []
    explicit = ledger_surface.list(state="open", facets=["quiet"])
    assert [item.action for item in explicit] == ["muted facet item"]


def test_hidden_record_exclusion(tmp_path, monkeypatch):
    from think.surfaces import ledger as ledger_surface

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _minimal_facet_tree(tmp_path)
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment()],
        hidden=True,
    )

    assert ledger_surface.list(state="all") == []


def test_sort_default_varies_by_state(tmp_path, monkeypatch):
    from think.surfaces import ledger as ledger_surface

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _minimal_facet_tree(tmp_path)
    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        now_ms - (3 * _DAY_MS),
        commitments=[_commitment(action="older open")],
    )
    _write_story_activity(
        "work",
        "20260411",
        "meeting_100000_300",
        now_ms - _DAY_MS,
        commitments=[_commitment(action="newer open")],
    )

    open_items = ledger_surface.list(state="open")
    assert [item.action for item in open_items] == ["older open", "newer open"]

    _write_story_activity(
        "work",
        "20260412",
        "meeting_110000_300",
        now_ms - (10 * _DAY_MS),
        commitments=[_commitment(action="older closed")],
    )
    _write_story_activity(
        "work",
        "20260413",
        "meeting_120000_300",
        now_ms - (5 * _DAY_MS),
        closures=[_closure(action="older closed")],
    )
    _write_story_activity(
        "work",
        "20260414",
        "meeting_130000_300",
        now_ms - (8 * _DAY_MS),
        commitments=[_commitment(action="newer closed")],
    )
    _write_story_activity(
        "work",
        "20260415",
        "meeting_140000_300",
        now_ms - _DAY_MS,
        closures=[_closure(action="newer closed")],
    )

    closed_items = ledger_surface.list(state="closed")
    assert [item.action for item in closed_items] == ["newer closed", "older closed"]


def test_manual_dropped_overrides_earlier_story_closure(tmp_path, monkeypatch):
    from think.activities import load_activity_records
    from think.surfaces import ledger as ledger_surface

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _minimal_facet_tree(tmp_path)
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment()],
    )
    _write_story_activity(
        "work",
        "20260411",
        "meeting_100000_300",
        _utc_ms("2026-04-11T10:00:00Z"),
        closures=[_closure()],
    )

    item = ledger_surface.list(state="closed")[0]
    refreshed = ledger_surface.close(
        item.id, note="operator says drop", as_state="dropped"
    )
    record = load_activity_records("work", "20260410", include_hidden=True)[0]
    manual_edit = _ledger_close_edits(record)[-1]

    assert refreshed.state == "dropped"
    assert refreshed.closed_at == _utc_ms(manual_edit["timestamp"])
    assert refreshed.closed_at != _utc_ms("2026-04-11T10:00:00Z")
    assert any(source.field == "closures" for source in refreshed.sources)
    assert any(
        source.field == "edits" and source.activity_id == "meeting_090000_300"
        for source in refreshed.sources
    )


def test_cli_list_smoke(tmp_path, monkeypatch):
    from think.tools.ledger import app

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    monkeypatch.setenv("SOL_SKIP_SUPERVISOR_CHECK", "1")
    _minimal_facet_tree(tmp_path)
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment()],
    )

    result = _RUNNER.invoke(app, ["list", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload[0]["summary"] == "send proposal"


def test_cli_get_smoke(tmp_path, monkeypatch):
    from think.surfaces import ledger as ledger_surface
    from think.tools.ledger import app

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    monkeypatch.setenv("SOL_SKIP_SUPERVISOR_CHECK", "1")
    _minimal_facet_tree(tmp_path)
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment()],
    )
    item = ledger_surface.list(state="open")[0]

    result = _RUNNER.invoke(app, ["get", item.id, "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload[0]["id"] == item.id


def test_cli_close_smoke(tmp_path, monkeypatch):
    from think.surfaces import ledger as ledger_surface
    from think.tools.ledger import app

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    monkeypatch.setenv("SOL_SKIP_SUPERVISOR_CHECK", "1")
    _minimal_facet_tree(tmp_path)
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment()],
    )
    item = ledger_surface.list(state="open")[0]

    result = _RUNNER.invoke(app, ["close", item.id, "--note", "done", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload[0]["state"] == "closed"


def test_cli_decisions_smoke(tmp_path, monkeypatch):
    from think.tools.ledger import app

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    monkeypatch.setenv("SOL_SKIP_SUPERVISOR_CHECK", "1")
    _minimal_facet_tree(tmp_path)
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        decisions=[_decision()],
    )

    result = _RUNNER.invoke(app, ["decisions", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload[0]["action"] == "move launch review"
