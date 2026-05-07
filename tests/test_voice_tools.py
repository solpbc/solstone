# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path

from solstone.think.indexer.journal import scan_journal
from solstone.think.voice import tools
from solstone.think.voice.observer_queue import get_observer_queue
from tests.test_surfaces_ledger import (
    _commitment,
    _minimal_facet_tree,
    _utc_ms,
    _write_story_activity,
)


def _set_today(monkeypatch, day_value: date) -> None:
    monkeypatch.setattr(tools, "_today", lambda: day_value)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_journal_get_day_happy(monkeypatch, journal_copy):
    summary_path = (
        journal_copy
        / "chronicle"
        / "20260304"
        / "default"
        / "090000_300"
        / "voice_summary.md"
    )
    summary_path.write_text("Morning journal summary", encoding="utf-8")

    result = tools.handle_journal_get_day({"day": "2026-03-04"}, object())

    assert result["day"] == "2026-03-04"
    assert result["_nav_target"] == "today/journal/2026-03-04"
    assert result["segments"]
    assert any(
        segment["summary"] == "Morning journal summary"
        for segment in result["segments"]
    )


def test_journal_get_day_failure():
    assert tools.handle_journal_get_day({"day": "bad"}, object()) == {
        "error": "invalid day"
    }


def test_journal_search_happy(monkeypatch, journal_copy):
    scan_journal(str(journal_copy.resolve()), full=True)

    result = tools.handle_journal_search({"query": "prototype", "limit": 3}, object())

    assert result["count"] >= 1
    assert result["_nav_target"] == "today/search?q=prototype"
    assert result["results"][0]["snippet"]


def test_journal_search_failure():
    assert tools.handle_journal_search({"query": " "}, object()) == {
        "error": "query is required"
    }


def test_entities_get_happy():
    result = tools.handle_entities_get({"entity_slug": "romeo_montague"}, object())

    assert result["slug"] == "romeo_montague"
    assert result["_nav_target"] == "entity/romeo_montague"
    assert result["name"]


def test_entities_get_failure():
    assert tools.handle_entities_get({"entity_slug": "missing_slug"}, object()) == {
        "error": "not found"
    }


def test_entities_recent_with_happy(monkeypatch, journal_copy):
    _set_today(monkeypatch, date(2026, 3, 27))
    activity_path = (
        journal_copy / "facets" / "montague" / "activities" / "20260327.jsonl"
    )
    _write_jsonl(
        activity_path,
        [
            {
                "id": "meeting_090000_300",
                "activity": "meeting",
                "title": "Founder sync",
                "description": "Planning session",
                "details": "Bring roadmap notes",
                "source": "user",
                "participation": [
                    {
                        "name": "Romeo Montague",
                        "role": "attendee",
                        "source": "user",
                        "confidence": 1.0,
                        "context": "Confirmed in the room",
                        "entity_id": "romeo_montague",
                    }
                ],
                "created_at": _utc_ms("2026-03-27T09:00:00Z"),
            }
        ],
    )

    result = tools.handle_entities_recent_with(
        {"entity_slug": "romeo_montague", "days": 7}, object()
    )

    assert result["slug"] == "romeo_montague"
    assert result["count"] == 1
    assert result["interactions"][0]["activity"] == "Founder sync"


def test_entities_recent_with_failure():
    assert tools.handle_entities_recent_with(
        {"entity_slug": "missing_slug"}, object()
    ) == {"error": "not found"}


def test_commitments_list_happy(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    _minimal_facet_tree(tmp_path)
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment()],
    )

    result = tools.handle_commitments_list({}, object())

    assert result["commitments"]
    assert result["commitments"][0]["state"] == "open"
    assert "sources" not in result["commitments"][0]


def test_commitments_list_failure():
    assert tools.handle_commitments_list({"state": "bad"}, object()) == {
        "error": "invalid state"
    }


def test_commitments_complete_happy(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    _minimal_facet_tree(tmp_path)
    _write_story_activity(
        "work",
        "20260410",
        "meeting_090000_300",
        _utc_ms("2026-04-10T09:00:00Z"),
        commitments=[_commitment()],
    )
    open_item = tools.handle_commitments_list({}, object())["commitments"][0]

    result = tools.handle_commitments_complete(
        {"commitment_id": open_item["id"], "resolution": "done"},
        object(),
    )

    assert result["ok"] is True
    assert result["commitment"]["resolution"] == "done"
    assert result["commitment"]["state"] == "closed"


def test_commitments_complete_failure():
    assert tools.handle_commitments_complete(
        {"commitment_id": "lg_missing", "resolution": "bad"}, object()
    ) == {"error": "invalid resolution"}


def test_calendar_today_happy(monkeypatch, journal_copy):
    _set_today(monkeypatch, date(2026, 3, 27))
    activity_path = (
        journal_copy / "facets" / "montague" / "activities" / "20260327.jsonl"
    )
    _write_jsonl(
        activity_path,
        [
            {
                "id": "anticipated_meeting_090000",
                "activity": "meeting",
                "title": "Launch sync",
                "source": "anticipated",
                "start": "09:00",
                "location": "Room A",
                "prep_notes": "Bring launch notes",
                "participation": [
                    {
                        "name": "Juliet Capulet",
                        "role": "attendee",
                        "source": "user",
                        "confidence": 1.0,
                        "context": "",
                    }
                ],
                "created_at": _utc_ms("2026-03-27T09:00:00Z"),
            }
        ],
    )

    result = tools.handle_calendar_today({}, object())

    assert result["date"] == "2026-03-27"
    assert result["_nav_target"] == "today"
    assert result["events"][0]["title"] == "Launch sync"


def test_calendar_today_failure(monkeypatch):
    monkeypatch.setattr(
        tools,
        "load_activity_records",
        lambda facet, day: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert tools.handle_calendar_today({}, object()) == {"error": "today unavailable"}


def test_briefing_get_happy(monkeypatch):
    _set_today(monkeypatch, date(2026, 3, 27))

    result = tools.handle_briefing_get({}, object())

    assert result["date"] == "2026-03-27"
    assert result["facet"] == "identity"
    assert result["_nav_target"] == "today"
    assert result["highlights"]


def test_briefing_get_failure(monkeypatch):
    _set_today(monkeypatch, date(2026, 3, 28))
    assert tools.handle_briefing_get({}, object()) == {"error": "no briefing today yet"}


def test_observer_start_listening_happy():
    assert tools.handle_observer_start_listening({"mode": "meeting"}, object()) == {
        "status": "requested",
        "mode": "meeting",
        "note": "sol will start listening shortly",
        "_observer_action": {"type": "start_observer", "mode": "meeting"},
    }


def test_observer_start_listening_failure():
    assert tools.handle_observer_start_listening({"mode": "bad"}, object()) == {
        "error": "invalid mode"
    }


def test_dispatch_tool_call_strips_nav_target(monkeypatch):
    queue = tools.get_nav_queue()
    queue.clear()
    result = asyncio.run(
        tools.dispatch_tool_call(
            "observer.start_listening",
            '{"mode":"meeting"}',
            "call-123",
            object(),
        )
    )
    assert json.loads(result)["status"] == "requested"
    assert queue.drain("call-123") == []

    stripped = asyncio.run(
        tools.dispatch_tool_call(
            "journal.get_day",
            '{"day":"2026-03-04"}',
            "call-123",
            object(),
        )
    )
    payload = json.loads(stripped)
    assert "_nav_target" not in payload
    assert queue.drain("call-123") == ["today/journal/2026-03-04"]


def test_dispatch_tool_call_strips_observer_action():
    queue = get_observer_queue()
    queue.clear()

    result = asyncio.run(
        tools.dispatch_tool_call(
            "observer.start_listening",
            json.dumps({"mode": "meeting"}),
            "call-obs-1",
            object(),
        )
    )

    assert json.loads(result) == {
        "status": "requested",
        "mode": "meeting",
        "note": "sol will start listening shortly",
    }
    assert queue.drain("call-obs-1") == [{"type": "start_observer", "mode": "meeting"}]
