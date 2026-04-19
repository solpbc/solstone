# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the storytelling spans post-hook."""

from __future__ import annotations

import json
from pathlib import Path

from talent.spans import post_process


def _activity(
    *,
    activity_id: str = "coding_100000_300",
    activity_type: str = "coding",
    segments: list[str] | None = None,
) -> dict:
    return {
        "id": activity_id,
        "activity": activity_type,
        "segments": segments or ["100000_300", "100500_300"],
    }


def _context(
    *,
    name: str = "work",
    facet: str = "work",
    day: str = "20260418",
    activity: dict | None = None,
) -> dict:
    return {
        "name": name,
        "facet": facet,
        "day": day,
        "activity": activity or _activity(),
    }


def _rows(tmp_path: Path, *, facet: str = "work", day: str = "20260418") -> list[dict]:
    path = tmp_path / "facets" / facet / "spans" / f"{day}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_post_process_writes_all_fields_and_renders_coding_span(monkeypatch, tmp_path):
    from think.spans import format_spans

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    result = json.dumps(
        {
            "body": "Implemented the retry path and verified the failing case.",
            "topics": ["Retry Logic", "Testing", "retry logic", "  Testing  "],
            "confidence": 0.82,
        }
    )

    returned = post_process(result, _context())

    assert returned == ""

    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert rows[0] == {
        "span_id": "coding_100000_300",
        "talent": "work",
        "facet": "work",
        "day": "20260418",
        "activity_type": "coding",
        "start": "10:00:00",
        "end": "10:10:00",
        "body": "Implemented the retry path and verified the failing case.",
        "topics": ["retry logic", "testing"],
        "confidence": 0.82,
    }

    file_path = tmp_path / "facets" / "work" / "spans" / "20260418.jsonl"
    chunks, meta = format_spans(rows, {"file_path": file_path})
    assert len(chunks) == 1
    assert chunks[0]["source"] == rows[0]
    assert "### Coding: coding_100000_300" in chunks[0]["markdown"]
    assert meta["indexer"] == {"agent": "span"}


def test_post_process_writes_single_conversation_row_for_meeting(monkeypatch, tmp_path):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    result = json.dumps(
        {
            "body": 'Aligned on next steps and confirmed "ship it Friday".',
            "topics": ["planning", "alignment", "delivery"],
            "confidence": 0.91,
        }
    )
    ctx = _context(
        name="conversation",
        activity=_activity(
            activity_id="meeting_090000_300",
            activity_type="meeting",
            segments=["090000_300", "091500_300"],
        ),
    )

    returned = post_process(result, ctx)

    assert returned == ""
    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["talent"] == "conversation"
    assert rows[0]["activity_type"] == "meeting"
    assert rows[0]["start"] == "09:00:00"
    assert rows[0]["end"] == "09:20:00"
    assert not (
        tmp_path
        / "facets"
        / "work"
        / "activities"
        / "20260418"
        / "meeting_090000_300"
        / "conversation.json"
    ).exists()


def test_post_process_clamps_confidence_and_logs(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    returned = post_process(
        json.dumps(
            {
                "body": "Shipped the fix.",
                "topics": ["release", "shipping", "qa"],
                "confidence": 1.4,
            }
        ),
        _context(),
    )

    assert returned == ""
    assert _rows(tmp_path)[0]["confidence"] == 1.0
    assert "clamped confidence" in caplog.text


def test_post_process_rejects_bad_confidence(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    returned = post_process(
        json.dumps(
            {
                "body": "Investigated the issue.",
                "topics": ["debugging", "logs", "triage"],
                "confidence": "high",
            }
        ),
        _context(),
    )

    assert returned == ""
    assert _rows(tmp_path) == []
    assert "invalid confidence" in caplog.text

    caplog.clear()
    returned = post_process(
        json.dumps(
            {
                "body": "Investigated the issue.",
                "topics": ["debugging", "logs", "triage"],
                "confidence": float("nan"),
            }
        ),
        _context(),
    )

    assert returned == ""
    assert _rows(tmp_path) == []
    assert "invalid confidence" in caplog.text


def test_post_process_rejects_missing_or_empty_topics(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    missing_topics = json.dumps({"body": "Worked through the task.", "confidence": 0.7})
    returned = post_process(missing_topics, _context())
    assert returned == ""
    assert _rows(tmp_path) == []
    assert "missing topics" in caplog.text

    caplog.clear()
    empty_topics = json.dumps(
        {"body": "Worked through the task.", "topics": [" ", "\t"], "confidence": 0.7}
    )
    returned = post_process(empty_topics, _context())
    assert returned == ""
    assert _rows(tmp_path) == []
    assert "empty topics" in caplog.text

    caplog.clear()
    invalid_topics = json.dumps(
        {
            "body": "Worked through the task.",
            "topics": ["valid", 7, "other"],
            "confidence": 0.7,
        }
    )
    returned = post_process(invalid_topics, _context())
    assert returned == ""
    assert _rows(tmp_path) == []
    assert "invalid topics" in caplog.text


def test_post_process_replaces_existing_row_by_span_and_talent(monkeypatch, tmp_path):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    ctx = _context()

    first = json.dumps(
        {"body": "First pass.", "topics": ["alpha", "beta", "gamma"], "confidence": 0.5}
    )
    second = json.dumps(
        {
            "body": "Second pass.",
            "topics": ["delta", "epsilon", "zeta"],
            "confidence": 0.9,
        }
    )

    assert post_process(first, ctx) == ""
    assert post_process(second, ctx) == ""

    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["body"] == "Second pass."
    assert rows[0]["topics"] == ["delta", "epsilon", "zeta"]
    assert rows[0]["confidence"] == 0.9


def test_post_process_appends_distinct_talent_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    activity = _activity(activity_id="event_130000_300", activity_type="event")

    event_ctx = _context(name="event", activity=activity)
    conversation_ctx = _context(name="conversation", activity=activity)

    assert (
        post_process(
            json.dumps(
                {
                    "body": "Wrapped the event.",
                    "topics": ["planning", "venue", "timeline"],
                    "confidence": 0.66,
                }
            ),
            event_ctx,
        )
        == ""
    )
    assert (
        post_process(
            json.dumps(
                {
                    "body": "Captured the side conversation.",
                    "topics": ["alignment", "follow-up", "owners"],
                    "confidence": 0.72,
                }
            ),
            conversation_ctx,
        )
        == ""
    )

    rows = _rows(tmp_path)
    assert len(rows) == 2
    assert {(row["span_id"], row["talent"]) for row in rows} == {
        ("event_130000_300", "event"),
        ("event_130000_300", "conversation"),
    }


def test_post_process_handles_parse_failures_and_fenced_json(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    assert post_process("{not-json", _context()) == ""
    assert _rows(tmp_path) == []
    assert "failed to parse JSON" in caplog.text

    caplog.clear()
    fenced = """```json
{"body":"Recovered.","topics":["alpha","beta","gamma"],"confidence":0.6}
```"""
    assert post_process(fenced, _context()) == ""
    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["body"] == "Recovered."
