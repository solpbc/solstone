# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
from pathlib import Path


def _write_detected_entities(tmp_path, facet: str, day: str, rows: list[dict]) -> None:
    path = tmp_path / "facets" / facet / "entities" / f"{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _activity_record(record_id: str = "meeting_090000_300") -> dict:
    return {
        "id": record_id,
        "activity": "meeting",
        "description": "Team sync",
        "segments": ["090000_300"],
        "created_at": 1,
    }


def _context(
    tmp_path: Path,
    *,
    facet: str = "work",
    day: str = "20260418",
    record_id: str = "meeting_090000_300",
) -> dict:
    return {
        "facet": facet,
        "day": day,
        "activity": {"id": record_id},
        "output_path": str(
            tmp_path / "facets" / facet / "activities" / day / record_id / "story.json"
        ),
    }


def _valid_result(**overrides) -> str:
    payload = {
        "body": "Aligned on launch work and assigned the follow-up.",
        "topics": ["launch", "follow-up"],
        "confidence": 0.82,
        "commitments": [
            {
                "owner": "Mina",
                "action": "send the revised deck",
                "counterparty": "Ravi",
                "when": "Friday morning",
                "context": "Mina committed to send the deck before the next investor call.",
            }
        ],
        "closures": [
            {
                "owner": "Ravi",
                "action": "intro email",
                "counterparty": "Mina",
                "resolution": "sent",
                "context": "Ravi confirmed the intro email already went out.",
            }
        ],
        "decisions": [
            {
                "owner": "Team",
                "action": "move the launch review to Tuesday",
                "context": "The group aligned on Tuesday after checking calendars.",
            }
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _load_record(facet: str, day: str):
    from think.activities import load_activity_records

    return load_activity_records(facet, day, include_hidden=True)[0]


def test_story_hook_parses_and_writes(tmp_path, monkeypatch):
    from talent.story import post_process
    from think.activities import append_activity_record

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    append_activity_record("work", "20260418", _activity_record())

    returned = post_process(
        _valid_result(body="  Wrapped the launch prep and assigned follow-up.  "),
        _context(tmp_path),
    )

    record = _load_record("work", "20260418")
    assert returned == ""
    assert record["story"] == {
        "body": "Wrapped the launch prep and assigned follow-up.",
        "topics": ["launch", "follow-up"],
        "confidence": 0.82,
    }
    assert record["commitments"][0]["owner"] == "Mina"
    assert record["closures"][0]["resolution"] == "sent"
    assert record["decisions"][0]["owner"] == "Team"
    assert record["edits"][-1]["actor"] == "story"
    assert record["edits"][-1]["fields"] == [
        "story",
        "commitments",
        "closures",
        "decisions",
    ]


def test_story_hook_empty_arrays(tmp_path, monkeypatch):
    from talent.story import post_process
    from think.activities import append_activity_record

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    append_activity_record("work", "20260418", _activity_record())

    post_process(
        _valid_result(commitments=[], closures=[], decisions=[]),
        _context(tmp_path),
    )

    record = _load_record("work", "20260418")
    assert (
        record["story"]["body"] == "Aligned on launch work and assigned the follow-up."
    )
    assert record["commitments"] == []
    assert record["closures"] == []
    assert record["decisions"] == []


def test_story_hook_bad_resolution_skipped(tmp_path, monkeypatch, caplog):
    from talent.story import post_process
    from think.activities import append_activity_record

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    append_activity_record("work", "20260418", _activity_record())

    post_process(
        _valid_result(
            closures=[
                {
                    "owner": "Ravi",
                    "action": "intro email",
                    "counterparty": "Mina",
                    "resolution": "sent",
                    "context": "The intro email went out.",
                },
                {
                    "owner": "Ravi",
                    "action": "budget request",
                    "counterparty": "Finance",
                    "resolution": "approved",
                    "context": "This resolution is invalid for the schema.",
                },
            ]
        ),
        _context(tmp_path),
    )

    record = _load_record("work", "20260418")
    assert [closure["action"] for closure in record["closures"]] == ["intro email"]
    assert "invalid resolution 'approved'" in caplog.text


def test_story_hook_missing_required_field_skipped(tmp_path, monkeypatch, caplog):
    from talent.story import post_process
    from think.activities import append_activity_record

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    append_activity_record("work", "20260418", _activity_record())

    post_process(
        _valid_result(
            commitments=[
                {
                    "owner": "Mina",
                    "action": "send the revised deck",
                    "counterparty": "Ravi",
                    "when": "Friday morning",
                    "context": "Valid commitment.",
                },
                {
                    "owner": "Mina",
                    "action": "book the room",
                    "when": "tomorrow",
                    "context": "Missing counterparty should skip.",
                },
            ],
            closures=[
                {
                    "owner": "Ravi",
                    "action": "intro email",
                    "counterparty": "Mina",
                    "resolution": "sent",
                    "context": "Valid closure.",
                },
                {
                    "action": "parking pass",
                    "counterparty": "Travel desk",
                    "resolution": "done",
                    "context": "Missing owner should skip.",
                },
            ],
            decisions=[
                {
                    "owner": "Team",
                    "action": "move the launch review to Tuesday",
                    "context": "Valid decision.",
                },
                {
                    "owner": "Team",
                    "context": "Missing action should skip.",
                },
            ],
        ),
        _context(tmp_path),
    )

    record = _load_record("work", "20260418")
    assert len(record["commitments"]) == 1
    assert len(record["closures"]) == 1
    assert len(record["decisions"]) == 1
    assert "missing required string field" in caplog.text


def test_story_hook_resolves_entities(tmp_path, monkeypatch):
    from talent.story import post_process
    from think.activities import append_activity_record

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    _write_detected_entities(
        tmp_path,
        "work",
        "20260418",
        [
            {"id": "mina_lee", "type": "Person", "name": "Mina Lee", "aka": ["Mina"]},
            {"id": "ravi_shah", "type": "Person", "name": "Ravi Shah", "aka": ["Ravi"]},
        ],
    )
    append_activity_record("work", "20260418", _activity_record())

    post_process(
        _valid_result(
            commitments=[
                {
                    "owner": "Mina",
                    "action": "send the revised deck",
                    "counterparty": "Ravi",
                    "when": "Friday morning",
                    "context": "Valid commitment.",
                },
                {
                    "owner": "Unknown Owner",
                    "action": "draft the note",
                    "counterparty": "Unknown Counterparty",
                    "when": "later",
                    "context": "Unmatched names should stay null.",
                },
            ],
            closures=[
                {
                    "owner": "Ravi",
                    "action": "intro email",
                    "counterparty": "Mina",
                    "resolution": "sent",
                    "context": "Valid closure.",
                }
            ],
            decisions=[
                {
                    "owner": "Mina Lee",
                    "action": "move the launch review to Tuesday",
                    "context": "Valid decision.",
                }
            ],
        ),
        _context(tmp_path),
    )

    record = _load_record("work", "20260418")
    assert record["commitments"][0]["owner_entity_id"] == "mina_lee"
    assert record["commitments"][0]["counterparty_entity_id"] == "ravi_shah"
    assert record["commitments"][1]["owner_entity_id"] is None
    assert record["commitments"][1]["counterparty_entity_id"] is None
    assert record["closures"][0]["owner_entity_id"] == "ravi_shah"
    assert record["closures"][0]["counterparty_entity_id"] == "mina_lee"
    assert record["decisions"][0]["owner_entity_id"] == "mina_lee"
    assert record["commitments"][0]["owner"] == "Mina"
    assert record["closures"][0]["counterparty"] == "Mina"


def test_story_hook_idempotent_rerun(tmp_path, monkeypatch):
    from talent.story import post_process
    from think.activities import append_activity_record

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    append_activity_record("work", "20260418", _activity_record())

    post_process(_valid_result(), _context(tmp_path))
    first = _load_record("work", "20260418")
    assert len(first["edits"]) == 1

    post_process(
        _valid_result(
            body="Second pass with a clearer summary.",
            topics=["handoff"],
            commitments=[],
            closures=[],
            decisions=[
                {
                    "owner": "Lead",
                    "action": "ship the patch on Wednesday",
                    "context": "The second pass reached a more specific plan.",
                }
            ],
        ),
        _context(tmp_path),
    )

    second = _load_record("work", "20260418")
    assert second["story"] == {
        "body": "Second pass with a clearer summary.",
        "topics": ["handoff"],
        "confidence": 0.82,
    }
    assert second["commitments"] == []
    assert second["closures"] == []
    assert second["decisions"] == [
        {
            "owner": "Lead",
            "action": "ship the patch on Wednesday",
            "context": "The second pass reached a more specific plan.",
            "owner_entity_id": None,
        }
    ]
    assert len(second["edits"]) == 2


def test_story_hook_missing_record_logs_and_returns(tmp_path, monkeypatch, caplog):
    from talent.story import post_process

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    returned = post_process(_valid_result(), _context(tmp_path))

    assert returned == ""
    assert "activity record not found" in caplog.text


def test_story_hook_no_json_file_written(tmp_path, monkeypatch):
    from talent.story import post_process
    from think.activities import append_activity_record

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    append_activity_record("work", "20260418", _activity_record())

    output_path = (
        tmp_path
        / "facets"
        / "work"
        / "activities"
        / "20260418"
        / "meeting_090000_300"
        / "story.json"
    )
    returned = post_process(_valid_result(), _context(tmp_path))

    assert returned == ""
    assert not output_path.exists()
