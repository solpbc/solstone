# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
import logging
from pathlib import Path

import frontmatter

SENSE_PATH = Path(__file__).resolve().parents[1] / "talent" / "sense.md"


def _role_section() -> str:
    content = frontmatter.load(SENSE_PATH).content
    start = content.index("#### role")
    end = content.index("#### source", start)
    return content[start:end]


def test_sense_role_section_contains_contamination_guard():
    role_section = _role_section()

    assert "tool or product names visible on screen" in role_section
    assert "`source: screen`" in role_section
    assert "`role: mentioned`" in role_section
    assert "Google Meet" in role_section
    assert "Zoom" in role_section


def test_sense_role_section_has_screen_and_mentioned_guidance_for_tools_and_apps():
    role_section = _role_section()

    assert "screen" in role_section
    assert "mentioned" in role_section
    assert "tool" in role_section
    assert "Video-conference app names" in role_section


def test_sense_role_section_gates_attendee_on_meeting_detected():
    role_section = _role_section()

    assert "`meeting_detected: true`" in role_section
    assert "`meeting_detected: false`" in role_section
    assert "`role: attendee`" in role_section


def _write_detected_entities(tmp_path, facet: str, day: str, rows: list[dict]) -> None:
    entities_path = tmp_path / "facets" / facet / "entities" / f"{day}.jsonl"
    entities_path.parent.mkdir(parents=True, exist_ok=True)
    entities_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_sense_json(
    tmp_path,
    day: str,
    stream: str,
    segment_key: str,
    payload: dict | None,
) -> None:
    talents_dir = tmp_path / "chronicle" / day / stream / segment_key / "talents"
    talents_dir.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        (talents_dir / "sense.json").write_text(json.dumps(payload), encoding="utf-8")


def _sense_payload(*, meeting_detected: bool) -> dict:
    return {
        "density": "idle",
        "content_type": "idle",
        "activity_summary": "Idle segment.",
        "entities": [],
        "facets": [],
        "meeting_detected": meeting_detected,
        "speakers": [],
        "recommend": {
            "screen_record": False,
            "speaker_attribution": False,
            "pulse_update": False,
        },
        "emotional_register": "neutral",
    }


def _activity_record(segments: list[str]) -> dict:
    return {
        "id": "meeting_090000_300",
        "activity": "meeting",
        "segments": segments,
        "level_avg": 1.0,
        "description": "Team sync",
        "active_entities": ["Guest Speaker"],
        "created_at": 1,
    }


def _participation_result(role: str) -> str:
    return json.dumps(
        {
            "participation": [
                {
                    "name": "Guest Speaker",
                    "role": role,
                    "source": "voice",
                    "confidence": 0.98,
                    "context": "Spoke during the session",
                    "entity_id": None,
                }
            ]
        }
    )


def test_participation_clamps_attendees_when_all_segments_are_non_meetings(
    tmp_path, monkeypatch, caplog
):
    from talent.participation import post_process
    from think.activities import append_activity_record, load_activity_records

    facet = "work"
    day = "20260418"
    stream = "default"
    segments = ["090000_300", "090500_300"]
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    _write_detected_entities(
        tmp_path,
        facet,
        day,
        [{"id": "guest_speaker", "type": "Person", "name": "Guest Speaker"}],
    )
    for segment_key in segments:
        _write_sense_json(
            tmp_path, day, stream, segment_key, _sense_payload(meeting_detected=False)
        )

    activity = _activity_record(segments)
    append_activity_record(facet, day, activity)

    with caplog.at_level(logging.WARNING, logger="talent.participation"):
        post_process(
            _participation_result("attendee"),
            {"activity": activity, "facet": facet, "day": day},
        )

    record = load_activity_records(facet, day)[0]
    assert record["participation"][0]["role"] == "mentioned"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert (
        warnings[0].getMessage()
        == "participation hook: clamped 1 attendee entries to mentioned on activity meeting_090000_300 (facet=work day=20260418); no contributing sense segment had meeting_detected=true"
    )


def test_participation_preserves_attendees_when_any_segment_is_meeting(
    tmp_path, monkeypatch, caplog
):
    from talent.participation import post_process
    from think.activities import append_activity_record, load_activity_records

    facet = "work"
    day = "20260418"
    stream = "default"
    segments = ["090000_300", "090500_300"]
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    _write_detected_entities(
        tmp_path,
        facet,
        day,
        [{"id": "guest_speaker", "type": "Person", "name": "Guest Speaker"}],
    )
    _write_sense_json(
        tmp_path, day, stream, segments[0], _sense_payload(meeting_detected=False)
    )
    _write_sense_json(
        tmp_path, day, stream, segments[1], _sense_payload(meeting_detected=True)
    )

    activity = _activity_record(segments)
    append_activity_record(facet, day, activity)

    with caplog.at_level(logging.WARNING, logger="talent.participation"):
        post_process(
            _participation_result("attendee"),
            {"activity": activity, "facet": facet, "day": day},
        )

    record = load_activity_records(facet, day)[0]
    assert record["participation"][0]["role"] == "attendee"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == []


def test_participation_clamp_is_idempotent_on_second_pass(
    tmp_path, monkeypatch, caplog
):
    from talent.participation import post_process
    from think.activities import append_activity_record, load_activity_records

    facet = "work"
    day = "20260418"
    stream = "default"
    segments = ["090000_300", "090500_300"]
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    _write_detected_entities(
        tmp_path,
        facet,
        day,
        [{"id": "guest_speaker", "type": "Person", "name": "Guest Speaker"}],
    )
    for segment_key in segments:
        _write_sense_json(
            tmp_path, day, stream, segment_key, _sense_payload(meeting_detected=False)
        )

    activity = _activity_record(segments)
    append_activity_record(facet, day, activity)

    with caplog.at_level(logging.WARNING, logger="talent.participation"):
        post_process(
            _participation_result("attendee"),
            {"activity": activity, "facet": facet, "day": day},
        )
    first_warning_count = len(
        [r for r in caplog.records if r.levelno == logging.WARNING]
    )
    assert first_warning_count == 1

    record = load_activity_records(facet, day)[0]
    second_result = json.dumps({"participation": record["participation"]})

    with caplog.at_level(logging.WARNING, logger="talent.participation"):
        post_process(
            second_result,
            {"activity": record, "facet": facet, "day": day},
        )

    updated = load_activity_records(facet, day)[0]
    assert updated["participation"] == record["participation"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_participation_treats_missing_sense_json_as_non_meeting(
    tmp_path, monkeypatch, caplog
):
    from talent.participation import post_process
    from think.activities import append_activity_record, load_activity_records

    facet = "work"
    day = "20260418"
    stream = "default"
    segments = ["090000_300", "090500_300"]
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    _write_detected_entities(
        tmp_path,
        facet,
        day,
        [{"id": "guest_speaker", "type": "Person", "name": "Guest Speaker"}],
    )
    _write_sense_json(tmp_path, day, stream, segments[0], None)
    _write_sense_json(
        tmp_path, day, stream, segments[1], _sense_payload(meeting_detected=False)
    )

    activity = _activity_record(segments)
    append_activity_record(facet, day, activity)

    with caplog.at_level(logging.WARNING, logger="talent.participation"):
        post_process(
            _participation_result("attendee"),
            {"activity": activity, "facet": facet, "day": day},
        )

    record = load_activity_records(facet, day)[0]
    assert record["participation"][0]["role"] == "mentioned"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
