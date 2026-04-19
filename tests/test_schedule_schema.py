# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from think.talent import get_talent

TALENT_DIR = Path(__file__).resolve().parents[1] / "talent"
PARTICIPATION_ENTRY_SCHEMA_PATH = TALENT_DIR / "participation_entry.schema.json"
SCHEDULE_SCHEMA_PATH = TALENT_DIR / "schedule.schema.json"

SCHEDULE_ACTIVITY_ENUM = {
    "meeting",
    "call",
    "deadline",
    "appointment",
    "event",
    "travel",
    "reminder",
    "errand",
    "celebration",
    "doctor_appointment",
}

SCHEDULE_REQUIRED_FIELDS = {
    "activity",
    "target_date",
    "start",
    "end",
    "title",
    "description",
    "details",
    "participation",
    "participation_confidence",
    "facet",
    "cancelled",
}


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_schedule_schema() -> dict:
    schema = _load_json(SCHEDULE_SCHEMA_PATH)
    assert isinstance(schema, dict)
    return schema


def _sample_schedule_payloads() -> list[list[dict]]:
    return [
        [
            {
                "activity": "meeting",
                "target_date": "2026-04-20",
                "start": "16:30:00",
                "end": "17:30:00",
                "title": "Yuri Namikawa intro call",
                "description": "Intro call with Yuri from Offline Ventures.",
                "details": "Google Meet",
                "participation": [
                    {
                        "name": "Yuri Namikawa",
                        "role": "attendee",
                        "source": "screen",
                        "confidence": 0.95,
                        "context": "calendar invite",
                    },
                    {
                        "name": "Scott Ward",
                        "role": "mentioned",
                        "source": "screen",
                        "confidence": 0.5,
                        "context": "mentioned in notes",
                    },
                    {
                        "name": "Unknown Guest",
                        "role": "attendee",
                        "source": "screen",
                        "confidence": 0.4,
                        "context": "guest field",
                    },
                ],
                "participation_confidence": 0.88,
                "facet": "work",
                "cancelled": False,
            }
        ],
        [
            {
                "activity": "meeting",
                "target_date": "2026-04-22",
                "start": "09:00:00",
                "end": "10:00:00",
                "title": "Scott Ward standup",
                "description": "Weekly standup with Scott Ward.",
                "details": "Recurring invite",
                "participation": [],
                "participation_confidence": 0.85,
                "facet": "work",
                "cancelled": True,
            }
        ],
        [
            {
                "activity": "call",
                "target_date": "2026-04-21",
                "start": "10:30:00",
                "end": "11:00:00",
                "title": "Mari Zumbro intro",
                "description": "Updated invite",
                "details": "Google Meet",
                "participation": [],
                "participation_confidence": 0.9,
                "facet": "work",
                "cancelled": False,
            }
        ],
        [
            {
                "activity": "deadline",
                "target_date": "2026-05-05",
                "start": None,
                "end": None,
                "title": "Demo Day",
                "description": "Betaworks Camp Demo Day.",
                "details": "Live demo presentation to cohort investors",
                "participation": [],
                "participation_confidence": 0.5,
                "facet": "work",
                "cancelled": False,
            }
        ],
    ]


def test_schedule_schema_file_is_valid_draft_2020_12():
    Draft202012Validator.check_schema(_load_schedule_schema())


def test_schedule_talent_loads_schema():
    assert get_talent("schedule")["json_schema"] == _load_schedule_schema()


def test_schedule_participation_entry_diverges_from_shared_fragment():
    """Schedule omits entity_id because the hook fills it via find_matching_entity."""
    schedule_schema = _load_schedule_schema()
    fragment = _load_json(PARTICIPATION_ENTRY_SCHEMA_PATH)

    assert isinstance(fragment, dict)
    fragment_without_schema = dict(fragment)
    fragment_without_schema.pop("$schema")
    fragment_without_schema["properties"] = dict(fragment_without_schema["properties"])
    fragment_without_schema["properties"].pop("entity_id")
    fragment_without_schema["required"] = [
        key for key in fragment_without_schema["required"] if key != "entity_id"
    ]

    inline_items = dict(
        schedule_schema["items"]["properties"]["participation"]["items"]
    )

    assert inline_items == fragment_without_schema


def test_schedule_schema_mirrors_hook_requirements():
    schedule_schema = _load_schedule_schema()
    item_schema = schedule_schema["items"]
    properties = item_schema["properties"]
    participation_items = properties["participation"]["items"]
    fragment = _load_json(PARTICIPATION_ENTRY_SCHEMA_PATH)

    assert schedule_schema["type"] == "array"
    assert set(item_schema["required"]) == SCHEDULE_REQUIRED_FIELDS
    assert set(properties["activity"]["enum"]) == SCHEDULE_ACTIVITY_ENUM
    assert (
        participation_items["properties"]["role"]["enum"]
        == fragment["properties"]["role"]["enum"]
    )
    assert (
        participation_items["properties"]["source"]["enum"]
        == fragment["properties"]["source"]["enum"]
    )
    assert properties["start"]["type"] == ["string", "null"]
    assert properties["end"]["type"] == ["string", "null"]
    assert properties["cancelled"]["type"] == "boolean"


def test_schedule_hook_fixtures_validate_against_schema():
    validator = Draft202012Validator(_load_schedule_schema())

    for payload in _sample_schedule_payloads():
        assert list(validator.iter_errors(payload)) == []
