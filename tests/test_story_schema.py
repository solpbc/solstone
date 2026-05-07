# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from solstone.talent.story import ALLOWED_RESOLUTIONS
from solstone.think.talent import get_talent
from tests.test_story_hook import _valid_result

STORY_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "solstone" / "talent" / "story.schema.json"
)


def _load_story_schema() -> dict:
    return json.loads(STORY_SCHEMA_PATH.read_text(encoding="utf-8"))


def test_story_schema_file_is_valid_draft_2020_12():
    Draft202012Validator.check_schema(_load_story_schema())


def test_story_talents_load_shared_schema():
    on_disk = _load_story_schema()

    for talent_name in ["conversation", "work", "event"]:
        assert get_talent(talent_name)["json_schema"] == on_disk


def test_story_schema_mirrors_hook_requirements():
    schema = _load_story_schema()
    properties = schema["properties"]

    assert set(schema["required"]) == {
        "body",
        "topics",
        "confidence",
        "commitments",
        "closures",
        "decisions",
    }
    assert set(properties["commitments"]["items"]["required"]) == {
        "owner",
        "action",
        "counterparty",
        "when",
        "context",
    }
    assert set(properties["closures"]["items"]["required"]) == {
        "owner",
        "action",
        "counterparty",
        "resolution",
        "context",
    }
    assert (
        set(properties["closures"]["items"]["properties"]["resolution"]["enum"])
        == ALLOWED_RESOLUTIONS
    )
    assert set(properties["decisions"]["items"]["required"]) == {
        "owner",
        "action",
        "context",
    }


def test_story_hook_fixtures_validate_against_schema():
    schema = _load_story_schema()
    payload = json.loads(_valid_result())

    errors = list(Draft202012Validator(schema).iter_errors(payload))

    assert errors == []
