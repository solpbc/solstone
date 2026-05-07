# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from solstone.think.talent import get_talent

DAILY_SCHEDULE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "solstone"
    / "talent"
    / "daily_schedule.schema.json"
)


def _load_daily_schedule_schema() -> dict:
    return json.loads(DAILY_SCHEDULE_SCHEMA_PATH.read_text(encoding="utf-8"))


def test_daily_schedule_schema_file_is_valid_draft_2020_12():
    Draft202012Validator.check_schema(_load_daily_schedule_schema())


def test_daily_schedule_loaded_json_schema_matches_on_disk_schema():
    assert get_talent("daily_schedule")["json_schema"] == _load_daily_schedule_schema()


def test_daily_schedule_pattern_accepts_and_rejects_expected_values():
    schema = _load_daily_schedule_schema()
    validator = Draft202012Validator(schema)

    for value in ["00:00", "03:00", "09:00", "23:59"]:
        assert validator.is_valid({"primary": value, "fallback": "00:00"})

    for value in ["9:00", "03:00:00", "24:00", "03:60", "", "ab:cd"]:
        assert not validator.is_valid({"primary": value, "fallback": "00:00"})
