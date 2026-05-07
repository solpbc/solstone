# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from solstone.think.talent import get_talent

SCHEMA_PATH = (
    Path(__file__).parent.parent
    / "solstone"
    / "talent"
    / "speaker_attribution.schema.json"
)


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_speaker_attribution_schema_file_is_valid_draft_2020_12():
    Draft202012Validator.check_schema(_load_schema())


def test_speaker_attribution_talent_loads_schema():
    assert get_talent("speaker_attribution")["json_schema"] == _load_schema()


@pytest.mark.parametrize(
    "payload",
    [
        [{"sentence_id": 1, "speaker": "Alice", "reasoning": "Introduced herself."}],
        [
            {"sentence_id": 1, "speaker": "Alice", "reasoning": "Introduced herself."},
            {"sentence_id": 2, "speaker": "Bob", "reasoning": "Replied to Alice."},
        ],
    ],
)
def test_positive_payload_validates(payload):
    validator = Draft202012Validator(_load_schema())

    assert validator.is_valid(payload)


def test_negative_wrapper_object_rejected():
    validator = Draft202012Validator(_load_schema())

    assert not validator.is_valid(
        {
            "attributions": [
                {
                    "sentence_id": 1,
                    "speaker": "Alice",
                    "reasoning": "Introduced herself.",
                }
            ]
        }
    )


def test_negative_missing_required_field_rejected():
    validator = Draft202012Validator(_load_schema())

    assert not validator.is_valid([{"sentence_id": 1, "speaker": "Alice"}])


def test_negative_empty_string_fields_rejected():
    validator = Draft202012Validator(_load_schema())

    assert not validator.is_valid([{"sentence_id": 1, "speaker": "", "reasoning": "x"}])


def test_negative_non_integer_sentence_id_rejected():
    validator = Draft202012Validator(_load_schema())

    assert not validator.is_valid(
        [{"sentence_id": "1", "speaker": "Alice", "reasoning": "x"}]
    )


def test_negative_additional_properties_rejected():
    validator = Draft202012Validator(_load_schema())

    assert not validator.is_valid(
        [
            {
                "sentence_id": 1,
                "speaker": "Alice",
                "reasoning": "x",
                "confidence": "high",
            }
        ]
    )
