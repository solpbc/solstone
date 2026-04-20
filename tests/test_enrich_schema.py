# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

enrich_mod = importlib.import_module("observe.enrich")

_SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "observe" / "enrich.schema.json").read_text(
        encoding="utf-8"
    )
)


def test_enrich_schema_file_is_valid_draft_2020_12():
    Draft202012Validator.check_schema(_SCHEMA)


def test_enrich_schema_accepts_and_rejects_expected_values():
    validator = Draft202012Validator(_SCHEMA)

    assert validator.is_valid(
        {
            "statements": [{"corrected": "Hello world.", "emotion": "calm"}],
            "topics": "",
            "setting": "",
            "warning": "",
        }
    )
    assert validator.is_valid(
        {
            "statements": [],
            "topics": "planning, testing",
            "setting": "work",
            "warning": "",
        }
    )
    assert validator.is_valid(
        {
            "statements": [],
            "topics": "",
            "setting": "",
            "warning": "",
        }
    )
    assert not validator.is_valid([{"corrected": "x", "emotion": "y"}])
    assert not validator.is_valid(
        {
            "statements": [{"corrected": "Hello world.", "emotion": "calm"}],
            "setting": "",
            "warning": "",
        }
    )
    assert not validator.is_valid(
        {
            "statements": [{"corrected": "Hello world.", "emotion": "calm"}],
            "topics": "",
            "warning": "",
        }
    )
    assert not validator.is_valid(
        {
            "statements": [{"corrected": "Hello world.", "emotion": "calm"}],
            "topics": "",
            "setting": "",
        }
    )
    assert not validator.is_valid(
        {
            "topics": "",
            "setting": "",
            "warning": "",
        }
    )
    assert not validator.is_valid(
        {
            "statements": [{"corrected": "Hello world.", "emotion": "calm"}],
            "topics": "",
            "setting": "",
            "warning": "",
            "extra": "nope",
        }
    )
    assert not validator.is_valid(
        {
            "statements": [{"emotion": "calm"}],
            "topics": "",
            "setting": "",
            "warning": "",
        }
    )
    assert not validator.is_valid(
        {
            "statements": [{"corrected": "Hello world."}],
            "topics": "",
            "setting": "",
            "warning": "",
        }
    )
    assert not validator.is_valid(
        {
            "statements": [
                {
                    "corrected": "Hello world.",
                    "emotion": "calm",
                    "extra": "nope",
                }
            ],
            "topics": "",
            "setting": "",
            "warning": "",
        }
    )


def test_enrich_transcript_passes_schema_to_generate(monkeypatch):
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "statements": [{"corrected": "Hello world.", "emotion": "neutral"}],
                "topics": "",
                "setting": "",
                "warning": "",
            }
        )

    monkeypatch.setattr(enrich_mod, "generate", fake_generate)

    import numpy as np

    result = enrich_mod.enrich_transcript(
        np.zeros(16000, dtype=np.float32),
        16000,
        [{"id": 1, "start": 0.0, "end": 1.0, "text": "Hello world."}],
    )

    assert captured["json_schema"] is enrich_mod._SCHEMA
    assert result == {
        "statements": [{"corrected": "Hello world.", "emotion": "neutral"}],
        "topics": "",
        "setting": "",
        "warning": "",
    }
