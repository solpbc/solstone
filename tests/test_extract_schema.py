# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

import think.models as models

extract_mod = importlib.import_module("observe.extract")

_SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "observe" / "extract.schema.json").read_text(
        encoding="utf-8"
    )
)


def test_extract_schema_file_is_valid_draft_2020_12():
    Draft202012Validator.check_schema(_SCHEMA)


def test_extract_schema_accepts_and_rejects_expected_values():
    validator = Draft202012Validator(_SCHEMA)

    assert validator.is_valid([])
    assert validator.is_valid([1, 15, 42, 89])
    assert validator.is_valid([1, 0])
    assert not validator.is_valid(["1"])
    assert not validator.is_valid([-1])
    assert not validator.is_valid([1.5])
    assert not validator.is_valid(42)
    assert not validator.is_valid({"ids": [1]})
    assert not validator.is_valid([[1, 2]])


def test_ai_select_frames_passes_schema_to_generate(monkeypatch):
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return "[1]"

    monkeypatch.setattr(models, "generate", fake_generate)

    frames = [
        {"frame_id": 1, "timestamp": 1.0, "analysis": {"primary": "code"}},
    ]
    categories = {"code": {"description": "Code editors"}}

    result = extract_mod._ai_select_frames(
        frames,
        max_extractions=5,
        categories=categories,
    )

    assert captured["json_schema"] is extract_mod._SCHEMA
    assert result == [1]
