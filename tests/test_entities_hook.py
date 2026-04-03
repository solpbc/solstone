# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for entity hook behavior across schedules."""

from __future__ import annotations

import json
from pathlib import Path


def test_entities_post_process_writes_without_segment(tmp_path):
    from muse.entities import post_process

    output_path = (
        tmp_path
        / "facets"
        / "work"
        / "activities"
        / "20240115"
        / "coding_120000_300"
        / "entities.md"
    )
    result = "* Person: Alice Smith - Mentioned in the meeting\n"

    post_process(result, {"output_path": str(output_path)})

    entities_path = output_path.parent / "entities.jsonl"
    assert entities_path.exists()
    rows = [json.loads(line) for line in entities_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "type": "Person",
            "name": "Alice Smith",
            "description": "Mentioned in the meeting",
        }
    ]


def test_entities_post_process_requires_output_path(caplog):
    from muse.entities import post_process

    post_process("* Person: Alice Smith - Mentioned in the meeting\n", {})

    assert "missing output_path" in caplog.text


def test_entities_muse_is_activity_scheduled():
    from think.muse import get_muse_configs

    segment_prompts = get_muse_configs(schedule="segment")
    activity_prompts = get_muse_configs(schedule="activity")

    assert "entities" not in segment_prompts
    assert activity_prompts["entities"]["activities"] == ["*"]
