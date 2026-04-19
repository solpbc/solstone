# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for spans JSONL formatting."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def test_format_spans_builds_chunks_and_metadata():
    from think.spans import format_spans

    entries = [
        {
            "span_id": "meeting_090000_300",
            "talent": "conversation",
            "facet": "work",
            "day": "20260101",
            "activity_type": "meeting",
            "start": "09:00:00",
            "end": "09:15:00",
            "body": "Aligned on the launch plan and confirmed owners.",
            "topics": ["launch", "owners", "planning"],
            "confidence": 0.93,
        },
        {
            "span_id": "coding_130000_300",
            "talent": "work",
            "facet": "work",
            "day": "20260101",
            "activity_type": "coding",
            "start": "13:00:00",
            "end": "13:10:00",
            "body": "Implemented the migration and updated the tests.",
            "topics": ["migration", "tests", "backend"],
            "confidence": 0.81,
        },
    ]

    file_path = Path("/tmp/journal/facets/work/spans/20260101.jsonl")
    chunks, meta = format_spans(entries, {"file_path": file_path})

    assert len(chunks) == 2
    assert meta["header"] == "# Spans for 'work' facet on 2026-01-01"
    assert meta["indexer"] == {"agent": "span"}

    first = chunks[0]
    expected_ts = int(datetime.strptime("20260101", "%Y%m%d").timestamp() * 1000)
    expected_ts += 9 * 3600 * 1000
    assert first["timestamp"] == expected_ts
    assert first["source"] == entries[0]
    assert "### Meeting: meeting_090000_300" in first["markdown"]
    assert "**Time:** 09:00:00-09:15:00" in first["markdown"]
    assert "**Activity Type:** meeting" in first["markdown"]
    assert "**Topics:** launch, owners, planning" in first["markdown"]
    assert "**Confidence:** 0.93" in first["markdown"]
    assert "**Talent:** conversation" in first["markdown"]
    assert "Aligned on the launch plan" in first["markdown"]


def test_format_spans_skips_invalid_rows_and_reports_error():
    from think.spans import format_spans

    entries = [
        {
            "span_id": "valid_1",
            "talent": "work",
            "facet": "work",
            "day": "20260101",
            "activity_type": "coding",
            "start": "08:00:00",
            "end": "08:05:00",
            "body": "Valid row.",
            "topics": ["alpha", "beta", "gamma"],
            "confidence": 0.5,
        },
        {
            "span_id": "invalid_1",
            "talent": "work",
            "facet": "work",
            "day": "20260101",
            "activity_type": "coding",
            "start": "08:05:00",
            "end": "08:10:00",
            "topics": ["alpha", "beta", "gamma"],
            "confidence": 0.5,
        },
    ]

    chunks, meta = format_spans(
        entries, {"file_path": Path("/tmp/journal/facets/work/spans/20260101.jsonl")}
    )

    assert len(chunks) == 1
    assert "Skipped 1 entries missing required fields" in meta["error"]
    assert "20260101.jsonl" in meta["error"]
    assert meta["indexer"] == {"agent": "span"}


def test_get_formatter_returns_spans_formatter():
    from think.formatters import get_formatter

    formatter = get_formatter("facets/foo/spans/20260101.jsonl")

    assert formatter is not None
    assert formatter.__name__ == "format_spans"
