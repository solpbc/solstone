# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for timeline segment summary hooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from solstone.apps.timeline.talent import segment_summary

DAY = "20260512"
SEGMENT = "120000_60"


def _make_activity(
    journal: Path, rel: str, text: str = "Resolved the display reset."
) -> Path:
    path = journal / "chronicle" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("rel", "stream", "expected_origin"),
    [
        (f"{DAY}/{SEGMENT}/activity.md", "", f"{DAY}/{SEGMENT}"),
        (f"{DAY}/{SEGMENT}/talents/activity.md", "", f"{DAY}/{SEGMENT}"),
        (f"{DAY}/archon/{SEGMENT}/activity.md", "archon", f"{DAY}/archon/{SEGMENT}"),
        (
            f"{DAY}/archon/{SEGMENT}/talents/activity.md",
            "archon",
            f"{DAY}/archon/{SEGMENT}",
        ),
    ],
)
def test_pre_process_finds_activity_md_in_layout_all_4(
    timeline_journal,
    rel,
    stream,
    expected_origin,
):
    """AC#2."""
    _make_activity(timeline_journal, rel)

    result = segment_summary.pre_process(
        {"day": DAY, "segment": SEGMENT, "stream": stream}
    )

    assert result is not None
    vars_ = result["template_vars"]
    assert vars_["activity_text"] == "Resolved the display reset."
    assert vars_["segment_rel_path"] == expected_origin


def test_pre_process_skip_when_no_activity_md(timeline_journal):
    """AC#4."""
    (timeline_journal / "chronicle" / DAY / "archon" / SEGMENT).mkdir(parents=True)

    result = segment_summary.pre_process(
        {"day": DAY, "segment": SEGMENT, "stream": "archon"}
    )

    assert result == {"skip_reason": "no_activity_md"}


def test_pre_process_skip_when_timeline_exists_without_refresh(timeline_journal):
    """AC#3."""
    _make_activity(timeline_journal, f"{DAY}/archon/{SEGMENT}/talents/activity.md")
    timeline = (
        timeline_journal / "chronicle" / DAY / "archon" / SEGMENT / "timeline.json"
    )
    timeline.write_text("{}", encoding="utf-8")

    result = segment_summary.pre_process(
        {"day": DAY, "segment": SEGMENT, "stream": "archon"}
    )

    assert result == {"skip_reason": "timeline_exists"}


def test_pre_process_returns_template_vars_with_origin(timeline_journal):
    """AC#2."""
    _make_activity(
        timeline_journal, f"{DAY}/archon/{SEGMENT}/talents/activity.md", "Text"
    )

    result = segment_summary.pre_process(
        {"day": DAY, "segment": SEGMENT, "stream": "archon"}
    )

    assert result == {
        "template_vars": {
            "activity_text": "Text",
            "segment_rel_path": f"{DAY}/archon/{SEGMENT}",
        }
    }


def test_post_process_writes_augmented_timeline_atomically(
    timeline_journal, monkeypatch
):
    """AC#2, AC#3, AC#18 atomicity."""
    _make_activity(timeline_journal, f"{DAY}/archon/{SEGMENT}/talents/activity.md")
    monkeypatch.setattr(segment_summary.time, "time", lambda: 1770000000.9)

    returned = segment_summary.post_process(
        json.dumps(
            {"title": "Display Reset", "description": "Restarts display manager."}
        ),
        {"day": DAY, "segment": SEGMENT, "stream": "archon"},
    )

    assert returned is None
    timeline = (
        timeline_journal / "chronicle" / DAY / "archon" / SEGMENT / "timeline.json"
    )
    assert not list(timeline.parent.glob("*.tmp"))
    assert json.loads(timeline.read_text(encoding="utf-8")) == {
        "title": "Display Reset",
        "description": "Restarts display manager.",
        "origin": f"{DAY}/archon/{SEGMENT}",
        "model": "gemini-3.1-flash-lite",
        "generated_at": 1770000000,
    }


def test_post_process_records_literal_model(timeline_journal):
    """AC#2."""
    _make_activity(timeline_journal, f"{DAY}/{SEGMENT}/activity.md")

    segment_summary.post_process(
        json.dumps({"title": "Short Title", "description": "Brief description here"}),
        {"day": DAY, "segment": SEGMENT, "stream": ""},
    )

    timeline = timeline_journal / "chronicle" / DAY / SEGMENT / "timeline.json"
    assert (
        json.loads(timeline.read_text(encoding="utf-8"))["model"]
        == "gemini-3.1-flash-lite"
    )


@pytest.mark.parametrize(
    ("rel", "stream", "expected_origin"),
    [
        (f"{DAY}/{SEGMENT}/activity.md", "", f"{DAY}/{SEGMENT}"),
        (f"{DAY}/{SEGMENT}/talents/activity.md", "", f"{DAY}/{SEGMENT}"),
        (f"{DAY}/archon/{SEGMENT}/activity.md", "archon", f"{DAY}/archon/{SEGMENT}"),
        (
            f"{DAY}/archon/{SEGMENT}/talents/activity.md",
            "archon",
            f"{DAY}/archon/{SEGMENT}",
        ),
    ],
)
def test_post_process_origin_matches_seed_for_all_4_layouts(
    timeline_journal,
    rel,
    stream,
    expected_origin,
):
    """AC#2."""
    _make_activity(timeline_journal, rel)

    segment_summary.post_process(
        json.dumps({"title": "Short Title", "description": "Brief description here"}),
        {"day": DAY, "segment": SEGMENT, "stream": stream},
    )

    timeline = timeline_journal / "chronicle" / expected_origin / "timeline.json"
    assert json.loads(timeline.read_text(encoding="utf-8"))["origin"] == expected_origin
