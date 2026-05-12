# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for timeline day rollup command."""

from __future__ import annotations

import asyncio
import json

from typer.testing import CliRunner

from solstone.apps.timeline.call import _rollup_day, app
from solstone.apps.timeline.tests.conftest import write_json

DAY = "20260512"


def _write_segment(journal, day, segment, title, hour_stream="archon"):
    write_json(
        journal / "chronicle" / day / hour_stream / segment / "timeline.json",
        {
            "title": title,
            "description": f"{title} description.",
            "origin": f"{day}/{hour_stream}/{segment}",
            "model": "gemini-3.1-flash-lite",
            "generated_at": 1770000000,
        },
    )


def test_rollup_day_help_flag_matrix():
    """AC#5."""
    result = CliRunner().invoke(app, ["rollup-day", "--help"])

    assert result.exit_code == 0
    assert "DAY" in result.output
    for flag in ("--top", "--force", "--jobs", "--dry-run"):
        assert flag in result.output


def test_rollup_day_dry_run_no_llm_calls(timeline_journal, mock_agenerate):
    """AC#7."""
    mock = mock_agenerate({"picks": [0], "rationale": "unused"})
    _write_segment(timeline_journal, DAY, "120000_60", "One")

    asyncio.run(
        _rollup_day(timeline_journal, DAY, top=4, jobs=5, dry_run=True, force=False)
    )

    assert mock.call_count == 0


def test_rollup_day_writes_seed_shape(timeline_journal, mock_agenerate):
    """AC#6."""
    for i in range(5):
        _write_segment(timeline_journal, DAY, f"12000{i}_60", f"Event {i}")
    mock = mock_agenerate({"picks": [0, 1, 2, 3], "rationale": "highest consequence"})

    asyncio.run(
        _rollup_day(timeline_journal, DAY, top=4, jobs=5, dry_run=False, force=False)
    )

    payload = json.loads(
        (timeline_journal / "chronicle" / DAY / "timeline.json").read_text()
    )
    assert payload["day"] == DAY
    assert payload["model"] == "gemini-3-flash-preview"
    assert payload["segment_count"] == 5
    assert payload["hour_count"] == 1
    assert len(payload["day_top"]) == 4
    assert payload["hours"]["12"]["rationale"] == "highest consequence"
    assert mock.call_args.kwargs["model"] == "gemini-3-flash-preview"


def test_rollup_day_skip_when_exists_without_force(timeline_journal, mock_agenerate):
    """AC#6, AC#7."""
    write_json(
        timeline_journal / "chronicle" / DAY / "timeline.json", {"existing": True}
    )
    mock = mock_agenerate({"picks": [0], "rationale": "unused"})
    _write_segment(timeline_journal, DAY, "120000_60", "One")

    asyncio.run(
        _rollup_day(timeline_journal, DAY, top=4, jobs=5, dry_run=False, force=False)
    )

    assert json.loads(
        (timeline_journal / "chronicle" / DAY / "timeline.json").read_text()
    ) == {"existing": True}
    assert mock.call_count == 0


def test_rollup_day_force_overwrites_atomically(timeline_journal):
    """AC#6."""
    write_json(
        timeline_journal / "chronicle" / DAY / "timeline.json", {"existing": True}
    )
    _write_segment(timeline_journal, DAY, "120000_60", "One")

    asyncio.run(
        _rollup_day(timeline_journal, DAY, top=4, jobs=5, dry_run=False, force=True)
    )

    timeline_path = timeline_journal / "chronicle" / DAY / "timeline.json"
    payload = json.loads(timeline_path.read_text())
    assert payload["day"] == DAY
    assert payload["day_top"][0]["title"] == "One"
    assert not list(timeline_path.parent.glob("*.tmp"))


def test_rollup_day_hour_error_continues_picks_empty_with_error_field(
    timeline_journal,
    mock_agenerate,
):
    """AC#20."""
    for i in range(5):
        _write_segment(timeline_journal, DAY, f"12000{i}_60", f"Noon {i}")
        _write_segment(timeline_journal, DAY, f"13000{i}_60", f"One {i}")
    mock_agenerate(
        RuntimeError("hour backend down"),
        {"picks": [0, 1, 2, 3], "rationale": "one pm"},
    )

    asyncio.run(
        _rollup_day(timeline_journal, DAY, top=4, jobs=5, dry_run=False, force=False)
    )

    payload = json.loads(
        (timeline_journal / "chronicle" / DAY / "timeline.json").read_text()
    )
    assert payload["hours"]["12"]["picks"] == []
    assert "hour backend down" in payload["hours"]["12"]["error"]
    assert len(payload["day_top"]) == 4


def test_rollup_day_final_error_skips_write_exits_zero(
    timeline_journal, mock_agenerate
):
    """AC#21."""
    for i in range(2):
        _write_segment(timeline_journal, DAY, f"12000{i}_60", f"Noon {i}")
        _write_segment(timeline_journal, DAY, f"13000{i}_60", f"One {i}")
    mock_agenerate(
        {"picks": [0], "rationale": "noon"},
        {"picks": [0], "rationale": "one"},
        RuntimeError("day backend down"),
    )

    result = asyncio.run(
        _rollup_day(timeline_journal, DAY, top=1, jobs=5, dry_run=False, force=False)
    )

    assert result is None
    assert not (timeline_journal / "chronicle" / DAY / "timeline.json").exists()
