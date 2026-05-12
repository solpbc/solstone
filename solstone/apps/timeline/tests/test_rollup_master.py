# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for timeline master rollup command."""

from __future__ import annotations

import asyncio
import json

from typer.testing import CliRunner

from solstone.apps.timeline.call import _rollup_master, app
from solstone.apps.timeline.tests.conftest import write_json


def _write_day(journal, day, titles):
    write_json(
        journal / "chronicle" / day / "timeline.json",
        {
            "day": day,
            "model": "gemini-3-flash-preview",
            "generated_at": 1770000000,
            "segment_count": len(titles),
            "hour_count": 1,
            "day_top": [
                {
                    "title": title,
                    "description": f"{title} description.",
                    "origin": f"{day}/archon/120000_60",
                }
                for title in titles
            ],
            "day_rationale": "fixture",
            "hours": {},
        },
    )


def test_rollup_master_help_flag_matrix():
    """AC#8."""
    result = CliRunner().invoke(app, ["rollup-master", "--help"])

    assert result.exit_code == 0
    for flag in ("--top", "--force", "--jobs", "--dry-run", "--months"):
        assert flag in result.output


def test_rollup_master_writes_seed_shape(timeline_journal, mock_agenerate):
    """AC#9, AC#10."""
    _write_day(timeline_journal, "20260510", ["A", "B", "C"])
    _write_day(timeline_journal, "20260511", ["D", "E"])
    mock = mock_agenerate({"picks": [4, 0, 1, 2], "rationale": "monthly consequence"})

    asyncio.run(
        _rollup_master(
            timeline_journal,
            top=4,
            jobs=5,
            dry_run=False,
            force=False,
            months_filter=None,
        )
    )

    payload = json.loads((timeline_journal / "timeline.json").read_text())
    assert payload["model"] == "gemini-3-flash-preview"
    assert payload["top_n"] == 4
    assert list(payload["months"]) == ["202605"]
    assert payload["months"]["202605"]["day_count"] == 2
    assert payload["months"]["202605"]["month_top"][0]["title"] == "E"
    assert payload["year_top"][0]["month"] == "202605"
    assert mock.call_args.kwargs["model"] == "gemini-3-flash-preview"


def test_rollup_master_year_top_is_month_top_first_per_month(timeline_journal):
    """AC#10."""
    _write_day(timeline_journal, "20260410", ["April Head", "April Other"])
    _write_day(timeline_journal, "20260510", ["May Head", "May Other"])

    asyncio.run(
        _rollup_master(
            timeline_journal,
            top=4,
            jobs=5,
            dry_run=False,
            force=False,
            months_filter=None,
        )
    )

    payload = json.loads((timeline_journal / "timeline.json").read_text())
    assert payload["year_top"] == [
        {
            "month": "202604",
            "title": "April Head",
            "description": "April Head description.",
            "origin": "20260410/archon/120000_60",
        },
        {
            "month": "202605",
            "title": "May Head",
            "description": "May Head description.",
            "origin": "20260510/archon/120000_60",
        },
    ]


def test_rollup_master_omits_empty_months(timeline_journal):
    """AC#23."""
    _write_day(timeline_journal, "20260410", [])
    _write_day(timeline_journal, "20260510", ["May Head"])

    asyncio.run(
        _rollup_master(
            timeline_journal,
            top=4,
            jobs=5,
            dry_run=False,
            force=False,
            months_filter=None,
        )
    )

    payload = json.loads((timeline_journal / "timeline.json").read_text())
    assert "202604" not in payload["months"]
    assert "202605" in payload["months"]


def test_rollup_master_month_filter_comma_separated_yyyymm(timeline_journal):
    """AC#8."""
    _write_day(timeline_journal, "20260410", ["April Head"])
    _write_day(timeline_journal, "20260510", ["May Head"])

    asyncio.run(
        _rollup_master(
            timeline_journal,
            top=4,
            jobs=5,
            dry_run=False,
            force=False,
            months_filter={"202605"},
        )
    )

    payload = json.loads((timeline_journal / "timeline.json").read_text())
    assert list(payload["months"]) == ["202605"]


def test_rollup_master_month_error_nonfatal(timeline_journal, mock_agenerate):
    """AC#22."""
    _write_day(timeline_journal, "20260510", ["A", "B", "C", "D", "E"])
    mock_agenerate(RuntimeError("month backend down"))

    asyncio.run(
        _rollup_master(
            timeline_journal,
            top=4,
            jobs=5,
            dry_run=False,
            force=False,
            months_filter=None,
        )
    )

    payload = json.loads((timeline_journal / "timeline.json").read_text())
    assert payload["months"]["202605"]["month_top"] == []
    assert payload["months"]["202605"]["month_rationale"] == "ERROR: month backend down"
    assert payload["year_top"] == []
