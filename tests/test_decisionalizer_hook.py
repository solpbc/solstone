# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for decisionalizer pre-hook."""

from unittest.mock import patch

from talent.decisionalizer import pre_process


def test_skip_when_no_decisions(tmp_path):
    """Skip when no decisions.md files exist for the day."""
    activities = tmp_path / "facets" / "somefacet" / "activities" / "20260410"
    activities.mkdir(parents=True)

    with patch("talent.decisionalizer.get_journal", return_value=str(tmp_path)):
        result = pre_process({"day": "20260410"})

    assert result == {"skip_reason": "no decision outputs for day"}


def test_proceed_when_decisions_exist(tmp_path):
    """Proceed when decisions.md files exist for the day."""
    decisions = (
        tmp_path
        / "facets"
        / "testfacet"
        / "activities"
        / "20260410"
        / "meeting_100000_300"
        / "decisions.md"
    )
    decisions.parent.mkdir(parents=True)
    decisions.write_text("")

    with patch("talent.decisionalizer.get_journal", return_value=str(tmp_path)):
        result = pre_process({"day": "20260410"})

    assert result == {}
