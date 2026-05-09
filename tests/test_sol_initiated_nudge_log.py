# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json

from solstone.convey.sol_initiated.copy import (
    CATEGORIES,
    KIND_SOL_CHAT_REQUEST,
    THROTTLE_RATE_FLOOR,
)
from solstone.convey.sol_initiated.nudge_log import record_nudge_log
from solstone.think.push.triggers import _append_nudge_log


def _rows(journal):
    path = journal / "push" / "nudge_log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_record_nudge_log_writes_outcome_rows(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    record_nudge_log(KIND_SOL_CHAT_REQUEST, "a", CATEGORIES[0], "written")
    record_nudge_log(KIND_SOL_CHAT_REQUEST, "b", CATEGORIES[1], "deduped")
    record_nudge_log(
        KIND_SOL_CHAT_REQUEST,
        "c",
        CATEGORIES[2],
        f"throttled:{THROTTLE_RATE_FLOOR}",
    )

    rows = _rows(tmp_path)
    assert [row["kind"] for row in rows] == [KIND_SOL_CHAT_REQUEST] * 3
    assert [row["dedupe_key"] for row in rows] == ["a", "b", "c"]
    assert [row["category"] for row in rows] == list(CATEGORIES[:3])
    assert [row["outcome"] for row in rows] == [
        "written",
        "deduped",
        f"throttled:{THROTTLE_RATE_FLOOR}",
    ]
    assert all(isinstance(row["ts"], int) for row in rows)


def test_record_nudge_log_preserves_older_rows(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    _append_nudge_log({"ts": 1, "category": "push"})

    record_nudge_log(KIND_SOL_CHAT_REQUEST, "a", CATEGORIES[0], "written")

    rows = _rows(tmp_path)
    assert rows[0] == {"ts": 1, "category": "push"}
    assert rows[1]["kind"] == KIND_SOL_CHAT_REQUEST
