# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from solstone.convey.chat_stream import (
    append_chat_event,
    find_unresponded_trigger,
    read_chat_events,
)
from solstone.convey.sol_initiated.copy import (
    CATEGORIES,
    CATEGORY_CAP_DEFAULTS,
    KIND_OWNER_CHAT_DISMISSED,
    KIND_OWNER_CHAT_OPEN,
    KIND_SOL_CHAT_REQUEST,
    KIND_SOL_CHAT_REQUEST_SUPERSEDED,
    THROTTLE_CATEGORY_SELF_MUTE,
    THROTTLE_DAILY_CAP,
)
from solstone.convey.sol_initiated.events import record_owner_chat_dismissed
from solstone.convey.sol_initiated.start import start_chat

_FIXED_DAY = "20260331"


def _setup_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    journal = tmp_path / "journal"
    journal.mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))
    monkeypatch.setattr("solstone.convey.chat_stream.index_file", lambda *_args: True)
    return journal


def _write_config(
    journal: Path,
    *,
    daily_cap: int = 10,
    category_self_mute_hours: int = 0,
) -> None:
    config_dir = journal / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "journal.json").write_text(
        json.dumps(
            {
                "sol_voice": {
                    "daily_cap": daily_cap,
                    "category_caps": {key: 10 for key in CATEGORY_CAP_DEFAULTS},
                    "rate_floor_minutes": 0,
                    "mute_window": {
                        "enabled": False,
                        "start_hour_local": 22,
                        "end_hour_local": 7,
                    },
                    "category_self_mute_hours": category_self_mute_hours,
                    "category_self_mute_clear_marker_ts": 0,
                    "default_dedupe_window": "24h",
                }
            }
        ),
        encoding="utf-8",
    )


def _fields(kind: str, ts: int = 1_775_000_000_000) -> dict:
    if kind == KIND_SOL_CHAT_REQUEST:
        return {
            "ts": ts,
            "request_id": "req",
            "summary": "summary",
            "message": None,
            "category": CATEGORIES[0],
            "dedupe": "dedupe",
            "dedupe_window": "24h",
            "since_ts": 1,
            "trigger_talent": "reflection",
        }
    if kind == KIND_SOL_CHAT_REQUEST_SUPERSEDED:
        return {"ts": ts, "request_id": "old", "replaced_by": "new"}
    if kind == KIND_OWNER_CHAT_OPEN:
        return {"ts": ts, "request_id": "req", "surface": "test"}
    if kind == KIND_OWNER_CHAT_DISMISSED:
        return {"ts": ts, "request_id": "req", "surface": "test", "reason": None}
    raise AssertionError(kind)


def _chat_rows(journal: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((journal / "chronicle").glob("*/chat/*/chat.jsonl")):
        rows.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    return rows


@pytest.mark.parametrize(
    "kind",
    [
        KIND_SOL_CHAT_REQUEST,
        KIND_SOL_CHAT_REQUEST_SUPERSEDED,
        KIND_OWNER_CHAT_OPEN,
        KIND_OWNER_CHAT_DISMISSED,
    ],
)
def test_new_kinds_validate_and_broadcast_after_disk_write(
    tmp_path, monkeypatch, kind
) -> None:
    journal = _setup_journal(tmp_path, monkeypatch)
    import solstone.convey.chat as chat

    calls: list[str] = []

    class FakeCallosum:
        def emit(self, tract, event, **fields):
            assert any(row["kind"] == event for row in _chat_rows(journal))
            calls.append(event)
            return True

    monkeypatch.setattr(chat, "_runtime", SimpleNamespace(callosum=FakeCallosum()))

    event = append_chat_event(kind, **_fields(kind))

    assert event["kind"] == kind
    assert calls == [kind]


def test_sol_request_requires_locked_fields(tmp_path, monkeypatch) -> None:
    _setup_journal(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="requires fields"):
        append_chat_event(
            KIND_SOL_CHAT_REQUEST,
            ts=1_775_000_000_000,
            request_id="req",
            summary="summary",
        )


def test_find_unresponded_trigger_handles_new_stream_facts(
    tmp_path, monkeypatch
) -> None:
    _setup_journal(tmp_path, monkeypatch)
    request = append_chat_event(KIND_SOL_CHAT_REQUEST, **_fields(KIND_SOL_CHAT_REQUEST))
    append_chat_event(KIND_OWNER_CHAT_OPEN, **_fields(KIND_OWNER_CHAT_OPEN))
    append_chat_event(KIND_OWNER_CHAT_DISMISSED, **_fields(KIND_OWNER_CHAT_DISMISSED))

    assert find_unresponded_trigger(_FIXED_DAY) == request

    append_chat_event(
        "sol_message",
        ts=1_775_000_001_000,
        use_id="u",
        text="done",
        notes="ok",
        requested_target=None,
        requested_task=None,
    )
    assert find_unresponded_trigger(_FIXED_DAY) is None


def test_start_chat_supersede_appends_replacement_atomically(
    tmp_path, monkeypatch
) -> None:
    journal = _setup_journal(tmp_path, monkeypatch)
    _write_config(journal)
    barrier = threading.Barrier(2)
    results = []

    def run(index: int) -> None:
        barrier.wait()
        results.append(
            start_chat(
                summary=f"summary {index}",
                message=None,
                category=CATEGORIES[0],
                dedupe=f"k-{index}",
                dedupe_window=None,
                since_ts=1,
                trigger_talent="reflection",
            )
        )

    threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(result.written for result in results) == [True, True]
    events = read_chat_events(datetime.now().strftime("%Y%m%d"))
    request_ids = {
        event["request_id"]
        for event in events
        if event["kind"] == KIND_SOL_CHAT_REQUEST
    }
    for event in events:
        if event["kind"] == KIND_SOL_CHAT_REQUEST_SUPERSEDED:
            assert event["replaced_by"] in request_ids


def test_start_chat_dedupes_live_duplicate(tmp_path, monkeypatch) -> None:
    journal = _setup_journal(tmp_path, monkeypatch)
    _write_config(journal)

    first = start_chat(
        summary="first",
        message=None,
        category=CATEGORIES[0],
        dedupe="same-key",
        dedupe_window=None,
        since_ts=1,
        trigger_talent="reflection",
    )
    second = start_chat(
        summary="second",
        message=None,
        category=CATEGORIES[0],
        dedupe="same-key",
        dedupe_window=None,
        since_ts=1,
        trigger_talent="reflection",
    )

    assert first.written
    assert second.deduped
    events = read_chat_events(datetime.now().strftime("%Y%m%d"))
    assert sum(1 for event in events if event["kind"] == KIND_SOL_CHAT_REQUEST) == 1


def test_daily_cap_counts_superseded_requests(tmp_path, monkeypatch) -> None:
    journal = _setup_journal(tmp_path, monkeypatch)
    _write_config(journal, daily_cap=5)

    for index in range(5):
        assert start_chat(
            summary=f"summary {index}",
            message=None,
            category=CATEGORIES[0],
            dedupe=f"k-{index}",
            dedupe_window=None,
            since_ts=1,
            trigger_talent="reflection",
        ).written

    result = start_chat(
        summary="blocked",
        message=None,
        category=CATEGORIES[0],
        dedupe="k-final",
        dedupe_window=None,
        since_ts=1,
        trigger_talent="reflection",
    )

    assert result.throttled == THROTTLE_DAILY_CAP
    events = read_chat_events(datetime.now().strftime("%Y%m%d"))
    assert sum(1 for event in events if event["kind"] == KIND_SOL_CHAT_REQUEST) == 5


def test_owner_dismissal_self_mutes_category(tmp_path, monkeypatch) -> None:
    journal = _setup_journal(tmp_path, monkeypatch)
    _write_config(journal, category_self_mute_hours=24)

    first = start_chat(
        summary="first",
        message=None,
        category=CATEGORIES[0],
        dedupe="k-first",
        dedupe_window=None,
        since_ts=1,
        trigger_talent="reflection",
    )
    assert first.request_id is not None
    record_owner_chat_dismissed(first.request_id, "chat", reason=None)

    result = start_chat(
        summary="blocked",
        message=None,
        category=CATEGORIES[0],
        dedupe="k-second",
        dedupe_window=None,
        since_ts=1,
        trigger_talent="reflection",
    )

    assert result.throttled == THROTTLE_CATEGORY_SELF_MUTE
