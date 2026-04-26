# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
import threading


def test_locked_modify_serializes_concurrent_edits(tmp_path, monkeypatch):
    from think.activities import (
        append_activity_record,
        append_edit,
        load_activity_records,
        locked_modify,
    )

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    facet = "work"
    day = "20260418"
    record_id = "coding_090000_300"
    record_path = tmp_path / "facets" / facet / "activities" / f"{day}.jsonl"

    append_activity_record(
        facet,
        day,
        {
            "id": record_id,
            "activity": "coding",
            "description": "Initial description",
            "segments": ["090000_300"],
            "created_at": 1,
        },
    )

    first_inside_lock = threading.Event()
    release_first = threading.Event()

    def worker(actor: str, note: str, hold_lock: bool = False) -> None:
        def modify_fn(records: list[dict]) -> list[dict]:
            updated = []
            for record in records:
                if record.get("id") == record_id:
                    record = append_edit(
                        record,
                        actor=actor,
                        fields=["details"],
                        note=note,
                    )
                    if hold_lock:
                        first_inside_lock.set()
                        release_first.wait(timeout=2)
                updated.append(record)
            return updated

        locked_modify(record_path, modify_fn)

    first = threading.Thread(
        target=worker, args=("cli:update", "first writer"), kwargs={"hold_lock": True}
    )
    second = threading.Thread(
        target=worker, args=("cli:mute", "second writer"), kwargs={"hold_lock": False}
    )

    first.start()
    assert first_inside_lock.wait(timeout=2)
    second.start()
    release_first.set()

    first.join(timeout=2)
    second.join(timeout=2)
    assert not first.is_alive()
    assert not second.is_alive()

    records = load_activity_records(facet, day, include_hidden=True)
    assert len(records) == 1
    assert [edit["note"] for edit in records[0]["edits"]] == [
        "first writer",
        "second writer",
    ]

    raw_lines = record_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 1
    assert json.loads(raw_lines[0])["edits"][1]["actor"] == "cli:mute"
