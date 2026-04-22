# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Sandbox-only seed helper for observer fixture records.

Writes four observer records covering active/stale/inactive/never-connected
states against a sandbox journal. NEVER run against a production journal.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from apps.observer.utils import list_observers, save_observer
from convey import state
from think.utils import get_journal, now_ms

SEEDS: list[tuple[str, int | None]] = [
    ("sandbox-active", 5 * 1000),
    ("sandbox-stale", 60 * 1000),
    ("sandbox-disconnected", 600 * 1000),
    ("sandbox-never-connected", None),
]


def _seed_key(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def seed_observers() -> int:
    journal = Path(get_journal()).expanduser().resolve()
    if not journal.exists() or not journal.is_dir():
        raise RuntimeError(f"Sandbox journal does not exist: {journal}")
    # save_observer() -> get_observers_dir() -> get_app_storage_path() reads
    # convey.state.journal_root (not get_journal()), so we must set it
    # explicitly when running outside a Flask request context.
    state.journal_root = str(journal)

    existing = list_observers()
    bad_names = sorted(
        name
        for observer in existing
        if (name := (observer.get("name") or "")) and not name.startswith("sandbox-")
    )
    if bad_names:
        raise RuntimeError(
            f"Refusing to seed: non-sandbox observer(s) present: {bad_names}"
        )

    existing_by_name = {
        observer.get("name"): observer
        for observer in existing
        if observer.get("name", "").startswith("sandbox-")
    }

    current_now = now_ms()
    written = 0
    for name, offset_ms in SEEDS:
        existing_record = existing_by_name.get(name, {})
        last_seen = None if offset_ms is None else current_now - offset_ms
        record = {
            "key": _seed_key(name),
            "name": name,
            "created_at": existing_record.get("created_at", current_now),
            "last_seen": last_seen,
            "last_segment": None,
            "enabled": True,
            "revoked": False,
            "revoked_at": None,
            "stats": {
                "segments_received": 0,
                "bytes_received": 0,
            },
        }
        if not save_observer(record):
            raise RuntimeError(f"Failed to save observer: {name}")
        written += 1

    print(f"Seeded {written} sandbox observers into {journal}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sandbox-only observer seed helper. Requires "
            "_SOLSTONE_JOURNAL_OVERRIDE to already point at a sandbox journal."
        )
    )
    parser.parse_args()

    try:
        return seed_observers()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
