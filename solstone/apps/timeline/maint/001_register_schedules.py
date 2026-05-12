# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Register timeline rollup scheduler entries."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solstone.think.utils import get_journal, setup_cli

logger = logging.getLogger(__name__)

EXPECTED_ENTRIES: dict[str, dict[str, Any]] = {
    "timeline-rollup-day": {
        "cmd": ["sol", "call", "timeline", "rollup-day"],
        "every": "daily",
        "max_runtime": "30m",
    },
    "timeline-rollup-master": {
        "cmd": ["sol", "call", "timeline", "rollup-master"],
        "every": "daily",
        "max_runtime": "30m",
    },
}


@dataclass
class RegistrationSummary:
    added: int = 0
    preserved: int = 0
    warnings: int = 0
    errors: int = 0
    skipped_reason: str | None = None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)
        raise


def _matches_expected(existing: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(existing.get(key) == value for key, value in expected.items())


def run_registration(
    journal_path: Path, *, dry_run: bool = False
) -> RegistrationSummary:
    summary = RegistrationSummary()
    schedules_path = journal_path / "config" / "schedules.json"

    if schedules_path.exists():
        try:
            raw_bytes = schedules_path.read_bytes()
        except OSError as exc:
            logger.warning("Failed to read %s: %s", schedules_path, exc)
            summary.errors += 1
            return summary

        if raw_bytes.strip():
            try:
                raw = json.loads(raw_bytes)
            except json.JSONDecodeError as exc:
                logger.warning("Malformed JSON in %s: %s", schedules_path, exc)
                summary.errors += 1
                return summary
            if not isinstance(raw, dict):
                logger.warning(
                    "Malformed schedules config in %s: expected object", schedules_path
                )
                summary.errors += 1
                return summary
        else:
            raw = {}
    else:
        raw = {}

    changed = False
    for name, expected in EXPECTED_ENTRIES.items():
        existing = raw.get(name)
        if existing is None:
            raw[name] = dict(expected)
            summary.added += 1
            changed = True
            continue
        if isinstance(existing, dict) and _matches_expected(existing, expected):
            summary.preserved += 1
            continue
        logger.warning("Preserving divergent schedule entry %s", name)
        summary.warnings += 1

    if not changed:
        return summary

    if dry_run:
        return summary

    try:
        _atomic_write_json(schedules_path, raw)
    except OSError as exc:
        logger.warning("Failed to write %s: %s", schedules_path, exc)
        summary.errors += 1

    return summary


def _print_summary(summary: RegistrationSummary) -> None:
    print("Summary")
    print(f"  added:     {summary.added}")
    print(f"  preserved: {summary.preserved}")
    print(f"  warnings:  {summary.warnings}")
    print(f"  errors:    {summary.errors}")
    if summary.skipped_reason is not None:
        print(f"  skipped:   {summary.skipped_reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview schedule registration without writing files.",
    )
    args = setup_cli(parser)

    summary = run_registration(Path(get_journal()), dry_run=args.dry_run)
    _print_summary(summary)
    if summary.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
