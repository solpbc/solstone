# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Register timeline segment summary provider context."""

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

CONTEXT_NAME = "talent.timeline.segment_summary"
EXPECTED_CONTEXT = {"provider": "google", "model": "gemini-3.1-flash-lite"}


@dataclass
class RegistrationSummary:
    added: int = 0
    preserved: int = 0
    warnings: int = 0
    errors: int = 0


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


def run_registration(
    journal_path: Path, *, dry_run: bool = False
) -> RegistrationSummary:
    summary = RegistrationSummary()
    config_path = journal_path / "config" / "journal.json"

    if config_path.exists():
        try:
            raw_bytes = config_path.read_bytes()
        except OSError as exc:
            logger.warning("Failed to read %s: %s", config_path, exc)
            summary.errors += 1
            return summary

        if raw_bytes.strip():
            try:
                raw = json.loads(raw_bytes)
            except json.JSONDecodeError as exc:
                logger.warning("Malformed JSON in %s: %s", config_path, exc)
                summary.errors += 1
                return summary
            if not isinstance(raw, dict):
                logger.warning(
                    "Malformed journal config in %s: expected object", config_path
                )
                summary.errors += 1
                return summary
        else:
            raw = {}
    else:
        raw = {}

    providers = raw.setdefault("providers", {})
    if not isinstance(providers, dict):
        logger.warning("Preserving divergent providers config in %s", config_path)
        summary.warnings += 1
        return summary

    contexts = providers.setdefault("contexts", {})
    if not isinstance(contexts, dict):
        logger.warning(
            "Preserving divergent providers.contexts config in %s", config_path
        )
        summary.warnings += 1
        return summary

    existing = contexts.get(CONTEXT_NAME)
    if existing == EXPECTED_CONTEXT:
        summary.preserved += 1
        return summary
    if existing is not None:
        logger.warning("Preserving divergent provider context %s", CONTEXT_NAME)
        summary.warnings += 1
        return summary

    contexts[CONTEXT_NAME] = dict(EXPECTED_CONTEXT)
    summary.added += 1

    if dry_run:
        return summary

    try:
        _atomic_write_json(config_path, raw)
    except OSError as exc:
        logger.warning("Failed to write %s: %s", config_path, exc)
        summary.errors += 1

    return summary


def _print_summary(summary: RegistrationSummary) -> None:
    print("Summary")
    print(f"  added:     {summary.added}")
    print(f"  preserved: {summary.preserved}")
    print(f"  warnings:  {summary.warnings}")
    print(f"  errors:    {summary.errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview provider context registration without writing files.",
    )
    args = setup_cli(parser)

    summary = run_registration(Path(get_journal()), dry_run=args.dry_run)
    _print_summary(summary)
    if summary.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
