# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Rename legacy unified and triage provider contexts for the chat refactor."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from solstone.think.utils import get_journal, setup_cli

logger = logging.getLogger(__name__)

_UNIFIED_CONTEXT = "talent.system.unified"
_CHAT_CONTEXT = "talent.system.chat"
_TRIAGE_CONTEXT = "talent.system.triage"


@dataclass
class MigrationSummary:
    renamed: int = 0
    removed: int = 0
    preserved: int = 0
    errors: int = 0
    skipped_reason: str | None = None


def run_migration(journal_path: Path, *, dry_run: bool) -> MigrationSummary:
    summary = MigrationSummary()
    config_path = journal_path / "config" / "journal.json"

    if not config_path.exists():
        summary.skipped_reason = "no file"
        return summary

    try:
        raw_bytes = config_path.read_bytes()
    except OSError:
        logger.exception("Failed to read %s", config_path)
        summary.errors += 1
        return summary

    if not raw_bytes.strip():
        summary.skipped_reason = "empty file"
        return summary

    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError:
        summary.skipped_reason = "unparseable"
        return summary

    if not isinstance(raw, dict):
        summary.skipped_reason = "unparseable"
        return summary

    providers = raw.get("providers")
    if not isinstance(providers, dict):
        summary.skipped_reason = "no providers"
        return summary

    contexts = providers.get("contexts")
    if not isinstance(contexts, dict):
        summary.skipped_reason = "no contexts"
        return summary

    changed = False
    if _UNIFIED_CONTEXT in contexts:
        legacy_chat = contexts[_UNIFIED_CONTEXT]
        if _CHAT_CONTEXT not in contexts:
            contexts[_CHAT_CONTEXT] = legacy_chat
            summary.renamed += 1
        else:
            summary.preserved += 1
        del contexts[_UNIFIED_CONTEXT]
        changed = True

    if _TRIAGE_CONTEXT in contexts:
        del contexts[_TRIAGE_CONTEXT]
        summary.removed += 1
        changed = True

    if not changed:
        return summary

    if dry_run:
        return summary

    try:
        _write_config(config_path, raw)
    except OSError:
        logger.exception("Failed to write %s", config_path)
        summary.errors += 1

    return summary


def _write_config(config_path: Path, config: dict) -> None:
    config_dir = config_path.parent
    fd, tmp_path = tempfile.mkstemp(
        dir=config_dir,
        suffix=".tmp",
        prefix=".journal_",
        text=True,
    )
    tmp_file = Path(tmp_path)
    try:
        with open(fd, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        tmp_file.replace(config_path)
    except BaseException:
        tmp_file.unlink(missing_ok=True)
        raise


def _print_summary(summary: MigrationSummary) -> None:
    logger.info("Summary")
    logger.info("  renamed:  %d", summary.renamed)
    logger.info("  removed:  %d", summary.removed)
    logger.info("  preserved:%d", summary.preserved)
    logger.info("  errors:   %d", summary.errors)
    if summary.skipped_reason is not None:
        logger.info("  skipped:  %s", summary.skipped_reason)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the provider-context rename without writing files.",
    )
    args = setup_cli(parser)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    journal_path = Path(get_journal())
    summary = run_migration(journal_path, dry_run=args.dry_run)

    _print_summary(summary)
    if summary.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
