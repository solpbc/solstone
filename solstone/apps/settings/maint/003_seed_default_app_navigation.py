# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Seed default Convey app navigation config."""

from __future__ import annotations

import logging
import sys

import solstone.convey.state as convey_state
from solstone.convey.config import (
    load_convey_config,
    save_convey_config,
    seed_default_app_navigation,
)
from solstone.think.utils import get_journal

logger = logging.getLogger(__name__)


def _fail(message: str, exc: Exception | None = None) -> None:
    if exc is None:
        logger.error(message)
    else:
        logger.exception(message)
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main():
    try:
        journal = get_journal()
    except Exception as exc:
        _fail("Could not resolve journal for default app navigation seed", exc)

    convey_state.journal_root = str(journal)

    config = load_convey_config()
    if not seed_default_app_navigation(config):
        print("Default app navigation already present.")
        return

    try:
        saved = save_convey_config(config)
    except Exception as exc:
        _fail("default app navigation seed convey-config PERSIST failed", exc)

    if not saved:
        _fail("default app navigation seed convey-config PERSIST failed")

    print("Seeded default app navigation.")


if __name__ == "__main__":
    main()
