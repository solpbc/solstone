#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc
"""Stdlib-only bootstrap shim for `sol doctor`.

Used as a pre-install entry point on machines that do not yet have `.venv`
populated. Delegates to `think.doctor.main`, which holds the canonical
diagnostic logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solstone.think.doctor import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
