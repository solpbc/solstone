# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Shared overlap-header readers for speaker attribution."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _read_segment_overlap_fraction(jsonl_path: Path) -> float:
    """Return overlap_fraction from a chronicle JSONL header, or 0.0 if absent."""
    try:
        with jsonl_path.open(encoding="utf-8") as f:
            line = f.readline()
        if not line:
            return 0.0
        header = json.loads(line)
    except FileNotFoundError:
        return 0.0
    except (OSError, json.JSONDecodeError) as exc:
        logger.info("overlap header read failed at %s: %s", jsonl_path, exc)
        return 0.0

    value = header.get("overlap_fraction", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
