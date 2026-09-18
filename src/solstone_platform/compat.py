# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Compatibility and installer revision helper functions."""

from pathlib import Path


def load_minimum_installer_revision(path: Path) -> int:
    """Load minimum installer revision integer from compat floor file."""
    text = path.read_text(encoding="utf-8").strip()
    return int(text)


def installer_revision_allows(actual: int, minimum: int) -> bool:
    """Check if actual installer revision satisfies minimum revision floor."""
    return actual >= minimum
