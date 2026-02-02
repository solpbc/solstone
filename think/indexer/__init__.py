# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Indexer package for journal content.

This module provides the unified journal index for all content types.
"""

# Import from cli
from .cli import main

# Import from journal (unified index)
from .journal import (
    get_events,
    get_journal_index,
    index_file,
    reset_journal_index,
    sanitize_fts_query,
    scan_journal,
    search_counts,
    search_journal,
)

# All public functions and constants
__all__ = [
    # Journal (unified index)
    "get_events",
    "get_journal_index",
    "index_file",
    "reset_journal_index",
    "sanitize_fts_query",
    "scan_journal",
    "search_counts",
    "search_journal",
    # CLI
    "main",
]
