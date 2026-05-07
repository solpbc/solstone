# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Compatibility wrapper - real implementation lives in think.merge."""

from solstone.think.merge import (  # noqa: F401
    MergeSummary,
    merge_journals,
)

__all__ = ["MergeSummary", "merge_journals"]
