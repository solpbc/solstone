# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pre-hook for document analysis talent — skips non-document-import segments."""

import os


def pre_process(context: dict) -> dict | None:
    """Skip segments that are not from the document import stream."""
    if os.environ.get("SOL_STREAM") != "import.document":
        return {"skip_reason": "not a document import segment"}
    return {}
