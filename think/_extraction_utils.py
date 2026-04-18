# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Generic helper for logging extraction (IncompleteJSONError) failures."""

import logging


def log_extraction_failure(e: Exception, name: str) -> None:
    """Log enhanced diagnostics for extraction generation failures.

    Handles IncompleteJSONError specially by logging a single-line summary
    with a head+tail sample and degenerate repetition detection.

    Args:
        e: The exception from generate().
        name: Generator name for log context.
    """
    from think.models import IncompleteJSONError

    if not isinstance(e, IncompleteJSONError):
        logging.error("Extraction generation failed for %s: %s", name, e)
        return

    partial = e.partial_text
    length = len(partial)

    # Build single-line head+tail sample (newlines collapsed for log grep)
    def _collapse(s: str) -> str:
        return s.replace("\n", "\\n").replace("\r", "")

    if length <= 300:
        sample = _collapse(partial)
    else:
        sample = f"{_collapse(partial[:150])} ... {_collapse(partial[-150:])}"

    # Repetition detection: count unique chars in last 1000
    tail = partial[-1000:] if length >= 1000 else partial
    unique_count = len(set(tail))
    repetition_flag = ""
    if unique_count < 20:
        repetition_flag = (
            f" [POSSIBLE DEGENERATE REPETITION: "
            f"{unique_count} unique chars in last {len(tail)}]"
        )

    logging.error(
        "Extraction generation failed for %s: %s "
        "(partial_text: %d chars, %d unique in tail%s) sample: %s",
        name,
        e,
        length,
        unique_count,
        repetition_flag,
        sample,
    )
