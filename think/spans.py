# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Formatting helpers for storytelling spans JSONL files."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _extract_spans_path_context(file_path: str | Path | None) -> tuple[str, str | None]:
    facet_name = "unknown"
    day_str: str | None = None

    if not file_path:
        return facet_name, day_str

    path = Path(file_path)
    path_str = str(path)
    facet_match = re.search(r"facets/([^/]+)/spans", path_str)
    if facet_match:
        facet_name = facet_match.group(1)

    if path.stem.isdigit() and len(path.stem) == 8:
        day_str = path.stem

    return facet_name, day_str


def _start_seconds(start: str) -> int | None:
    try:
        parts = start.split(":")
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 else 0
        seconds = int(parts[2]) if len(parts) > 2 else 0
    except (IndexError, ValueError, TypeError):
        return None
    return hours * 3600 + minutes * 60 + seconds


def format_spans(
    entries: list[dict],
    context: dict | None = None,
) -> tuple[list[dict], dict]:
    """Format storytelling span JSONL rows into markdown chunks."""
    ctx = context or {}
    file_path = ctx.get("file_path")
    meta: dict[str, Any] = {"indexer": {"agent": "span"}}
    chunks: list[dict[str, Any]] = []
    skipped_count = 0

    facet_name, day_str = _extract_spans_path_context(file_path)

    base_ts = 0
    if day_str:
        try:
            dt = datetime.strptime(day_str, "%Y%m%d")
            base_ts = int(dt.timestamp() * 1000)
        except ValueError:
            pass

    if day_str:
        formatted_day = f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}"
        meta["header"] = f"# Spans for '{facet_name}' facet on {formatted_day}"
    else:
        meta["header"] = f"# Spans for '{facet_name}' facet"

    for entry in entries:
        body = str(entry.get("body") or "").strip()
        start = str(entry.get("start") or "").strip()
        talent = str(entry.get("talent") or "").strip()
        activity_type = str(entry.get("activity_type") or "").strip()
        span_id = str(entry.get("span_id") or "").strip()

        if not all((body, start, talent, activity_type, span_id)):
            skipped_count += 1
            continue

        ts = base_ts
        start_seconds = _start_seconds(start)
        if start_seconds is not None and base_ts:
            ts = base_ts + start_seconds * 1000

        end = str(entry.get("end") or "").strip()
        time_display = start if not end else f"{start}-{end}"
        topics = entry.get("topics", [])
        topics_display = (
            ", ".join(str(topic).strip() for topic in topics if str(topic).strip())
            if isinstance(topics, list)
            else ""
        )

        confidence = entry.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence_display = f"{float(confidence):.2f}"
        else:
            confidence_display = str(confidence or "")

        lines = [f"### {activity_type.capitalize()}: {span_id}\n", ""]
        lines.append(f"**Time:** {time_display}")
        lines.append(f"**Activity Type:** {activity_type}")
        lines.append(f"**Topics:** {topics_display}")
        lines.append(f"**Confidence:** {confidence_display}")
        lines.append(f"**Talent:** {talent}")
        lines.append("")
        lines.append(body)
        lines.append("")

        chunks.append(
            {
                "timestamp": ts,
                "markdown": "\n".join(lines),
                "source": entry,
            }
        )

    if skipped_count > 0:
        error_msg = f"Skipped {skipped_count} entries missing required fields"
        if file_path:
            error_msg += f" in {file_path}"
        meta["error"] = error_msg
        logging.info(error_msg)

    return chunks, meta
