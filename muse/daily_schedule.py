# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pre-hook for daily_schedule generator.

Generates activity span data from journal segments to identify optimal
maintenance windows when the user is consistently inactive.
"""

import os
import re
from datetime import datetime, timedelta

from think.utils import get_journal


def _parse_segment(folder_name: str) -> tuple[datetime, int] | None:
    """Parse segment folder name into start time and duration."""
    match = re.match(r"^(\d{6})_(\d+)(?:_\w+)?$", folder_name)
    if not match:
        return None

    time_str, duration_str = match.groups()
    try:
        hour = int(time_str[0:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        duration = int(duration_str)
        start = datetime(2000, 1, 1, hour, minute, second)
        return start, duration
    except ValueError:
        return None


def _get_segments(day_path: str) -> list[tuple[datetime, datetime]]:
    """Get sorted list of segment time ranges for a day."""
    segments = []

    if not os.path.isdir(day_path):
        return segments

    for entry in os.listdir(day_path):
        entry_path = os.path.join(day_path, entry)
        if not os.path.isdir(entry_path):
            continue

        parsed = _parse_segment(entry)
        if parsed is None:
            continue

        start, duration = parsed
        end = start + timedelta(seconds=duration)
        segments.append((start, end))

    segments.sort(key=lambda x: x[0])
    return segments


def _build_spans(
    segments: list[tuple[datetime, datetime]],
    gap_seconds: int = 300,
    min_minutes: int = 10,
) -> list[tuple[datetime, datetime]]:
    """Group segments into spans based on gap threshold."""
    if not segments:
        return []

    spans = []
    span_start, span_end = segments[0]

    for seg_start, seg_end in segments[1:]:
        gap = (seg_start - span_end).total_seconds()

        if gap > gap_seconds:
            duration_minutes = (span_end - span_start).total_seconds() / 60
            if duration_minutes >= min_minutes:
                spans.append((span_start, span_end))
            span_start = seg_start

        span_end = seg_end

    duration_minutes = (span_end - span_start).total_seconds() / 60
    if duration_minutes >= min_minutes:
        spans.append((span_start, span_end))

    return spans


def _format_time(dt: datetime) -> str:
    """Format datetime as HH:MM."""
    return dt.strftime("%H:%M")


def _format_duration(start: datetime, end: datetime) -> str:
    """Format duration as Xh Ym."""
    total_minutes = int((end - start).total_seconds() / 60)
    hours, minutes = divmod(total_minutes, 60)
    if hours > 0 and minutes > 0:
        return f"{hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h"
    return f"{minutes}m"


def _get_weekday(date_str: str) -> str:
    """Get weekday name from YYYYMMDD string."""
    dt = datetime.strptime(date_str, "%Y%m%d")
    return dt.strftime("%A")


def generate_span_summary(days: int = 7) -> str:
    """Generate activity span summary for the past N days.

    Args:
        days: Number of days of history to analyze.

    Returns:
        Formatted text summarizing activity windows per day.
    """
    journal = get_journal()
    lines = []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days - 1)

    current = start_date
    days_with_data = 0

    while current <= end_date:
        day_str = current.strftime("%Y%m%d")
        day_path = os.path.join(journal, day_str)

        segments = _get_segments(day_path)
        spans = _build_spans(segments)

        if spans:
            days_with_data += 1
            weekday = _get_weekday(day_str)
            lines.append(f"{day_str} ({weekday}):")

            for span_start, span_end in spans:
                duration = _format_duration(span_start, span_end)
                lines.append(
                    f"  {_format_time(span_start)} - {_format_time(span_end)} ({duration})"
                )

            lines.append("")

        current += timedelta(days=1)

    if days_with_data == 0:
        return "No activity data found for the past week."

    header = f"Activity windows for the past {days} days ({days_with_data} days with data):\n\n"
    return header + "\n".join(lines)


def pre_process(context: dict) -> dict | None:
    """Generate span data to replace transcript content.

    Args:
        context: PreHookContext with day, meta, etc.

    Returns:
        Dict with transcript replacement, or None if insufficient data.
    """
    # Get lookback window from meta, default 7 days
    meta = context.get("meta", {})
    days = meta.get("lookback_days", 7)

    span_summary = generate_span_summary(days=days)

    return {"transcript": span_summary}
