# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Hook for daily_schedule generator.

Generates activity span data from journal segments to identify optimal
maintenance windows when the user is consistently inactive.
"""

import json
import logging
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from think.utils import get_journal, iter_segments


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


def _get_segments(day_str: str) -> list[tuple[datetime, datetime]]:
    """Get sorted list of segment time ranges for a day."""
    segments = []

    for _stream, seg_key, _seg_path in iter_segments(day_str):
        parsed = _parse_segment(seg_key)
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
    lines = []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days - 1)

    current = start_date
    days_with_data = 0

    while current <= end_date:
        day_str = current.strftime("%Y%m%d")

        segments = _get_segments(day_str)
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


def post_process(result: str, context: dict) -> str | None:
    """Persist primary schedule time to config/schedules.json.

    Extracts the primary time from the generator's JSON output and writes
    it as daily_time in the schedules config for the scheduler to use.

    Args:
        result: The generated JSON output string.
        context: Config dict from the agent pipeline.

    Returns:
        None — side-effect only, does not transform the result.
    """
    logger = logging.getLogger(__name__)

    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        logger.warning("daily_schedule: could not parse result as JSON")
        return None

    primary = data.get("primary")
    if not primary or not isinstance(primary, str):
        logger.warning("daily_schedule: no primary time in result")
        return None

    # Validate HH:MM format
    try:
        datetime.strptime(primary, "%H:%M")
    except ValueError:
        logger.warning("daily_schedule: invalid primary time format: %s", primary)
        return None

    # Atomic read-modify-write of config/schedules.json
    config_path = Path(get_journal()) / "config" / "schedules.json"

    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}
    except (json.JSONDecodeError, OSError):
        config = {}

    config["daily_time"] = primary

    config_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=config_path.parent, suffix=".tmp", prefix=".schedules_"
    )
    tmp_file = Path(tmp_path)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        tmp_file.replace(config_path)
    except BaseException:
        tmp_file.unlink(missing_ok=True)
        raise

    logger.info("daily_schedule: saved daily_time=%s to schedules config", primary)
    return None
