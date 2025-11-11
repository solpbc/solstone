import os
import re
import time
from datetime import datetime
from typing import Optional

from think.utils import day_dirs

DATE_RE = re.compile(r"\d{8}")


def adjacent_days(journal: str, day: str) -> tuple[Optional[str], Optional[str]]:
    """Return previous and next day folder names if they exist."""
    if not journal or not os.path.isdir(journal):
        return None, None
    days = sorted(day_dirs())
    if day not in days:
        return None, None
    idx = days.index(day)
    prev_day = days[idx - 1] if idx > 0 else None
    next_day = days[idx + 1] if idx < len(days) - 1 else None
    return prev_day, next_day


def format_date(date_str: str) -> str:
    """Convert YYYYMMDD to 'Wednesday April 2nd' format."""
    try:
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        day = date_obj.day
        if 10 <= day % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return date_obj.strftime(f"%A %B {day}{suffix}")
    except ValueError:
        return date_str


def time_since(epoch: int) -> str:
    """Return short human readable age for ``epoch`` seconds."""
    seconds = int(time.time() - epoch)
    if seconds < 60:
        return f"{seconds} seconds ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"
    weeks = days // 7
    return f"{weeks} week{'s' if weeks != 1 else ''} ago"
