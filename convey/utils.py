import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from think.models import GEMINI_FLASH, gemini_generate
from think.utils import day_dirs, day_path, get_topics

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


def _combine(day: str, time_str: str) -> str:
    """Return ISO timestamp string for ``day`` + ``time_str``."""

    return f"{day[:4]}-{day[4:6]}-{day[6:]}T{time_str}"


def build_occurrence_index(journal: str) -> Dict[str, List[Dict[str, Any]]]:
    """Aggregate occurrences from all topic JSON files for each day."""

    index: Dict[str, List[Dict[str, Any]]] = {}
    for name, path in day_dirs().items():

        occs: List[Dict[str, Any]] = []
        topics_dir = os.path.join(path, "topics")
        if not os.path.isdir(topics_dir):
            continue
        topics = get_topics()
        for fname in os.listdir(topics_dir):
            base, ext = os.path.splitext(fname)
            if ext != ".json" or base not in topics:
                continue
            file_path = os.path.join(topics_dir, fname)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data.get("occurrences", []) if isinstance(data, dict) else data
            except Exception:
                continue
            if not isinstance(items, list):
                continue
            topic_counts: Dict[str, int] = {}
            topic = base
            for occ in items:
                count = topic_counts.get(topic, 0)
                topic_counts[topic] = count + 1

                o: Dict[str, Any] = {
                    "title": occ.get("title", ""),
                    "summary": occ.get("summary", ""),
                    "subject": occ.get("subject", ""),
                    "details": occ.get("details", occ.get("description", "")),
                    "participants": occ.get("participants", []),
                    "topic": topic,
                    "color": topics[topic]["color"],
                    "path": os.path.join(name, "topics", fname),
                    "index": count,
                }
                if occ.get("start"):
                    o["startTime"] = _combine(name, occ["start"])
                if occ.get("end"):
                    o["endTime"] = _combine(name, occ["end"])
                occs.append(o)

        if occs:
            index[name] = occs

    return index


# Backwards compatibility
build_index = build_occurrence_index
