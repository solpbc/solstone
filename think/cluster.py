import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .utils import day_path, setup_cli

TIME_RE = r"(\d{6})"
AUDIO_PATTERN = re.compile(rf"^{TIME_RE}_audio\.json$")
SCREEN_SUMMARY_PATTERN = re.compile(rf"^{TIME_RE}_screen\.md$")
SCREEN_DIFF_PATTERN = re.compile(rf"^{TIME_RE}_monitor_(\d+)_diff\.json$")


def _date_str(day_dir: str) -> str:
    base = os.path.basename(os.path.normpath(day_dir))
    if not re.fullmatch(r"\d{8}", base):
        raise ValueError("day_dir must end with YYYYMMDD")
    return base


def _load_entries(day_dir: str, audio: bool, screen_mode: str) -> List[Dict[str, str]]:
    date_str = _date_str(day_dir)
    entries: List[Dict[str, str]] = []
    for filename in os.listdir(day_dir):
        match = None
        prefix = None
        monitor: Optional[str] = None

        if audio and (match := AUDIO_PATTERN.match(filename)):
            time_part = match.group(1)
            prefix = "audio"
        elif screen_mode == "summary" and (match := SCREEN_SUMMARY_PATTERN.match(filename)):
            time_part = match.group(1)
            prefix = "screen"
        elif screen_mode == "raw" and (match := SCREEN_DIFF_PATTERN.match(filename)):
            time_part = match.group(1)
            monitor = match.group(2)
            prefix = "monitor"
        else:
            continue

        timestamp = datetime.strptime(date_str + time_part, "%Y%m%d%H%M%S")
        path = os.path.join(day_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:  # pragma: no cover - warning only
            print(f"Warning: Could not read file {filename}: {e}", file=sys.stderr)
            continue

        entries.append(
            {
                "timestamp": timestamp,
                "prefix": prefix,
                "content": content,
                "monitor": monitor,
            }
        )

    entries.sort(key=lambda e: e["timestamp"])
    return entries


def _group_entries(entries: List[Dict[str, str]]) -> Dict[datetime, List[Dict[str, str]]]:
    grouped: Dict[datetime, List[Dict[str, str]]] = defaultdict(list)
    for e in entries:
        ts = e["timestamp"]
        interval = ts.replace(minute=ts.minute - (ts.minute % 5), second=0, microsecond=0)
        grouped[interval].append(e)
    return grouped


def _groups_to_markdown(groups: Dict[datetime, List[Dict[str, str]]]) -> str:
    lines: List[str] = []
    for interval_start in sorted(groups.keys()):
        interval_end = interval_start + timedelta(minutes=5)
        lines.append(
            f"## {interval_start.strftime('%Y-%m-%d %H:%M')} - {interval_end.strftime('%H:%M')}"
        )
        lines.append("")

        for entry in groups[interval_start]:
            if entry["prefix"] == "audio":
                lines.append("### Audio Transcript")
                lines.append("```json")
                lines.append(entry["content"].strip())
                lines.append("```")
                lines.append("")
            elif entry["prefix"] == "screen":
                lines.append("### Screen Activity Summary")
                lines.append('"""')
                lines.append(entry["content"].strip())
                lines.append('"""')
                lines.append("")
            elif entry["prefix"] == "monitor":
                lines.append(
                    f"### Monitor {entry['monitor']} {entry['timestamp'].strftime('%H:%M:%S')}"
                )
                lines.append("```json")
                lines.append(entry["content"].strip())
                lines.append("```")
                lines.append("")

    return "\n".join(lines)


def cluster(day: str) -> Tuple[str, int]:
    """Return Markdown summary for one day's JSON files and the number processed."""

    day_dir = day_path(day)
    if not os.path.isdir(day_dir):
        return f"Day folder not found: {day_dir}", 0

    entries = _load_entries(day_dir, True, "summary")
    if not entries:
        return f"No audio or screen files found for date {day} in {day_dir}.", 0

    groups = _group_entries(entries)
    markdown = _groups_to_markdown(groups)
    return markdown, len(entries)


def cluster_range(
    day: str,
    start: str,
    end: str,
    audio: bool = True,
    screen: str = "summary",
) -> str:
    """Return markdown for ``day`` limited to ``start``-``end`` (HHMMSS)."""

    if screen not in {"summary", "raw"}:
        raise ValueError("screen must be 'summary' or 'raw'")

    day_dir = day_path(day)
    date_str = _date_str(day_dir)
    start_dt = datetime.strptime(date_str + start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(date_str + end, "%Y%m%d%H%M%S")

    entries = _load_entries(day_dir, audio, screen)
    entries = [e for e in entries if start_dt <= e["timestamp"] < end_dt]
    groups = _group_entries(entries)
    return _groups_to_markdown(groups)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown report for a day's JSON files grouped by 5-minute intervals."
    )
    parser.add_argument(
        "day",
        help="Day in YYYYMMDD format",
    )

    args = setup_cli(parser)

    markdown, _ = cluster(args.day)
    print(markdown)


if __name__ == "__main__":
    main()
