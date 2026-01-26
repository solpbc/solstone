# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from observe.screen import format_screen_text

from .utils import day_path, setup_cli


def _date_str(day_dir: str) -> str:
    base = os.path.basename(os.path.normpath(day_dir))
    if not re.fullmatch(r"\d{8}", base):
        raise ValueError("day_dir must end with YYYYMMDD")
    return base


def _process_segment(
    segment_path: Path,
    date_str: str,
    audio: bool,
    screen: bool,
    agents: bool,
) -> List[Dict[str, Any]]:
    """Process a single segment directory and return entries.

    Args:
        segment_path: Path to segment directory
        date_str: Date in YYYYMMDD format
        audio: Whether to load audio transcripts
        screen: Whether to load raw screen data from *screen.jsonl files
        agents: Whether to load agent output summaries from *.md files

    Returns:
        List of entry dicts with timestamp, segment_key, prefix, content, name, etc.
    """
    from think.utils import segment_parse

    entries: List[Dict[str, Any]] = []

    start_time, end_time = segment_parse(segment_path.name)
    if not start_time or not end_time:
        return entries

    # Compute segment times
    segment_key = segment_path.name
    day_date = datetime.strptime(date_str, "%Y%m%d").date()
    segment_start = datetime.combine(day_date, start_time)
    segment_end = datetime.combine(day_date, end_time)

    # Process audio transcripts
    if audio:
        audio_files = [f for f in segment_path.glob("*audio.jsonl") if f.is_file()]
        for audio_file in audio_files:
            from observe.hear import load_transcript

            metadata, transcript_entries, formatted_text = load_transcript(
                str(audio_file)
            )
            if transcript_entries is None:
                print(
                    f"Warning: Could not load transcript {audio_file.name}: {metadata.get('error')}",
                    file=sys.stderr,
                )
                continue

            entries.append(
                {
                    "timestamp": segment_start,
                    "segment_key": segment_key,
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "prefix": "audio",
                    "content": formatted_text,
                    "name": f"{segment_path.name}/{audio_file.name}",
                }
            )

    # Process raw screen data from screen.jsonl and *_screen.jsonl
    if screen:
        screen_files = list(segment_path.glob("*screen.jsonl"))
        for screen_jsonl in screen_files:
            try:
                content = format_screen_text(screen_jsonl)
                if content:
                    entries.append(
                        {
                            "timestamp": segment_start,
                            "segment_key": segment_key,
                            "segment_start": segment_start,
                            "segment_end": segment_end,
                            "prefix": "screen",
                            "content": content,
                            "name": f"{segment_path.name}/{screen_jsonl.name}",
                        }
                    )
            except Exception as e:  # pragma: no cover - warning only
                print(
                    f"Warning: Could not read JSONL file {screen_jsonl.name}: {e}",
                    file=sys.stderr,
                )

    # Process agent output summaries from all *.md files
    if agents:
        for md_file in sorted(segment_path.glob("*.md")):
            if not md_file.is_file():
                continue
            try:
                content = md_file.read_text()
                if content.strip():
                    entries.append(
                        {
                            "timestamp": segment_start,
                            "segment_key": segment_key,
                            "segment_start": segment_start,
                            "segment_end": segment_end,
                            "prefix": "agent_output",
                            "output_name": md_file.stem,
                            "content": content,
                            "name": f"{segment_path.name}/{md_file.name}",
                        }
                    )
            except Exception as e:  # pragma: no cover - warning only
                print(
                    f"Warning: Could not read file {md_file.name}: {e}",
                    file=sys.stderr,
                )

    return entries


def _load_entries(
    day_dir: str, audio: bool, screen: bool, agents: bool
) -> List[Dict[str, Any]]:
    """Load all transcript entries from a day directory."""
    from think.utils import segment_parse

    date_str = _date_str(day_dir)
    entries: List[Dict[str, Any]] = []
    day_path_obj = Path(day_dir)

    for item in day_path_obj.iterdir():
        start_time, _ = segment_parse(item.name)
        if not (item.is_dir() and start_time):
            continue
        entries.extend(_process_segment(item, date_str, audio, screen, agents))

    entries.sort(key=lambda e: e["timestamp"])
    return entries


def _group_entries(
    entries: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group entries by segment key.

    Returns dict mapping segment_key to list of entries for that segment.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        grouped[e["segment_key"]].append(e)
    return grouped


def _groups_to_markdown(groups: Dict[str, List[Dict[str, Any]]]) -> str:
    """Render grouped entries as markdown with segment-based headers."""
    lines: List[str] = []

    # Sort by segment start time (entries within each group have same segment_start)
    def sort_key(segment_key: str) -> datetime:
        entries = groups[segment_key]
        return entries[0]["segment_start"] if entries else datetime.min

    for segment_key in sorted(groups.keys(), key=sort_key):
        segment_entries = groups[segment_key]
        if not segment_entries:
            continue

        # Use segment times from first entry (all entries in group share same segment)
        segment_start = segment_entries[0]["segment_start"]
        segment_end = segment_entries[0]["segment_end"]
        lines.append(
            f"## {segment_start.strftime('%Y-%m-%d %H:%M:%S')} - {segment_end.strftime('%H:%M:%S')}"
        )
        lines.append("")

        for entry in segment_entries:
            if entry["prefix"] == "audio":
                lines.append("### Audio Transcript")
                lines.append(entry["content"].strip())
                lines.append("")
            elif entry["prefix"] == "screen":
                lines.append("### Screen Activity")
                lines.append(entry["content"].strip())
                lines.append("")
            elif entry["prefix"] == "agent_output":
                output_name = entry.get("output_name", "output")
                lines.append(f"### {output_name} summary")
                lines.append(entry["content"].strip())
                lines.append("")

    return "\n".join(lines)


def _slots_to_ranges(slots: List[datetime]) -> List[Tuple[str, str]]:
    """Collapse 15-minute slots into start/end pairs.

    Args:
        slots: Sorted list of datetimes marking 15-minute interval starts.

    Returns:
        List of (start, end) time strings in ``HH:MM`` format representing
        contiguous 15-minute ranges.
    """

    ranges: List[Tuple[str, str]] = []
    if not slots:
        return ranges

    start = slots[0]
    prev = slots[0]
    for current in slots[1:]:
        if current - prev == timedelta(minutes=15):
            prev = current
            continue
        ranges.append(
            (start.strftime("%H:%M"), (prev + timedelta(minutes=15)).strftime("%H:%M"))
        )
        start = prev = current

    ranges.append(
        (start.strftime("%H:%M"), (prev + timedelta(minutes=15)).strftime("%H:%M"))
    )
    return ranges


def cluster_scan(day: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Return 15-minute ranges with audio and screen transcripts for ``day``.

    Args:
        day: Day folder in ``YYYYMMDD`` format.

    Returns:
        Two lists containing ``(start, end)`` pairs (``HH:MM``) for audio and
        screen transcripts respectively.
    """

    day_dir = str(day_path(day))
    # day_path now ensures dir exists
    if not os.path.isdir(day_dir):
        return [], []

    date_str = _date_str(day_dir)
    audio_slots: Set[datetime] = set()
    screen_slots: Set[datetime] = set()
    day_path_obj = Path(day_dir)

    # Check timestamp subdirectories for transcript files
    from think.utils import segment_parse

    for item in day_path_obj.iterdir():
        start_time, _ = segment_parse(item.name)
        if item.is_dir() and start_time:
            # Found segment - combine with date to get datetime
            day_date = datetime.strptime(date_str, "%Y%m%d").date()
            dt = datetime.combine(day_date, start_time)
            slot = dt.replace(
                minute=dt.minute - (dt.minute % 15), second=0, microsecond=0
            )

            # Check for audio transcripts
            if (item / "audio.jsonl").exists() or any(item.glob("*_audio.jsonl")):
                audio_slots.add(slot)

            # Check for screen content
            if (item / "screen.jsonl").exists() or any(item.glob("*_screen.jsonl")):
                screen_slots.add(slot)

    audio_ranges = _slots_to_ranges(sorted(audio_slots))
    screen_ranges = _slots_to_ranges(sorted(screen_slots))
    return audio_ranges, screen_ranges


def cluster_segments(day: str) -> List[Dict[str, any]]:
    """Return individual recording segments for a day with their content types.

    Unlike ``cluster_scan()`` which collapses segments into 15-minute ranges,
    this returns actual segment directories with their precise times.

    Args:
        day: Day folder in ``YYYYMMDD`` format.

    Returns:
        List of dicts with segment info:
        - key: segment directory name (HHMMSS_LEN format)
        - start: start time as HH:MM
        - end: end time as HH:MM
        - types: list of content types present ("audio", "screen", or both)
    """
    from think.utils import segment_parse

    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return []

    day_path_obj = Path(day_dir)
    segments: List[Dict[str, any]] = []

    for item in day_path_obj.iterdir():
        start_time, end_time = segment_parse(item.name)
        if not (item.is_dir() and start_time and end_time):
            continue

        types = []
        # Check for audio transcripts
        if (item / "audio.jsonl").exists() or any(item.glob("*_audio.jsonl")):
            types.append("audio")

        # Check for screen content
        if (item / "screen.jsonl").exists() or any(item.glob("*_screen.jsonl")):
            types.append("screen")

        if not types:
            continue

        start_str = start_time.strftime("%H:%M")
        end_str = end_time.strftime("%H:%M")

        segments.append(
            {
                "key": item.name,
                "start": start_str,
                "end": end_str,
                "types": types,
            }
        )

    # Sort by start time
    segments.sort(key=lambda s: s["start"])
    return segments


def cluster(
    day: str,
    sources: Dict[str, bool] | None = None,
) -> Tuple[str, int]:
    """Return Markdown summary for one day's JSON files and the number processed.

    By default uses insight summaries (*.md files) rather than raw screen data
    for daily view. Override with sources parameter.

    Args:
        day: Day in YYYYMMDD format
        sources: Optional dict with keys "audio", "screen", "agents" (bools).
            Defaults to {"audio": True, "screen": False, "agents": True}.
    """
    # Default sources for daily insights: audio + insight summaries, no raw screen
    if sources is None:
        sources = {"audio": True, "screen": False, "agents": True}

    day_dir = str(day_path(day))
    # day_path now ensures dir exists, but check anyway for safety
    if not os.path.isdir(day_dir):
        return f"Day folder not found: {day_dir}", 0

    entries = _load_entries(
        day_dir,
        audio=sources.get("audio", True),
        screen=sources.get("screen", False),
        agents=sources.get("agents", True),
    )
    if not entries:
        return f"No audio or screen files found for date {day} in {day_dir}.", 0

    groups = _group_entries(entries)
    markdown = _groups_to_markdown(groups)
    return markdown, len(entries)


def cluster_period(
    day: str,
    segment: str,
    sources: Dict[str, bool] | None = None,
) -> Tuple[str, int]:
    """Return Markdown summary for one segment's JSON files and the number processed.

    By default uses raw screen data for segment insights (more granular than summaries).
    Override with sources parameter.

    Args:
        day: Day in YYYYMMDD format
        segment: Segment key in HHMMSS_LEN format (e.g., "163045_300")
        sources: Optional dict with keys "audio", "screen", "agents" (bools).
            Defaults to {"audio": True, "screen": True, "agents": False}.

    Returns:
        (markdown, file_count) tuple
    """
    # Default sources for segment insights: audio + raw screen, no insight summaries
    if sources is None:
        sources = {"audio": True, "screen": True, "agents": False}

    day_dir = str(day_path(day))
    segment_dir = Path(day_dir) / segment

    if not segment_dir.is_dir():
        return f"Segment folder not found: {segment_dir}", 0

    entries = _load_entries_from_segment(
        str(segment_dir),
        audio=sources.get("audio", True),
        screen=sources.get("screen", True),
        agents=sources.get("agents", False),
    )
    if not entries:
        return f"No audio or screen files found for segment {segment}", 0

    groups = _group_entries(entries)
    markdown = _groups_to_markdown(groups)
    return markdown, len(entries)


def _load_entries_from_segment(
    segment_dir: str, audio: bool, screen: bool, agents: bool
) -> List[Dict[str, Any]]:
    """Load entries from a single segment directory.

    Args:
        segment_dir: Path to segment directory (e.g., /path/to/20251109/163045_300)
        audio: Whether to load audio transcripts
        screen: Whether to load raw screen data from *screen.jsonl files
        agents: Whether to load agent output summaries from *.md files

    Returns:
        List of entry dicts with timestamp, prefix, content, etc.
    """
    segment_path = Path(segment_dir)
    date_str = _date_str(str(segment_path.parent))
    entries = _process_segment(segment_path, date_str, audio, screen, agents)
    entries.sort(key=lambda e: e["timestamp"])
    return entries


def cluster_segments_multi(
    day: str,
    segments: List[str],
    sources: Dict[str, bool] | None = None,
) -> Tuple[str, int]:
    """Return Markdown summary for multiple segments and the number of entries processed.

    By default uses raw screen data for segment insights (more granular than summaries).
    Validates all segments exist before processing; raises ValueError if any are missing.

    Args:
        day: Day in YYYYMMDD format
        segments: List of segment keys in HHMMSS_LEN format (e.g., ["163045_300", "170000_600"])
        sources: Optional dict with keys "audio", "screen", "agents" (bools).
            Defaults to {"audio": True, "screen": True, "agents": False}.

    Returns:
        (markdown, file_count) tuple

    Raises:
        ValueError: If any segment directories are missing
    """
    # Default sources for segment insights: audio + raw screen, no insight summaries
    if sources is None:
        sources = {"audio": True, "screen": True, "agents": False}

    day_dir = str(day_path(day))

    # Validate all segments exist upfront (fail fast)
    missing = []
    for segment in segments:
        segment_dir = Path(day_dir) / segment
        if not segment_dir.is_dir():
            missing.append(segment)

    if missing:
        raise ValueError(f"Segment directories not found: {', '.join(missing)}")

    # Load entries from all segments
    entries: List[Dict[str, Any]] = []
    for segment in segments:
        segment_dir = Path(day_dir) / segment
        segment_entries = _load_entries_from_segment(
            str(segment_dir),
            audio=sources.get("audio", True),
            screen=sources.get("screen", True),
            agents=sources.get("agents", False),
        )
        entries.extend(segment_entries)

    if not entries:
        return f"No audio or screen files found in segments: {', '.join(segments)}", 0

    # Sort all entries by timestamp, group, and render
    entries.sort(key=lambda e: e["timestamp"])
    groups = _group_entries(entries)
    markdown = _groups_to_markdown(groups)
    return markdown, len(entries)


def _segments_overlap(
    seg_start: datetime, seg_end: datetime, range_start: datetime, range_end: datetime
) -> bool:
    """Check if a segment overlaps with a time range.

    Returns True if any part of the segment falls within the range.
    """
    return seg_start < range_end and seg_end > range_start


def cluster_range(
    day: str,
    start: str,
    end: str,
    audio: bool = True,
    screen: bool = False,
    agents: bool = True,
) -> str:
    """Return markdown for ``day`` limited to ``start``-``end`` (HHMMSS).

    Includes any segment that overlaps with the requested time range,
    even if only partially.

    Args:
        day: Day in YYYYMMDD format
        start: Start time in HHMMSS format
        end: End time in HHMMSS format
        audio: Whether to include audio transcripts
        screen: Whether to include raw screen data from *screen.jsonl files
        agents: Whether to include agent output summaries from *.md files
    """

    day_dir = str(day_path(day))
    date_str = _date_str(day_dir)
    start_dt = datetime.strptime(date_str + start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(date_str + end, "%Y%m%d%H%M%S")

    entries = _load_entries(day_dir, audio, screen, agents)
    # Include segments that overlap with the requested range
    entries = [
        e
        for e in entries
        if _segments_overlap(e["segment_start"], e["segment_end"], start_dt, end_dt)
    ]
    groups = _group_entries(entries)
    return _groups_to_markdown(groups)


def get_entries_for_range(
    day: str,
    start: str,
    end: str,
    audio: bool = True,
    screen: bool = True,
    agents: bool = False,
) -> List[Dict[str, Any]]:
    """Return filtered transcript entries for a time range.

    Public API for routes/tools that need raw entry data (not markdown).
    Returns entries with metadata for further processing (e.g., media file lookup).

    Args:
        day: Day in YYYYMMDD format
        start: Start time in HHMMSS format
        end: End time in HHMMSS format
        audio: Whether to include audio transcripts
        screen: Whether to include raw screen data from *screen.jsonl files
        agents: Whether to include agent output summaries from *.md files

    Returns:
        List of entry dicts with keys:
        - timestamp: datetime of segment start
        - segment_key: segment directory name
        - segment_start: datetime of segment start
        - segment_end: datetime of segment end
        - prefix: "audio", "screen", or "agent_output"
        - output_name: (agents only) stem of the .md filename
        - content: formatted transcript text
        - name: relative path like "HHMMSS_LEN/audio.jsonl"
    """

    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return []

    date_str = _date_str(day_dir)
    start_dt = datetime.strptime(date_str + start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(date_str + end, "%Y%m%d%H%M%S")

    entries = _load_entries(day_dir, audio, screen, agents)
    return [
        e
        for e in entries
        if _segments_overlap(e["segment_start"], e["segment_end"], start_dt, end_dt)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown report for a day's JSON files grouped by recording segments."
    )
    parser.add_argument(
        "day",
        help="Day in YYYYMMDD format",
    )
    parser.add_argument(
        "--start",
        metavar="HHMMSS",
        help="Start time for range (HHMMSS)",
    )
    parser.add_argument(
        "--length",
        type=int,
        help="Length of range in minutes",
    )

    args = setup_cli(parser)

    if args.start and args.length is not None:
        start_dt = datetime.strptime(args.start, "%H%M%S")
        end_dt = start_dt + timedelta(minutes=args.length)
        markdown = cluster_range(
            args.day,
            args.start,
            end_dt.strftime("%H%M%S"),
            audio=True,
            screen=True,
            agents=False,
        )
        print(markdown)
    elif args.start or args.length is not None:
        parser.error("--start and --length must be used together")
    else:
        markdown, _ = cluster(args.day)
        print(markdown)


if __name__ == "__main__":
    main()
