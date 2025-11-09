import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from observe.utils import load_analysis_frames

from .utils import day_path, setup_cli


def _date_str(day_dir: str) -> str:
    base = os.path.basename(os.path.normpath(day_dir))
    if not re.fullmatch(r"\d{8}", base):
        raise ValueError("day_dir must end with YYYYMMDD")
    return base


def _load_entries(
    day_dir: str, audio: bool, screen_mode: Optional[str]
) -> List[Dict[str, str]]:
    date_str = _date_str(day_dir)
    entries: List[Dict[str, str]] = []
    day_path_obj = Path(day_dir)

    # Check timestamp subdirectories for transcript files
    from think.utils import period_name

    for item in day_path_obj.iterdir():
        time_part = period_name(item.name)
        if not (item.is_dir() and time_part):
            continue

        # Process audio transcripts
        if audio:
            # Check for audio.jsonl or split audio files
            audio_files = [f for f in item.glob("*audio.jsonl") if f.is_file()]
            for audio_file in audio_files:
                # Load JSONL format using shared utility
                from observe.hear import load_transcript

                metadata, transcript_entries, formatted_text = load_transcript(str(audio_file))
                if transcript_entries is None:
                    print(
                        f"Warning: Could not load transcript {audio_file.name}: {metadata.get('error')}",
                        file=sys.stderr,
                    )
                    continue

                # Use formatted text for human-readable display
                timestamp = datetime.strptime(date_str + time_part, "%Y%m%d%H%M%S")
                entries.append(
                    {
                        "timestamp": timestamp,
                        "prefix": "audio",
                        "content": formatted_text,
                        "monitor": None,
                        "source": None,
                        "id": None,
                        "name": f"{time_part}/{audio_file.name}",
                    }
                )

        # Process screen summaries or transcripts
        if screen_mode == "summary":
            screen_md = item / "screen.md"
            if screen_md.exists():
                try:
                    content = screen_md.read_text()
                    timestamp = datetime.strptime(date_str + time_part, "%Y%m%d%H%M%S")
                    entries.append(
                        {
                            "timestamp": timestamp,
                            "prefix": "screen",
                            "content": content,
                            "monitor": None,
                            "source": None,
                            "id": None,
                            "name": f"{time_part}/screen.md",
                        }
                    )
                except Exception as e:
                    print(f"Warning: Could not read file screen.md: {e}", file=sys.stderr)

        elif screen_mode == "raw":
            # Check for screen.jsonl
            screen_jsonl = item / "screen.jsonl"
            if screen_jsonl.exists():
                try:
                    # Use shared loading logic from observe module
                    frames = load_analysis_frames(screen_jsonl)

                    # Sort frames by timestamp
                    frames.sort(key=lambda f: f.get("timestamp", 0))

                    # Create one entry with all frames as JSON content
                    if frames:
                        base_timestamp = datetime.strptime(
                            date_str + time_part, "%Y%m%d%H%M%S"
                        )
                        entries.append(
                            {
                                "timestamp": base_timestamp,
                                "prefix": "screen_jsonl",
                                "content": json.dumps(frames, indent=2),
                                "monitor": None,
                                "source": None,
                                "id": None,
                                "name": f"{time_part}/screen.jsonl",
                            }
                        )
                except Exception as e:  # pragma: no cover - warning only
                    print(
                        f"Warning: Could not read JSONL file screen.jsonl: {e}",
                        file=sys.stderr,
                    )

    entries.sort(key=lambda e: e["timestamp"])
    return entries


def _group_entries(
    entries: List[Dict[str, str]],
) -> Dict[datetime, List[Dict[str, str]]]:
    grouped: Dict[datetime, List[Dict[str, str]]] = defaultdict(list)
    for e in entries:
        ts = e["timestamp"]
        interval = ts.replace(
            minute=ts.minute - (ts.minute % 5), second=0, microsecond=0
        )
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
                lines.append(entry["content"].strip())
                lines.append("")
            elif entry["prefix"] == "screen":
                lines.append("### Screen Activity Summary")
                lines.append('"""')
                lines.append(entry["content"].strip())
                lines.append('"""')
                lines.append("")
            elif entry["prefix"] == "screen_jsonl":
                lines.append("### Screen Activity (JSONL)")
                lines.append("```json")
                lines.append(entry["content"].strip())
                lines.append("```")
                lines.append("")
            else:
                src = entry.get("source") or entry["prefix"]
                ident = entry.get("id") or entry.get("monitor")
                title = (
                    f"{src.capitalize()} {ident}"
                    if ident is not None
                    else src.capitalize()
                )
                lines.append(f"### {title} {entry['timestamp'].strftime('%H:%M:%S')}")
                lines.append("```json")
                lines.append(entry["content"].strip())
                lines.append("```")
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
    from think.utils import period_name

    for item in day_path_obj.iterdir():
        time_part = period_name(item.name)
        if item.is_dir() and time_part:
            # Found period (HHMMSS)
            dt = datetime.strptime(date_str + time_part, "%Y%m%d%H%M%S")
            slot = dt.replace(
                minute=dt.minute - (dt.minute % 15), second=0, microsecond=0
            )

            # Check for audio transcripts
            if (item / "audio.jsonl").exists() or any(item.glob("*_audio.jsonl")):
                audio_slots.add(slot)

            # Check for screen summaries or transcripts
            if (item / "screen.md").exists() or (item / "screen.jsonl").exists():
                screen_slots.add(slot)

    audio_ranges = _slots_to_ranges(sorted(audio_slots))
    screen_ranges = _slots_to_ranges(sorted(screen_slots))
    return audio_ranges, screen_ranges


def cluster(day: str) -> Tuple[str, int]:
    """Return Markdown summary for one day's JSON files and the number processed."""

    day_dir = str(day_path(day))
    # day_path now ensures dir exists, but check anyway for safety
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
    screen: Optional[str] = "summary",
) -> str:
    """Return markdown for ``day`` limited to ``start``-``end`` (HHMMSS)."""

    if screen is not None and screen not in {"summary", "raw"}:
        raise ValueError("screen must be 'summary', 'raw', or None")

    day_dir = str(day_path(day))
    date_str = _date_str(day_dir)
    start_dt = datetime.strptime(date_str + start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(date_str + end, "%Y%m%d%H%M%S")

    entries = _load_entries(day_dir, audio, screen)
    entries = [e for e in entries if start_dt <= e["timestamp"] < end_dt]
    groups = _group_entries(entries)
    return _groups_to_markdown(groups)


def cluster_files(
    day: str, start: str, end: str, type: Optional[str] = None
) -> List[str]:
    """Return raw audio and screen transcript contents for ``day`` in ``start``-``end`` (HHMMSS).

    Args:
        day: Day in YYYYMMDD format
        start: Start time in HHMMSS format
        end: End time in HHMMSS format
        type: Filter by content type - 'audio', 'screen', or None for both

    Returns:
        List of strings containing the raw JSON/markdown content for each matching file,
        ordered chronologically.
    """
    if type is not None and type not in {"audio", "screen"}:
        raise ValueError("type must be 'audio', 'screen', or None")

    day_dir = str(day_path(day))
    date_str = _date_str(day_dir)
    start_dt = datetime.strptime(date_str + start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(date_str + end, "%Y%m%d%H%M%S")

    # Determine loading parameters based on type filter
    if type == "audio":
        entries = _load_entries(day_dir, audio=True, screen_mode=None)
    elif type == "screen":
        entries = _load_entries(day_dir, audio=False, screen_mode="raw")
    else:  # type is None - load both
        entries = _load_entries(day_dir, audio=True, screen_mode="raw")

    # Filter to the requested window and return the raw contents.
    return [
        e["content"].strip() for e in entries if start_dt <= e["timestamp"] < end_dt
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown report for a day's JSON files grouped by 5-minute intervals."
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
            screen="summary",
        )
        print(markdown)
    elif args.start or args.length is not None:
        parser.error("--start and --length must be used together")
    else:
        markdown, _ = cluster(args.day)
        print(markdown)


if __name__ == "__main__":
    main()
