# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from observe.screen import format_screen_text

from .streams import read_segment_stream
from .utils import day_path


def _date_str(day_dir: str) -> str:
    base = os.path.basename(os.path.normpath(day_dir))
    if not re.fullmatch(r"\d{8}", base):
        raise ValueError("day_dir must end with YYYYMMDD")
    return base


def _filename_to_agent_key(filename: str) -> str:
    """Convert output filename stem to agent key.

    Reverse of get_output_topic(): converts filesystem names back to agent keys.

    Args:
        filename: Filename stem (e.g., "entities" or "_todos_review")

    Returns:
        Agent key (e.g., "entities" or "todos:review")
    """
    if filename.startswith("_"):
        # App agent: "_app_topic" -> "app:topic"
        parts = filename[1:].split("_", 1)
        if len(parts) == 2:
            return f"{parts[0]}:{parts[1]}"
    return filename


def _agent_matches_filter(
    filename: str, agent_filter: dict[str, bool | str] | None
) -> bool:
    """Check if an agent output file matches the filter.

    Args:
        filename: Filename stem (e.g., "entities" or "_todos_review")
        agent_filter: Dict mapping agent keys to bool/"required", or None for all

    Returns:
        True if the file should be included
    """
    if agent_filter is None:
        # None means include all agents
        return True

    if not agent_filter:
        # Empty dict means no agents
        return False

    agent_key = _filename_to_agent_key(filename)

    # Check if this agent is enabled in the filter
    if agent_key in agent_filter:
        value = agent_filter[agent_key]
        return value is True or value == "required"

    return False


def _process_segment(
    segment_path: Path,
    date_str: str,
    audio: bool,
    screen: bool,
    agents: bool | dict[str, bool | str],
) -> list[dict[str, Any]]:
    """Process a single segment directory and return entries.

    Args:
        segment_path: Path to segment directory
        date_str: Date in YYYYMMDD format
        audio: Whether to load audio transcripts
        screen: Whether to load raw screen data from *screen.jsonl files
        agents: Whether to load agent output summaries from *.md files.
            Can be bool (all/none) or dict for selective filtering
            (e.g., {"entities": True, "meetings": "required"}).

    Returns:
        List of entry dicts with timestamp, segment_key, prefix, content, name, etc.
    """
    from think.utils import segment_parse

    entries: list[dict[str, Any]] = []

    start_time, end_time = segment_parse(segment_path.name)
    if not start_time or not end_time:
        return entries

    # Read stream identity
    marker = read_segment_stream(segment_path)
    stream = marker.get("stream") if marker else None

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
                    "stream": stream,
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
                            "stream": stream,
                        }
                    )
            except Exception as e:  # pragma: no cover - warning only
                print(
                    f"Warning: Could not read JSONL file {screen_jsonl.name}: {e}",
                    file=sys.stderr,
                )

    # Process agent output summaries from agents/**/*.md files (with optional filtering)
    if agents:
        # Convert bool to filter: True -> None (all), False handled by outer if
        agent_filter = (
            None if agents is True else agents if isinstance(agents, dict) else None
        )
        agents_dir = segment_path / "agents"
        if agents_dir.is_dir():
            for md_file in sorted(agents_dir.rglob("*.md")):
                if not md_file.is_file():
                    continue

                # Check if this agent matches the filter
                if not _agent_matches_filter(md_file.stem, agent_filter):
                    continue

                try:
                    content = md_file.read_text()
                    if content.strip():
                        rel_md_path = md_file.relative_to(agents_dir).as_posix()
                        entries.append(
                            {
                                "timestamp": segment_start,
                                "segment_key": segment_key,
                                "segment_start": segment_start,
                                "segment_end": segment_end,
                                "prefix": "agent_output",
                                "output_name": md_file.stem,
                                "content": content,
                                "name": f"{segment_path.name}/agents/{rel_md_path}",
                                "stream": stream,
                            }
                        )
                except Exception as e:  # pragma: no cover - warning only
                    print(
                        f"Warning: Could not read file {md_file.name}: {e}",
                        file=sys.stderr,
                    )

    return entries


def _load_entries(
    day_dir: str, audio: bool, screen: bool, agents: bool | dict[str, bool | str]
) -> list[dict[str, Any]]:
    """Load all transcript entries from a day directory."""
    from think.utils import segment_parse

    date_str = _date_str(day_dir)
    entries: list[dict[str, Any]] = []
    day_path_obj = Path(day_dir)

    from think.utils import iter_segments

    for _stream, _seg_key, seg_path in iter_segments(day_path_obj):
        start_time, _ = segment_parse(seg_path.name)
        if not start_time:
            continue
        entries.extend(_process_segment(seg_path, date_str, audio, screen, agents))

    entries.sort(key=lambda e: e["timestamp"])
    return entries


def _group_entries(
    entries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group entries by segment key.

    Returns dict mapping segment_key to list of entries for that segment.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        grouped[e["segment_key"]].append(e)
    return grouped


def _count_by_source(entries: list[dict[str, Any]]) -> dict[str, int]:
    """Count entries by source type (prefix).

    Maps the internal prefix names to source config names:
    - "audio" -> "audio"
    - "screen" -> "screen"
    - "agent_output" -> "agents"

    Returns:
        Dict with counts for each source type, e.g., {"audio": 2, "screen": 1, "agents": 0}
    """
    # Map internal prefix to source config name
    prefix_to_source = {
        "audio": "audio",
        "screen": "screen",
        "agent_output": "agents",
    }

    counts = Counter(prefix_to_source.get(e["prefix"], e["prefix"]) for e in entries)

    # Ensure all standard sources are present (even if 0)
    return {
        "audio": counts.get("audio", 0),
        "screen": counts.get("screen", 0),
        "agents": counts.get("agents", 0),
    }


def _groups_to_markdown(groups: dict[str, list[dict[str, Any]]]) -> str:
    """Render grouped entries as markdown with segment-based headers."""
    lines: list[str] = []

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


def _slots_to_ranges(slots: list[datetime]) -> list[tuple[str, str]]:
    """Collapse 15-minute slots into start/end pairs.

    Args:
        slots: Sorted list of datetimes marking 15-minute interval starts.

    Returns:
        List of (start, end) time strings in ``HH:MM`` format representing
        contiguous 15-minute ranges.
    """

    ranges: list[tuple[str, str]] = []
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


def cluster_scan(day: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
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
    audio_slots: set[datetime] = set()
    screen_slots: set[datetime] = set()
    day_path_obj = Path(day_dir)

    # Check timestamp subdirectories for transcript files
    from think.utils import iter_segments, segment_parse

    for _stream, _seg_key, seg_path in iter_segments(day_path_obj):
        start_time, _ = segment_parse(seg_path.name)
        if start_time:
            # Found segment - combine with date to get datetime
            day_date = datetime.strptime(date_str, "%Y%m%d").date()
            dt = datetime.combine(day_date, start_time)
            slot = dt.replace(
                minute=dt.minute - (dt.minute % 15), second=0, microsecond=0
            )

            # Check for audio transcripts
            if (seg_path / "audio.jsonl").exists() or any(
                seg_path.glob("*_audio.jsonl")
            ):
                audio_slots.add(slot)

            # Check for screen content
            if (seg_path / "screen.jsonl").exists() or any(
                seg_path.glob("*_screen.jsonl")
            ):
                screen_slots.add(slot)

    audio_ranges = _slots_to_ranges(sorted(audio_slots))
    screen_ranges = _slots_to_ranges(sorted(screen_slots))
    return audio_ranges, screen_ranges


def cluster_segments(day: str) -> list[dict[str, Any]]:
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

    from think.utils import iter_segments

    day_path_obj = Path(day_dir)
    segments: list[dict[str, Any]] = []

    for stream_name, seg_key, seg_path in iter_segments(day_path_obj):
        start_time, end_time = segment_parse(seg_path.name)
        if not (start_time and end_time):
            continue

        types = []
        # Check for audio transcripts
        if (seg_path / "audio.jsonl").exists() or any(seg_path.glob("*_audio.jsonl")):
            types.append("audio")

        # Check for screen content
        if (seg_path / "screen.jsonl").exists() or any(
            seg_path.glob("*_screen.jsonl")
        ):
            types.append("screen")

        if not types:
            continue

        start_str = start_time.strftime("%H:%M")
        end_str = end_time.strftime("%H:%M")

        marker = read_segment_stream(seg_path)
        segments.append(
            {
                "key": seg_path.name,
                "start": start_str,
                "end": end_str,
                "types": types,
                "stream": marker.get("stream") if marker else None,
            }
        )

    # Sort by start time
    segments.sort(key=lambda s: s["start"])
    return segments


def _find_segment_dir(day: str, segment: str, stream: str | None) -> Path | None:
    """Locate a segment directory, optionally searching across streams.

    Args:
        day: Day in YYYYMMDD format
        segment: Segment key in HHMMSS_LEN format
        stream: Stream name. If None, searches all streams under the day.

    Returns:
        Path to the segment directory, or None if not found.
    """
    from think.utils import segment_path as _segment_path

    if stream:
        path = _segment_path(day, segment, stream)
        return path if path.is_dir() else None

    # Search all streams for this segment
    from think.utils import iter_segments

    for _s, _k, seg_path in iter_segments(day):
        if seg_path.name == segment:
            return seg_path
    return None


def cluster(
    day: str,
    sources: dict[str, bool | str | dict],
) -> tuple[str, dict[str, int]]:
    """Return Markdown summary for one day's JSON files and counts by source.

    Args:
        day: Day in YYYYMMDD format
        sources: Dict with keys "audio", "screen", "agents".
            Values can be bool, "required" string, or dict (for agents).
            The "agents" source can be a dict for selective filtering,
            e.g., {"entities": True, "meetings": "required"}.

    Returns:
        Tuple of (markdown, source_counts) where source_counts is a dict
        with keys "audio", "screen", "agents" mapping to entry counts.
    """
    empty_counts = {"audio": 0, "screen": 0, "agents": 0}

    day_dir = str(day_path(day))
    # day_path now ensures dir exists, but check anyway for safety
    if not os.path.isdir(day_dir):
        return f"Day folder not found: {day_dir}", empty_counts

    entries = _load_entries(
        day_dir,
        audio=sources.get("audio", False),
        screen=sources.get("screen", False),
        agents=sources.get("agents", False),
    )
    if not entries:
        return (
            f"No audio or screen files found for date {day} in {day_dir}.",
            empty_counts,
        )

    groups = _group_entries(entries)
    markdown = _groups_to_markdown(groups)
    return markdown, _count_by_source(entries)


def cluster_period(
    day: str,
    segment: str,
    sources: dict[str, bool | str | dict],
    stream: str | None = None,
) -> tuple[str, dict[str, int]]:
    """Return Markdown summary for one segment's JSON files and counts by source.

    Args:
        day: Day in YYYYMMDD format
        segment: Segment key in HHMMSS_LEN format (e.g., "163045_300")
        sources: Dict with keys "audio", "screen", "agents".
            Values can be bool, "required" string, or dict (for agents).
        stream: Stream name. If None, searches all streams for the segment.

    Returns:
        Tuple of (markdown, source_counts) where source_counts is a dict
        with keys "audio", "screen", "agents" mapping to entry counts.
    """
    empty_counts = {"audio": 0, "screen": 0, "agents": 0}

    segment_dir = _find_segment_dir(day, segment, stream)

    if segment_dir is None or not segment_dir.is_dir():
        return f"Segment folder not found: {day}/{segment}", empty_counts

    entries = _load_entries_from_segment(
        str(segment_dir),
        audio=sources.get("audio", False),
        screen=sources.get("screen", False),
        agents=sources.get("agents", False),
    )
    if not entries:
        return f"No audio or screen files found for segment {segment}", empty_counts

    groups = _group_entries(entries)
    markdown = _groups_to_markdown(groups)
    return markdown, _count_by_source(entries)


def _load_entries_from_segment(
    segment_dir: str, audio: bool, screen: bool, agents: bool | dict[str, bool | str]
) -> list[dict[str, Any]]:
    """Load entries from a single segment directory.

    Args:
        segment_dir: Path to segment directory (e.g., /path/to/20251109/163045_300)
        audio: Whether to load audio transcripts
        screen: Whether to load raw screen data from *screen.jsonl files
        agents: Whether to load agent output summaries from *.md files

    Returns:
        List of entry dicts with timestamp, prefix, content, etc.
    """
    segment_path_obj = Path(segment_dir)
    # Parent is stream dir; grandparent is day dir
    date_str = _date_str(str(segment_path_obj.parent.parent))
    entries = _process_segment(segment_path_obj, date_str, audio, screen, agents)
    entries.sort(key=lambda e: e["timestamp"])
    return entries


def cluster_span(
    day: str,
    span: list[str],
    sources: dict[str, bool | str | dict],
    stream: str | None = None,
) -> tuple[str, dict[str, int]]:
    """Return Markdown summary for a span of segments and counts by source.

    A span is a list of sequential segment keys (e.g., from an import that created
    multiple 5-minute segments from one audio file).

    Validates all segments exist before processing; raises ValueError if any are missing.

    Args:
        day: Day in YYYYMMDD format
        span: List of segment keys in HHMMSS_LEN format (e.g., ["163045_300", "170000_600"])
        sources: Dict with keys "audio", "screen", "agents".
            Values can be bool, "required" string, or dict (for agents).
        stream: Stream name. If None, searches all streams for each segment.

    Returns:
        Tuple of (markdown, source_counts) where source_counts is a dict
        with keys "audio", "screen", "agents" mapping to entry counts.

    Raises:
        ValueError: If any segment directories are missing
    """
    empty_counts = {"audio": 0, "screen": 0, "agents": 0}

    # Validate all segments in span exist upfront (fail fast)
    missing = []
    found_dirs: list[Path] = []
    for seg_key in span:
        seg_dir = _find_segment_dir(day, seg_key, stream)
        if seg_dir is None:
            missing.append(seg_key)
        else:
            found_dirs.append(seg_dir)

    if missing:
        raise ValueError(f"Segment directories not found: {', '.join(missing)}")

    # Load entries from all segments in span
    entries: list[dict[str, Any]] = []
    for seg_dir in found_dirs:
        segment_entries = _load_entries_from_segment(
            str(seg_dir),
            audio=sources.get("audio", False),
            screen=sources.get("screen", False),
            agents=sources.get("agents", False),
        )
        entries.extend(segment_entries)

    if not entries:
        return (
            f"No audio or screen files found in span: {', '.join(span)}",
            empty_counts,
        )

    # Sort all entries by timestamp, group, and render
    entries.sort(key=lambda e: e["timestamp"])
    groups = _group_entries(entries)
    markdown = _groups_to_markdown(groups)
    return markdown, _count_by_source(entries)


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
    sources: dict[str, bool | str | dict],
) -> str:
    """Return markdown for ``day`` limited to ``start``-``end`` (HHMMSS).

    Includes any segment that overlaps with the requested time range,
    even if only partially.

    Args:
        day: Day in YYYYMMDD format
        start: Start time in HHMMSS format
        end: End time in HHMMSS format
        sources: Dict with keys "audio", "screen", "agents".
            Values can be bool, "required" string, or dict (for agents).
    """
    day_dir = str(day_path(day))
    date_str = _date_str(day_dir)
    start_dt = datetime.strptime(date_str + start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(date_str + end, "%Y%m%d%H%M%S")

    entries = _load_entries(
        day_dir,
        audio=sources.get("audio", False),
        screen=sources.get("screen", False),
        agents=sources.get("agents", False),
    )
    # Include segments that overlap with the requested range
    entries = [
        e
        for e in entries
        if _segments_overlap(e["segment_start"], e["segment_end"], start_dt, end_dt)
    ]
    groups = _group_entries(entries)
    return _groups_to_markdown(groups)
