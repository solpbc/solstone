"""Transcript indexing and search functionality."""

import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List

import sqlite_utils

from think.utils import day_dirs

from .core import _scan_files, get_index

# Transcript file helpers
AUDIO_RE = re.compile(r"^(?P<time>\d{6}).*_audio\.json$")
SCREEN_RE = re.compile(r"^(?P<time>\d{6})_[a-z]+_\d+_diff\.json$")
SCREEN_JSONL_RE = re.compile(r"^(?P<time>\d{6})_screen\.jsonl$")


def find_transcript_files(journal: str) -> Dict[str, str]:
    """Return mapping of transcript JSON file paths relative to ``journal``."""
    files: Dict[str, str] = {}
    for day, day_path in day_dirs().items():
        for name in os.listdir(day_path):
            if (
                AUDIO_RE.match(name)
                or SCREEN_RE.match(name)
                or SCREEN_JSONL_RE.match(name)
            ):
                rel = os.path.join(day, name)
                files[rel] = os.path.join(day_path, name)
    return files


def _parse_audio_json(path: str) -> List[tuple[str, str]]:
    """Return transcript texts and sources from ``*_audio.json`` file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    texts: List[tuple[str, str]] = []
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                text = entry.get("text")
                source = entry.get("source", "unknown")
                if text:
                    texts.append((str(text), str(source)))
    elif isinstance(data, dict):
        text = data.get("text")
        source = data.get("source", "unknown")
        if text:
            texts.append((str(text), str(source)))
    return texts


def _parse_screen_diff(path: str, source: str) -> List[tuple[str, str]]:
    """Return visual description and OCR text from ``*_diff.json`` file with source."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    texts: List[tuple[str, str]] = []
    if isinstance(data, dict):
        desc = data.get("visual_description")
        if desc:
            texts.append((str(desc), source))
        ocr = data.get("full_ocr")
        if ocr:
            texts.append((str(ocr), source))
    return texts


def _parse_screen_jsonl(path: str) -> List[tuple[str, str]]:
    """Return visual descriptions, extracted text, and meeting info from ``*_screen.jsonl`` file.

    Returns list of (text, source) tuples where source is the monitor position
    (or monitor name if position not available).
    """
    texts: List[tuple[str, str]] = []

    # Lazy import to avoid dependency issues during module init
    from observe.utils import load_analysis_frames

    # Use shared loading logic from observe module
    frames = load_analysis_frames(Path(path))

    for frame in frames:
        # Determine source: prefer monitor_position, fallback to monitor
        source = frame.get("monitor_position") or frame.get("monitor", "unknown")

        # Extract visual_description from analysis
        analysis = frame.get("analysis")
        if analysis and isinstance(analysis, dict):
            visual_desc = analysis.get("visual_description")
            if visual_desc:
                texts.append((str(visual_desc), source))

        # Extract text content if present
        extracted_text = frame.get("extracted_text")
        if extracted_text:
            texts.append((str(extracted_text), source))

        # Extract meeting analysis if present
        meeting_analysis = frame.get("meeting_analysis")
        if meeting_analysis and isinstance(meeting_analysis, dict):
            # Index platform
            platform = meeting_analysis.get("platform")
            if platform:
                texts.append((f"Meeting platform: {platform}", source))

            # Index participant names
            participants = meeting_analysis.get("participants", [])
            if participants and isinstance(participants, list):
                for participant in participants:
                    if isinstance(participant, dict):
                        name = participant.get("name")
                        if name and name.lower() != "unknown":
                            texts.append((f"Meeting participant: {name}", source))

            # Index screen share content
            screen_share = meeting_analysis.get("screen_share")
            if screen_share and isinstance(screen_share, dict):
                presenter = screen_share.get("presenter")
                if presenter:
                    texts.append((f"Screen share presenter: {presenter}", source))

                description = screen_share.get("description")
                if description:
                    texts.append((f"Screen share: {description}", source))

                formatted_text = screen_share.get("formatted_text")
                if formatted_text:
                    texts.append((str(formatted_text), source))

    return texts


def _index_transcripts(
    conn: sqlite3.Connection, rel: str, path: str, verbose: bool
) -> None:
    """Index text from transcript audio, screen diff JSON, or screen JSONL files."""
    logger = logging.getLogger(__name__)

    name = os.path.basename(path)
    m = AUDIO_RE.match(name)
    if m:
        rtype = "audio"
        texts = _parse_audio_json(path)
    else:
        # Try new JSONL format first
        m = SCREEN_JSONL_RE.match(name)
        if m:
            rtype = "screen"
            texts = _parse_screen_jsonl(path)
        else:
            # Fall back to legacy diff.json format
            m = SCREEN_RE.match(name)
            if not m:
                return
            rtype = "screen"
            # Extract source from filename: <time>_<name>_<id>_diff.json
            # Example: 123456_monitor_1_diff.json -> source is "monitor_1"
            # Remove .json extension first
            name_no_ext = name.replace(".json", "")
            parts = name_no_ext.split("_")
            if len(parts) >= 4 and parts[-1] == "diff":
                # Join the parts between time and _diff to get source
                source = "_".join(parts[1:-1])
            else:
                source = "unknown"
            texts = _parse_screen_diff(path, source)

    if not m:
        return

    time_part = m.group("time")
    day = rel.split(os.sep, 1)[0]
    for text, source in texts:
        conn.execute(
            (
                "INSERT INTO transcripts_text(content, path, day, time, type, source) VALUES (?, ?, ?, ?, ?, ?)"
            ),
            (text, rel, day, time_part, rtype, source),
        )
    if verbose:
        logger.info("  indexed transcript %s entries", len(texts))


def scan_transcripts(journal: str, verbose: bool = False) -> bool:
    """Index transcript audio, screen diff JSON, and screen JSONL files on a per-day basis."""
    logger = logging.getLogger(__name__)
    files = find_transcript_files(journal)
    if not files:
        return False

    grouped: Dict[str, Dict[str, str]] = {}
    for rel, path in files.items():
        day = rel.split(os.sep, 1)[0]
        grouped.setdefault(day, {})[rel] = path

    changed = False
    for day, day_files in grouped.items():
        conn, _ = get_index(index="transcripts", journal=journal, day=day)
        logger.info("\nIndexing %s transcript files for %s...", len(day_files), day)
        if _scan_files(
            conn,
            day_files,
            "DELETE FROM transcripts_text WHERE path=?",
            _index_transcripts,
            verbose,
        ):
            conn.commit()
            changed = True
        conn.close()
    return changed


def search_transcripts(
    query: str,
    limit: int = 5,
    offset: int = 0,
    day: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    source: str | None = None,
) -> tuple[int, List[Dict[str, str]]]:
    """Search transcript indexes and return total count and results.

    If ``day`` is provided only that day's index is searched and results are
    ordered chronologically. Otherwise all available per-day indexes are
    queried and results are ordered by relevance.

    If ``start_date`` and ``end_date`` are provided, only days within that
    range are searched (format: YYYYMMDD).

    If ``source`` is provided, results are filtered to only that source.
    """

    results: List[Dict[str, str]] = []
    total = 0

    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    # Determine which days to search
    if day:
        days = [day]
    else:
        all_days = sorted(day_dirs())
        # Filter by date range if provided
        if start_date and end_date:
            days = [d for d in all_days if start_date <= d <= end_date]
        elif start_date:  # Only start date
            days = [d for d in all_days if d >= start_date]
        elif end_date:  # Only end date
            days = [d for d in all_days if d <= end_date]
        else:
            days = all_days
    for d in days:
        conn, _ = get_index(index="transcripts", day=d)
        db = sqlite_utils.Database(conn)
        quoted = db.quote(query)

        # Build WHERE clause with optional source filter
        where_clause = f"transcripts_text MATCH {quoted}"
        if source:
            where_clause += f" AND source = {db.quote(source)}"

        total += conn.execute(
            f"SELECT count(*) FROM transcripts_text WHERE {where_clause}"
        ).fetchone()[0]

        order_clause = "time" if day else "rank"
        cursor = conn.execute(
            f"""
            SELECT content, path, day, time, type, source, bm25(transcripts_text) as rank
            FROM transcripts_text WHERE {where_clause} ORDER BY {order_clause}
            """
        )

        for (
            content,
            path,
            day_label,
            time_part,
            rtype,
            source_val,
            rank,
        ) in cursor.fetchall():
            results.append(
                {
                    "id": path,
                    "text": content,
                    "metadata": {
                        "day": day_label,
                        "path": path,
                        "time": time_part,
                        "type": rtype,
                        "source": source_val,
                    },
                    "score": rank,
                }
            )
        conn.close()

    if day:
        results.sort(key=lambda r: r["metadata"]["time"])
    else:
        results.sort(key=lambda r: r["score"])
    start = offset
    end = offset + limit
    return total, results[start:end]
