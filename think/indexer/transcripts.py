"""Transcript indexing and search functionality."""

import json
import logging
import os
import re
import sqlite3
from typing import Dict, List

import sqlite_utils

from .core import _scan_files, find_day_dirs, get_index

# Transcript file helpers
AUDIO_RE = re.compile(r"^(?P<time>\d{6})_audio\.json$")
SCREEN_RE = re.compile(r"^(?P<time>\d{6})_[a-z]+_\d+_diff\.json$")


def find_transcript_files(journal: str) -> Dict[str, str]:
    """Return mapping of transcript JSON file paths relative to ``journal``."""
    files: Dict[str, str] = {}
    for day, day_path in find_day_dirs(journal).items():
        for name in os.listdir(day_path):
            if AUDIO_RE.match(name) or SCREEN_RE.match(name):
                rel = os.path.join(day, name)
                files[rel] = os.path.join(day_path, name)
    return files


def _parse_audio_json(path: str) -> List[str]:
    """Return transcript texts from ``*_audio.json`` file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    texts: List[str] = []
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                text = entry.get("text")
                if text:
                    texts.append(str(text))
    elif isinstance(data, dict):
        text = data.get("text")
        if text:
            texts.append(str(text))
    return texts


def _parse_screen_diff(path: str) -> List[str]:
    """Return visual description and OCR text from ``*_diff.json`` file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    texts: List[str] = []
    if isinstance(data, dict):
        desc = data.get("visual_description")
        if desc:
            texts.append(str(desc))
        ocr = data.get("full_ocr")
        if ocr:
            texts.append(str(ocr))
    return texts


def _index_transcripts(
    conn: sqlite3.Connection, rel: str, path: str, verbose: bool
) -> None:
    """Index text from transcript audio or screen diff JSON files."""
    logger = logging.getLogger(__name__)

    name = os.path.basename(path)
    m = AUDIO_RE.match(name)
    if m:
        rtype = "audio"
        texts = _parse_audio_json(path)
    else:
        m = SCREEN_RE.match(name)
        if not m:
            return
        rtype = "screen"
        texts = _parse_screen_diff(path)

    if not m:
        return

    time_part = m.group("time")
    day = rel.split(os.sep, 1)[0]
    for text in texts:
        conn.execute(
            (
                "INSERT INTO transcripts_text(content, path, day, time, type) VALUES (?, ?, ?, ?, ?)"
            ),
            (text, rel, day, time_part, rtype),
        )
    if verbose:
        logger.info("  indexed transcript %s entries", len(texts))


def scan_transcripts(journal: str, verbose: bool = False) -> bool:
    """Index transcript audio and screen diff JSON files on a per-day basis."""
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
) -> tuple[int, List[Dict[str, str]]]:
    """Search transcript indexes and return total count and results.

    If ``day`` is provided only that day's index is searched and results are
    ordered chronologically. Otherwise all available per-day indexes are
    queried and results are ordered by relevance.
    """

    results: List[Dict[str, str]] = []
    total = 0

    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    days = [day] if day else sorted(find_day_dirs(journal))
    for d in days:
        conn, _ = get_index(index="transcripts", day=d)
        db = sqlite_utils.Database(conn)
        quoted = db.quote(query)

        total += conn.execute(
            f"SELECT count(*) FROM transcripts_text WHERE transcripts_text MATCH {quoted}"
        ).fetchone()[0]

        order_clause = "time" if day else "rank"
        cursor = conn.execute(
            f"""
            SELECT content, path, day, time, type, bm25(transcripts_text) as rank
            FROM transcripts_text WHERE transcripts_text MATCH {quoted} ORDER BY {order_clause}
            """
        )

        for content, path, day_label, time_part, rtype, rank in cursor.fetchall():
            results.append(
                {
                    "id": path,
                    "text": content,
                    "metadata": {
                        "day": day_label,
                        "path": path,
                        "time": time_part,
                        "type": rtype,
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
