"""Event indexing and search functionality."""

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List

import sqlite_utils

from .core import _scan_files, get_index
from .summaries import find_summary_files


def _index_events(conn: sqlite3.Connection, rel: str, path: str, verbose: bool) -> None:
    """Index events from a JSON file."""
    logger = logging.getLogger(__name__)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("occurrences", []) if isinstance(data, dict) else data
    if verbose:
        logger.info("  indexed %s events", len(events))
    day = rel.split(os.sep, 1)[0]
    topic = os.path.splitext(os.path.basename(rel))[0]
    for idx, event in enumerate(events):
        conn.execute(
            ("INSERT INTO events_text(content, path, day, idx) " "VALUES (?, ?, ?, ?)"),
            (
                json.dumps(event, ensure_ascii=False),
                rel,
                day,
                idx,
            ),
        )
        conn.execute(
            (
                "INSERT INTO event_match(path, day, idx, topic, start, end) VALUES (?, ?, ?, ?, ?, ?)"
            ),
            (
                rel,
                day,
                idx,
                topic,
                event.get("start", ""),
                event.get("end", ""),
            ),
        )


def scan_events(journal: str, verbose: bool = False) -> bool:
    """Index event JSON files."""
    logger = logging.getLogger(__name__)
    conn, _ = get_index(index="events", journal=journal)
    files = find_summary_files(journal, (".json",))
    if files:
        logger.info("\nIndexing %s event files...", len(files))
    changed = _scan_files(
        conn,
        files,
        [
            "DELETE FROM events_text WHERE path=?",
            "DELETE FROM event_match WHERE path=?",
        ],
        _index_events,
        verbose,
    )
    if changed:
        conn.commit()
    conn.close()
    return changed


def search_events(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    day: str | None = None,
    start: str | None = None,
    end: str | None = None,
    topic: str | None = None,
) -> tuple[int, List[Dict[str, Any]]]:
    """Search the events index and return total count and results."""
    conn, _ = get_index(index="events")
    db = sqlite_utils.Database(conn)
    quoted = db.quote(query)

    # Build WHERE clause and parameters
    where_clause = f"events_text MATCH {quoted}"
    params: List[str] = []

    if day:
        where_clause += " AND m.day=?"
        params.append(day)
    if topic:
        where_clause += " AND m.topic=?"
        params.append(topic)
    if start:
        where_clause += " AND m.end>=?"
        params.append(start)
    if end:
        where_clause += " AND m.start<=?"
        params.append(end)

    # Get total count
    total = conn.execute(
        f"""
        SELECT count(*)
        FROM events_text t JOIN event_match m ON t.path=m.path AND t.idx=m.idx
        WHERE {where_clause}
        """,
        params,
    ).fetchone()[0]

    # Get results with limit and offset, ordered by day and start time (newest first)
    sql = f"""
        SELECT t.content,
               m.path, m.day, m.idx, m.topic, m.start, m.end,
               bm25(events_text) as rank
        FROM events_text t JOIN event_match m ON t.path=m.path AND t.idx=m.idx
        WHERE {where_clause}
        ORDER BY m.day DESC, m.start DESC LIMIT ? OFFSET ?
    """

    cursor = conn.execute(sql, params + [limit, offset])
    results = []
    for row in cursor.fetchall():
        (
            content,
            path,
            day_label,
            idx,
            topic_label,
            start_val,
            end_val,
            rank,
        ) = row
        try:
            occ_obj = json.loads(content)
        except Exception:
            occ_obj = {}
        text = (
            occ_obj.get("title")
            or occ_obj.get("summary")
            or occ_obj.get("subject")
            or occ_obj.get("details")
            or content
        )
        results.append(
            {
                "id": f"{path}:{idx}",
                "text": text,
                "metadata": {
                    "day": day_label,
                    "path": path,
                    "index": idx,
                    "topic": topic_label,
                    "start": start_val,
                    "end": end_val,
                    "participants": occ_obj.get("participants"),
                },
                "score": rank,
                "event": occ_obj,
            }
        )
    conn.close()
    return total, results
