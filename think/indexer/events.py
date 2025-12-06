"""Event indexing and search functionality."""

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List

from .core import _scan_files, get_index, sanitize_fts_query
from .insights import find_insight_files


def _index_events(conn: sqlite3.Connection, rel: str, path: str, verbose: bool) -> None:
    """Index events from a JSON file.

    Handles both occurrences (occurred=1) and anticipations (occurred=0).
    For anticipations, the event's 'date' field is used as the index day.
    For occurrences, the file's day directory is used.
    """
    logger = logging.getLogger(__name__)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check for anticipations first, then occurrences
    if isinstance(data, dict) and "anticipations" in data:
        events = data.get("anticipations", [])
        occurred = 0
    elif isinstance(data, dict):
        events = data.get("occurrences", [])
        occurred = 1
    else:
        events = data
        occurred = 1

    if verbose:
        event_type = "anticipations" if occurred == 0 else "occurrences"
        logger.info("  indexed %s %s", len(events), event_type)

    file_day = rel.split(os.sep, 1)[0]
    topic = os.path.splitext(os.path.basename(rel))[0]

    for idx, event in enumerate(events):
        # For anticipations, use the event's date field as the index day
        # This allows querying "what's scheduled for date X"
        if occurred == 0:
            event_date = event.get("date", "")
            # Convert YYYY-MM-DD to YYYYMMDD for consistency
            index_day = event_date.replace("-", "") if event_date else file_day
        else:
            index_day = file_day

        conn.execute(
            "INSERT INTO events_text(content, path, day, idx) VALUES (?, ?, ?, ?)",
            (
                json.dumps(event, ensure_ascii=False),
                rel,
                index_day,
                idx,
            ),
        )
        conn.execute(
            "INSERT INTO event_match(path, day, idx, topic, facet, start, end, occurred) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rel,
                index_day,
                idx,
                topic,
                event.get("facet", ""),
                event.get("start") or "",
                event.get("end") or "",
                occurred,
            ),
        )


def scan_events(journal: str, verbose: bool = False) -> bool:
    """Index event JSON files."""
    logger = logging.getLogger(__name__)
    conn, _ = get_index(index="events", journal=journal)
    files = find_insight_files(journal, (".json",))
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
    facet: str | None = None,
    start: str | None = None,
    end: str | None = None,
    topic: str | None = None,
    occurred: bool | None = None,
) -> tuple[int, List[Dict[str, Any]]]:
    """Search the events index and return total count and results.

    Parameters
    ----------
    query : str
        FTS search query (empty string for no text search).
    limit : int
        Maximum results to return.
    offset : int
        Number of results to skip.
    day : str, optional
        Filter by day (YYYYMMDD). For anticipations, this is the event date.
    facet : str, optional
        Filter by facet identifier.
    start : str, optional
        Filter events ending at or after this time (HH:MM:SS).
    end : str, optional
        Filter events starting at or before this time (HH:MM:SS).
    topic : str, optional
        Filter by insight topic.
    occurred : bool, optional
        Filter by occurred status. True for occurrences, False for anticipations.
        None returns both.
    """
    conn, _ = get_index(index="events")

    # Build WHERE clause and parameters
    params: list = []

    # Only use FTS MATCH if query is non-empty
    if query:
        sanitized = sanitize_fts_query(query)
        where_clause = f"events_text MATCH '{sanitized}'"
    else:
        # No search query, just filter by metadata
        where_clause = "1=1"

    if day:
        where_clause += " AND m.day=?"
        params.append(day)
    if facet:
        where_clause += " AND m.facet=?"
        params.append(facet)
    if topic:
        where_clause += " AND m.topic=?"
        params.append(topic)
    if start:
        where_clause += " AND m.end>=?"
        params.append(start)
    if end:
        where_clause += " AND m.start<=?"
        params.append(end)
    if occurred is not None:
        where_clause += " AND m.occurred=?"
        params.append(1 if occurred else 0)

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
               m.path, m.day, m.idx, m.topic, m.facet, m.start, m.end, m.occurred,
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
            facet_val,
            start_val,
            end_val,
            occurred_val,
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
                    "facet": facet_val,
                    "start": start_val,
                    "end": end_val,
                    "occurred": bool(occurred_val),
                    "participants": occ_obj.get("participants"),
                },
                "score": rank,
                "event": occ_obj,
            }
        )
    conn.close()
    return total, results
