# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unified journal index for all content types.

This module provides a single FTS5 index over journal content:
- Agent outputs (markdown files)
- Events (facet event JSONL)
- Entities (facet entity JSONL)
- Todos (facet todo JSONL)
- Action logs (facet/journal-level JSONL)

All content is converted to markdown chunks via the formatters framework,
then indexed with metadata fields for filtering (day, facet, topic).
Raw audio/screen transcripts are formattable but not indexed by default.
"""

import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from think.formatters import (
    extract_path_metadata,
    find_formattable_files,
    format_file,
    get_formatter,
    load_jsonl,
)
from think.utils import DATE_RE, get_journal

logger = logging.getLogger(__name__)

# Database constants
INDEX_DIR = "indexer"
DB_NAME = "journal.sqlite"

# Schema for the unified journal index
SCHEMA = [
    "CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, mtime INTEGER)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
        content,
        path UNINDEXED,
        day UNINDEXED,
        facet UNINDEXED,
        topic UNINDEXED,
        stream UNINDEXED,
        idx UNINDEXED
    )
    """,
]


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create required tables if they don't exist.

    Detects stale schemas (e.g., missing ``stream`` column from pre-Phase 2)
    and recreates the index. Data is fully regenerable via ``--rescan-full``.
    """
    for statement in SCHEMA:
        conn.execute(statement)

    # Verify the chunks table has all expected columns. FTS5 virtual tables
    # cannot be ALTERed, so if the schema is stale we must drop and recreate.
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    except Exception:
        return  # table_info failed — schema was just created, nothing to fix

    expected = {"content", "path", "day", "facet", "topic", "stream", "idx"}
    if not expected.issubset(cols):
        logger.info("Index schema outdated (missing columns), rebuilding")
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute("DROP TABLE IF EXISTS files")
        for statement in SCHEMA:
            conn.execute(statement)


def get_journal_index(journal: str | None = None) -> tuple[sqlite3.Connection, str]:
    """Return SQLite connection for the journal index.

    Args:
        journal: Path to journal root. Uses JOURNAL_PATH env var if not provided.

    Returns:
        Tuple of (connection, db_path)
    """
    journal = journal or get_journal()

    db_dir = os.path.join(journal, INDEX_DIR)
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, DB_NAME)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _ensure_schema(conn)

    return conn, db_path


def reset_journal_index(journal: str) -> None:
    """Remove the journal index database file."""
    db_path = os.path.join(journal, INDEX_DIR, DB_NAME)
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


def index_file(journal: str, file_path: str, verbose: bool = False) -> bool:
    """Index a single file into the journal index.

    Validates that the file exists, is under the journal directory, and has
    a registered formatter. Then indexes it (replacing any existing chunks).

    Args:
        journal: Path to journal root directory
        file_path: Absolute or journal-relative path to file
        verbose: If True, log detailed progress

    Returns:
        True if file was indexed successfully

    Raises:
        ValueError: If file is outside journal or has no formatter
        FileNotFoundError: If file doesn't exist
    """
    journal_path = Path(journal).resolve()

    # Resolve file path (handle both absolute and relative)
    if os.path.isabs(file_path):
        abs_path = Path(file_path).resolve()
    else:
        abs_path = (journal_path / file_path).resolve()

    # Validate file exists
    if not abs_path.is_file():
        raise FileNotFoundError(f"File not found: {abs_path}")

    # Validate file is under journal
    try:
        rel_path = str(abs_path.relative_to(journal_path))
    except ValueError:
        raise ValueError(f"File is outside journal directory: {abs_path}") from None

    # Validate formatter exists
    if get_formatter(rel_path) is None:
        raise ValueError(f"No formatter found for: {rel_path}")

    # Get file mtime
    mtime = int(os.path.getmtime(abs_path))

    # Index the file
    conn, _ = get_journal_index(journal)

    # Delete existing chunks for this file
    conn.execute("DELETE FROM chunks WHERE path=?", (rel_path,))

    if verbose:
        logger.info("Indexing %s", rel_path)

    stream = _extract_stream(journal, rel_path)
    _index_file(conn, rel_path, str(abs_path), verbose, stream=stream)

    # Update file mtime
    conn.execute("REPLACE INTO files(path, mtime) VALUES (?, ?)", (rel_path, mtime))

    conn.commit()
    conn.close()

    return True


def _extract_stream(journal: str, rel: str) -> str | None:
    """Extract stream name from a journal-relative path's segment directory.

    Reads stream.json from the segment dir if the path is inside a segment
    (e.g., "20240101/142500_300/agents/facet/flow.md").

    Returns stream name string or None for non-segment paths or pre-stream segments.
    """
    from think.streams import read_segment_stream
    from think.utils import segment_key

    parts = rel.replace("\\", "/").split("/")
    # Segment paths: parts[0]=day, parts[1]=segment, parts[2+]=file
    if len(parts) >= 2 and segment_key(parts[1]):
        seg_dir = os.path.join(journal, parts[0], parts[1])
        marker = read_segment_stream(seg_dir)
        if marker:
            return marker.get("stream")
    return None


def _index_file(
    conn: sqlite3.Connection,
    rel: str,
    path: str,
    verbose: bool,
    stream: str | None = None,
) -> None:
    """Index a single file into the chunks table.

    Uses format_file() to convert content to markdown chunks,
    then inserts each chunk with metadata.

    Metadata is sourced from two places:
    - Path-derived: day and facet from extract_path_metadata()
    - Formatter-provided: topic from meta["indexer"]["topic"]
    For markdown files, topic is also path-derived.
    """
    try:
        chunks, meta = format_file(path)
    except (ValueError, FileNotFoundError) as e:
        logger.warning("Skipping %s: %s", rel, e)
        return

    # Get path-derived metadata (day, facet, topic for .md files)
    path_meta = extract_path_metadata(rel)

    # Get formatter-provided metadata (topic for JSONL files)
    formatter_indexer = meta.get("indexer", {})

    # Merge: formatter values override path values, normalize to lowercase
    day = formatter_indexer.get("day") or path_meta["day"]
    facet = (formatter_indexer.get("facet") or path_meta["facet"]).lower()
    topic = (formatter_indexer.get("topic") or path_meta["topic"]).lower()

    if verbose:
        logger.info(
            "  %s chunks, day=%s, facet=%s, topic=%s, stream=%s",
            len(chunks),
            day,
            facet,
            topic,
            stream,
        )

    for idx, chunk in enumerate(chunks):
        content = chunk.get("markdown", "")
        if not content:
            continue

        conn.execute(
            "INSERT INTO chunks(content, path, day, facet, topic, stream, idx) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (content, rel, day, facet, topic, stream, idx),
        )


def _is_historical_day(rel_path: str) -> bool:
    """Check if path is in a historical YYYYMMDD directory (before today).

    Returns True for paths like "20240101/..." where the date is before today.
    Returns False for non-day paths (facets/, imports/, apps/) or today/future.
    """
    from datetime import datetime

    if not rel_path or "/" not in rel_path:
        return False

    first_part = rel_path.split("/")[0]
    if not DATE_RE.fullmatch(first_part):
        return False  # Not a day directory

    today = datetime.now().strftime("%Y%m%d")
    return first_part < today


def scan_journal(journal: str, verbose: bool = False, full: bool = False) -> bool:
    """Scan and index journal content.

    Args:
        journal: Path to journal root directory
        verbose: If True, log detailed progress
        full: If True, scan all files. If False (default), exclude historical
            YYYYMMDD directories (before today) for lighter incremental scans.

    Returns:
        True if any files were indexed or removed
    """
    conn, db_path = get_journal_index(journal)
    files = find_formattable_files(journal)

    # Light mode: exclude historical day directories
    if not full:
        files = {
            rel: path for rel, path in files.items() if not _is_historical_day(rel)
        }

    logger.info("Scanning %s files...", len(files))

    # Get current file mtimes from database
    db_mtimes = {
        path: mtime for path, mtime in conn.execute("SELECT path, mtime FROM files")
    }

    to_index = []
    for rel, path in files.items():
        try:
            mtime = int(os.path.getmtime(path))
        except OSError:
            continue
        if db_mtimes.get(rel) != mtime:
            to_index.append((rel, path, mtime))

    cached = len(files) - len(to_index)
    logger.info(
        "%s total files, %s cached, %s to index", len(files), cached, len(to_index)
    )

    start = time.time()

    for i, (rel, path, mtime) in enumerate(to_index, 1):
        if verbose:
            logger.info("[%s/%s] %s", i, len(to_index), rel)

        # Delete existing chunks for this file
        conn.execute("DELETE FROM chunks WHERE path=?", (rel,))

        # Index the file
        stream = _extract_stream(journal, rel)
        _index_file(conn, rel, path, verbose, stream=stream)

        # Update file mtime
        conn.execute("REPLACE INTO files(path, mtime) VALUES (?, ?)", (rel, mtime))

    # Remove files that no longer exist
    # In full mode: remove all missing entries
    # In light mode: only remove entries that would have been scanned (non-historical)
    removed: set[str] = set()
    if full:
        removed = set(db_mtimes) - set(files)
    else:
        # Filter db entries to those in light mode's scan scope, then find missing
        in_scope_db = {rel for rel in db_mtimes if not _is_historical_day(rel)}
        removed = in_scope_db - set(files)

    for rel in removed:
        conn.execute("DELETE FROM chunks WHERE path=?", (rel,))
        conn.execute("DELETE FROM files WHERE path=?", (rel,))

    if to_index or removed:
        conn.commit()

    elapsed = time.time() - start
    logger.info(
        "%s indexed, %s removed in %.2f seconds", len(to_index), len(removed), elapsed
    )

    conn.close()
    return bool(to_index or removed)


def sanitize_fts_query(query: str) -> str:
    """Sanitize query for FTS5: keep alphanumeric, spaces, quotes, apostrophes, and *.

    This allows FTS5 operators (OR, AND, NOT), quoted phrases, and prefix
    matching while preventing syntax errors from special characters.
    """
    result = re.sub(r"[^a-zA-Z0-9\s\"'*]", " ", query)
    # Remove all quotes if unbalanced
    if result.count('"') % 2:
        result = result.replace('"', "")
    return result


def _build_where_clause(
    query: str,
    day: str | None = None,
    day_from: str | None = None,
    day_to: str | None = None,
    facet: str | None = None,
    topic: str | None = None,
    stream: str | None = None,
) -> tuple[str, list[Any]]:
    """Build WHERE clause and params for FTS5 search.

    Args:
        query: FTS5 search query
        day: Filter by exact day (YYYYMMDD) - mutually exclusive with day_from/day_to
        day_from: Filter by date range start (YYYYMMDD, inclusive)
        day_to: Filter by date range end (YYYYMMDD, inclusive)
        facet: Filter by facet name
        topic: Filter by topic
        stream: Filter by stream name

    Returns:
        Tuple of (where_clause, params)
    """
    params: list[Any] = []

    if query:
        sanitized = sanitize_fts_query(query)
        where_clause = f"chunks MATCH '{sanitized}'"
    else:
        where_clause = "1=1"

    if day:
        where_clause += " AND day=?"
        params.append(day)
    elif day_from or day_to:
        if day_from:
            where_clause += " AND day>=?"
            params.append(day_from)
        if day_to:
            where_clause += " AND day<=?"
            params.append(day_to)
    if facet:
        where_clause += " AND facet=?"
        params.append(facet.lower())
    if topic:
        where_clause += " AND topic=?"
        params.append(topic.lower())
    if stream:
        where_clause += " AND stream=?"
        params.append(stream)

    return where_clause, params


def search_journal(
    query: str,
    limit: int = 10,
    offset: int = 0,
    *,
    day: str | None = None,
    day_from: str | None = None,
    day_to: str | None = None,
    facet: str | None = None,
    topic: str | None = None,
    stream: str | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Search the journal index.

    Args:
        query: FTS5 search query. Words are AND'd by default; use OR to match any,
            quotes for exact phrases, * for prefix match. Empty string returns all.
        limit: Maximum results to return
        offset: Number of results to skip for pagination
        day: Filter by exact day (YYYYMMDD) - mutually exclusive with day_from/day_to
        day_from: Filter by date range start (YYYYMMDD, inclusive)
        day_to: Filter by date range end (YYYYMMDD, inclusive)
        facet: Filter by facet name
        topic: Filter by topic (e.g., "flow", "event", "news")
        stream: Filter by stream name

    Returns:
        Tuple of (total_count, results) where each result has:
            - id: "{path}:{idx}"
            - text: The matched markdown chunk
            - metadata: {day, facet, topic, stream, path, idx}
            - score: BM25 relevance score
    """
    conn, _ = get_journal_index()
    where_clause, params = _build_where_clause(
        query, day, day_from, day_to, facet, topic, stream
    )

    # Get total count
    total = conn.execute(
        f"SELECT count(*) FROM chunks WHERE {where_clause}", params
    ).fetchone()[0]

    # Get results
    cursor = conn.execute(
        f"""
        SELECT content, path, day, facet, topic, stream, idx, bm25(chunks) as rank
        FROM chunks WHERE {where_clause}
        ORDER BY rank LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    )

    results = []
    for (
        content,
        path,
        day_val,
        facet_val,
        topic_val,
        stream_val,
        idx,
        rank,
    ) in cursor.fetchall():
        results.append(
            {
                "id": f"{path}:{idx}",
                "text": content,
                "metadata": {
                    "day": day_val,
                    "facet": facet_val,
                    "topic": topic_val,
                    "stream": stream_val,
                    "path": path,
                    "idx": idx,
                },
                "score": rank,
            }
        )

    conn.close()
    return total, results


def search_counts(
    query: str,
    *,
    day: str | None = None,
    day_from: str | None = None,
    day_to: str | None = None,
    facet: str | None = None,
    topic: str | None = None,
    stream: str | None = None,
) -> dict[str, Any]:
    """Get aggregated counts for a search query.

    Uses single query + Python aggregation for efficiency.

    Args:
        query: FTS5 search query (empty string for all)
        day: Filter by exact day (YYYYMMDD) - mutually exclusive with day_from/day_to
        day_from: Filter by date range start (YYYYMMDD, inclusive)
        day_to: Filter by date range end (YYYYMMDD, inclusive)
        facet: Filter by facet name
        topic: Filter by topic
        stream: Filter by stream name

    Returns:
        Dict with:
            - total: Total matching chunks
            - facets: Counter of facet_name -> count
            - topics: Counter of topic_name -> count
            - days: Counter of day -> count
            - streams: Counter of stream_name -> count
    """
    from collections import Counter

    conn, _ = get_journal_index()
    where_clause, params = _build_where_clause(
        query, day, day_from, day_to, facet, topic, stream
    )

    rows = conn.execute(
        f"SELECT facet, topic, day, stream FROM chunks WHERE {where_clause}", params
    ).fetchall()

    conn.close()

    return {
        "total": len(rows),
        "facets": Counter(r[0] for r in rows if r[0]),
        "topics": Counter(r[1] for r in rows if r[1]),
        "days": Counter(r[2] for r in rows if r[2]),
        "streams": Counter(r[3] for r in rows if r[3]),
    }


def get_events(
    day: str,
    facet: str | None = None,
) -> list[dict[str, Any]]:
    """Get structured events for a day, re-hydrated from source files.

    This function reads the source JSONL files directly to return full
    event objects with all fields (title, summary, start, end, participants, etc.).

    Args:
        day: Day in YYYYMMDD format
        facet: Optional facet name to filter by

    Returns:
        List of event dicts with full structured data
    """
    events = []
    facets_dir = Path(get_journal()) / "facets"

    if not facets_dir.is_dir():
        return events

    for facet_dir in facets_dir.iterdir():
        if not facet_dir.is_dir():
            continue

        facet_name = facet_dir.name
        if facet and facet_name.lower() != facet.lower():
            continue

        events_file = facet_dir / "events" / f"{day}.jsonl"
        if not events_file.is_file():
            continue

        entries = load_jsonl(str(events_file))
        for entry in entries:
            # Add facet to event if not present
            entry.setdefault("facet", facet_name)
            events.append(entry)

    return events
