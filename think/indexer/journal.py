"""Unified journal index for all content types.

This module provides a single FTS5 index over all journal content:
- Insights (markdown files)
- Transcripts (audio/screen JSONL)
- Events (facet event JSONL)
- Entities (facet entity JSONL)
- Todos (facet todo JSONL)

All content is converted to markdown chunks via the formatters framework,
then indexed with metadata fields for filtering (day, facet, topic).
"""

import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from think.formatters import find_formattable_files, format_file, load_jsonl
from think.utils import segment_key

logger = logging.getLogger(__name__)

# Database constants
INDEX_DIR = "indexer"
DB_NAME = "journal.sqlite"

# Date pattern for path parsing
DATE_RE = re.compile(r"^\d{8}$")

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
        idx UNINDEXED
    )
    """,
]


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create required tables if they don't exist."""
    for statement in SCHEMA:
        conn.execute(statement)


def get_journal_index(journal: str | None = None) -> tuple[sqlite3.Connection, str]:
    """Return SQLite connection for the journal index.

    Args:
        journal: Path to journal root. Uses JOURNAL_PATH env var if not provided.

    Returns:
        Tuple of (connection, db_path)
    """
    journal = journal or os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

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


def parse_path_metadata(rel_path: str) -> dict[str, str]:
    """Extract day, facet, and topic from a journal-relative path.

    Args:
        rel_path: Journal-relative path (e.g., "20240101/insights/flow.md")

    Returns:
        Dict with keys: day, facet, topic
    """
    parts = rel_path.replace("\\", "/").split("/")
    filename = parts[-1]
    basename = os.path.splitext(filename)[0]

    # YYYYMMDD/... patterns (day directories)
    if parts[0] and DATE_RE.match(parts[0]):
        day = parts[0]

        # Daily insights: YYYYMMDD/insights/*.md
        if len(parts) >= 3 and parts[1] == "insights":
            return {"day": day, "facet": "", "topic": basename}

        # Segment content: YYYYMMDD/HHMMSS*/*
        if len(parts) >= 3 and segment_key(parts[1]):
            # Markdown: screen.md, audio.md -> topic is basename
            if filename.endswith(".md"):
                return {"day": day, "facet": "", "topic": basename}

            # JSONL transcripts: topic encoded with source later
            # audio.jsonl -> "audio", screen.jsonl -> "screen"
            # *_audio.jsonl -> "audio" (mic_audio.jsonl, sys_audio.jsonl)
            if filename == "audio.jsonl" or filename.endswith("_audio.jsonl"):
                return {"day": day, "facet": "", "topic": "audio"}
            if filename == "screen.jsonl":
                return {"day": day, "facet": "", "topic": "screen"}

    # facets/*/* patterns
    if parts[0] == "facets" and len(parts) >= 3:
        facet = parts[1]

        # Events: facets/*/events/YYYYMMDD.jsonl
        if parts[2] == "events" and len(parts) >= 4:
            event_day = basename  # filename is YYYYMMDD.jsonl
            return {"day": event_day, "facet": facet, "topic": "event"}

        # Entities attached: facets/*/entities.jsonl
        if parts[2] == "entities.jsonl":
            return {"day": "", "facet": facet, "topic": "entity:attached"}

        # Entities detected: facets/*/entities/YYYYMMDD.jsonl
        if parts[2] == "entities" and len(parts) >= 4:
            entity_day = basename
            return {"day": entity_day, "facet": facet, "topic": "entity:detected"}

        # Todos: facets/*/todos/YYYYMMDD.jsonl
        if parts[2] == "todos" and len(parts) >= 4:
            todo_day = basename
            return {"day": todo_day, "facet": facet, "topic": "todo"}

        # News: facets/*/news/YYYYMMDD.md
        if parts[2] == "news" and len(parts) >= 4:
            news_day = basename
            return {"day": news_day, "facet": facet, "topic": "news"}

    # imports/*/summary.md
    if parts[0] == "imports" and len(parts) >= 3 and parts[2] == "summary.md":
        import_id = parts[1]
        # Extract day from import_id (format: YYYYMMDD_HHMMSS or YYYYMMDD)
        import_day = import_id.split("_")[0] if "_" in import_id else import_id[:8]
        return {"day": import_day, "facet": "", "topic": "import"}

    # apps/*/insights/*.md
    if parts[0] == "apps" and len(parts) >= 4 and parts[2] == "insights":
        app_name = parts[1]
        return {"day": "", "facet": "", "topic": f"{app_name}:{basename}"}

    # Fallback
    return {"day": "", "facet": "", "topic": basename}


def _index_file(
    conn: sqlite3.Connection,
    rel: str,
    path: str,
    verbose: bool,
) -> None:
    """Index a single file into the chunks table.

    Uses format_file() to convert content to markdown chunks,
    then inserts each chunk with metadata.
    """
    try:
        chunks, meta = format_file(path)
    except (ValueError, FileNotFoundError) as e:
        logger.warning("Skipping %s: %s", rel, e)
        return

    metadata = parse_path_metadata(rel)
    day = metadata["day"]
    facet = metadata["facet"]
    base_topic = metadata["topic"]

    if verbose:
        logger.info(
            "  %s chunks, day=%s, facet=%s, topic=%s",
            len(chunks),
            day,
            facet,
            base_topic,
        )

    for idx, chunk in enumerate(chunks):
        content = chunk.get("markdown", "")
        if not content:
            continue

        # For transcripts, encode source in topic if available
        # Chunks from format_audio have source info we could extract
        # For now, use base topic (audio/screen)
        topic = base_topic

        conn.execute(
            "INSERT INTO chunks(content, path, day, facet, topic, idx) VALUES (?, ?, ?, ?, ?, ?)",
            (content, rel, day, facet, topic, idx),
        )


def scan_journal(journal: str, verbose: bool = False) -> bool:
    """Scan and index all journal content.

    Args:
        journal: Path to journal root directory
        verbose: If True, log detailed progress

    Returns:
        True if any files were indexed or removed
    """
    conn, db_path = get_journal_index(journal)
    files = find_formattable_files(journal)

    if not files:
        logger.info("No files to index")
        conn.close()
        return False

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
        _index_file(conn, rel, path, verbose)

        # Update file mtime
        conn.execute("REPLACE INTO files(path, mtime) VALUES (?, ?)", (rel, mtime))

    # Remove files that no longer exist
    removed = set(db_mtimes) - set(files)
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
) -> tuple[str, list[Any]]:
    """Build WHERE clause and params for FTS5 search.

    Args:
        query: FTS5 search query
        day: Filter by exact day (YYYYMMDD) - mutually exclusive with day_from/day_to
        day_from: Filter by date range start (YYYYMMDD, inclusive)
        day_to: Filter by date range end (YYYYMMDD, inclusive)
        facet: Filter by facet name
        topic: Filter by topic

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
        params.append(facet)
    if topic:
        where_clause += " AND topic=?"
        params.append(topic)

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
        topic: Filter by topic (e.g., "flow", "audio", "event")

    Returns:
        Tuple of (total_count, results) where each result has:
            - id: "{path}:{idx}"
            - text: The matched markdown chunk
            - metadata: {day, facet, topic, path, idx}
            - score: BM25 relevance score
    """
    conn, _ = get_journal_index()
    where_clause, params = _build_where_clause(
        query, day, day_from, day_to, facet, topic
    )

    # Get total count
    total = conn.execute(
        f"SELECT count(*) FROM chunks WHERE {where_clause}", params
    ).fetchone()[0]

    # Get results
    cursor = conn.execute(
        f"""
        SELECT content, path, day, facet, topic, idx, bm25(chunks) as rank
        FROM chunks WHERE {where_clause}
        ORDER BY rank LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    )

    results = []
    for content, path, day_val, facet_val, topic_val, idx, rank in cursor.fetchall():
        results.append(
            {
                "id": f"{path}:{idx}",
                "text": content,
                "metadata": {
                    "day": day_val,
                    "facet": facet_val,
                    "topic": topic_val,
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

    Returns:
        Dict with:
            - total: Total matching chunks
            - facets: Counter of facet_name -> count
            - topics: Counter of topic_name -> count
            - days: Counter of day -> count
    """
    from collections import Counter

    conn, _ = get_journal_index()
    where_clause, params = _build_where_clause(
        query, day, day_from, day_to, facet, topic
    )

    rows = conn.execute(
        f"SELECT facet, topic, day FROM chunks WHERE {where_clause}", params
    ).fetchall()

    conn.close()

    return {
        "total": len(rows),
        "facets": Counter(r[0] for r in rows if r[0]),
        "topics": Counter(r[1] for r in rows if r[1]),
        "days": Counter(r[2] for r in rows if r[2]),
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
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    events = []
    facets_dir = Path(journal) / "facets"

    if not facets_dir.is_dir():
        return events

    for facet_dir in facets_dir.iterdir():
        if not facet_dir.is_dir():
            continue

        facet_name = facet_dir.name
        if facet and facet_name != facet:
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
