"""Core database and file system utilities for the indexer."""

import os
import re
import sqlite3
import time
from typing import Callable, Dict, List, Union

# Common regex patterns
DATE_RE = re.compile(r"\d{8}")

# Database constants
INDEX_DIR = "indexer"

# Mapping of index types to their SQLite filenames
DB_NAMES = {
    "summaries": "summaries.sqlite",
    "events": "events.sqlite",
    "transcripts": "transcripts.sqlite",
    "entities": "entities.sqlite",
}

# SQL statements to create required tables per index
SCHEMAS = {
    "summaries": [
        "CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, mtime INTEGER)",
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS summaries_text USING fts5(
            sentence, path UNINDEXED, day UNINDEXED, topic UNINDEXED, position UNINDEXED
        )
        """,
    ],
    "events": [
        "CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, mtime INTEGER)",
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS events_text USING fts5(
            content,
            path UNINDEXED, day UNINDEXED, idx UNINDEXED
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS event_match(
            path TEXT,
            day TEXT,
            idx INTEGER,
            topic TEXT,
            start TEXT,
            end TEXT,
            PRIMARY KEY(path, idx)
        )
        """,
        "CREATE INDEX IF NOT EXISTS event_match_day_start_end ON event_match(day, start, end)",
        "CREATE INDEX IF NOT EXISTS event_match_day_topic ON event_match(day, topic)",
    ],
    "transcripts": [
        "CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, mtime INTEGER)",
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_text USING fts5(
            content, path UNINDEXED, day UNINDEXED, time UNINDEXED, type UNINDEXED
        )
        """,
    ],
    "entities": [
        "CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, mtime INTEGER)",
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS entities USING fts5(
            name,
            desc,
            first_seen UNINDEXED,
            last_seen UNINDEXED,
            days UNINDEXED,
            top UNINDEXED,
            type UNINDEXED
        )
        """,
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS entity_appearances USING fts5(
            name,
            desc,
            day UNINDEXED,
            type UNINDEXED,
            path UNINDEXED
        )
        """,
    ],
}


def find_day_dirs(journal: str) -> Dict[str, str]:
    """Return mapping of YYYYMMDD strings to absolute paths."""
    days: Dict[str, str] = {}
    for name in os.listdir(journal):
        if DATE_RE.fullmatch(name):
            path = os.path.join(journal, name)
            if os.path.isdir(path):
                days[name] = path
    return days


def _ensure_schema(conn: sqlite3.Connection, index: str) -> None:
    """Create required tables for *index* if they don't exist."""
    for statement in SCHEMAS[index]:
        conn.execute(statement)


def get_index(
    *,
    index: str,
    journal: str | None = None,
    day: str | None = None,
) -> tuple[sqlite3.Connection, str]:
    """Return SQLite connection for the given *index* type."""

    journal = journal or os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    if index == "transcripts":
        if not day:
            raise ValueError("day required for transcripts index")
        db_dir = os.path.join(journal, day, INDEX_DIR)
    else:
        db_dir = os.path.join(journal, INDEX_DIR)

    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, DB_NAMES[index])
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _ensure_schema(conn, index)
    return conn, db_path


def reset_index(journal: str, index: str, *, day: str | None = None) -> None:
    """Remove SQLite files for the given *index*."""

    if index == "transcripts":
        if day:
            paths = [os.path.join(journal, day, INDEX_DIR, DB_NAMES[index])]
        else:
            paths = [
                os.path.join(journal, d, INDEX_DIR, DB_NAMES[index])
                for d in find_day_dirs(journal)
            ]
    else:
        paths = [os.path.join(journal, INDEX_DIR, DB_NAMES[index])]

    for p in paths:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


def _scan_files(
    conn: sqlite3.Connection,
    files: Dict[str, str],
    delete_sql: Union[str, List[str]],
    index_func: Callable[[sqlite3.Connection, str, str, bool], None],
    verbose: bool = False,
) -> bool:
    """Scan and index files, handling mtime-based caching."""
    import logging

    logger = logging.getLogger(__name__)
    total = len(files)

    # Get current file mtimes from database
    db_mtimes = {
        path: mtime for path, mtime in conn.execute("SELECT path, mtime FROM files")
    }

    to_index = []
    for rel, path in files.items():
        mtime = int(os.path.getmtime(path))
        if db_mtimes.get(rel) != mtime:
            to_index.append((rel, path, mtime))

    cached = total - len(to_index)
    logger.info("%s total files, %s cached, %s to index", total, cached, len(to_index))
    start = time.time()
    if isinstance(delete_sql, str):
        delete_sqls = [delete_sql]
    else:
        delete_sqls = list(delete_sql)

    for idx, (rel, path, mtime) in enumerate(to_index, 1):
        if verbose:
            logger.info("[%s/%s] %s", idx, len(to_index), rel)
        for dsql in delete_sqls:
            conn.execute(dsql, (rel,))
        index_func(conn, rel, path, verbose)
        conn.execute("REPLACE INTO files(path, mtime) VALUES (?, ?)", (rel, mtime))

    # Remove files that no longer exist
    removed = set(db_mtimes) - set(files)
    for rel in removed:
        for dsql in delete_sqls:
            conn.execute(dsql, (rel,))
        conn.execute("DELETE FROM files WHERE path=?", (rel,))

    elapsed = time.time() - start
    logger.info("%s total indexed in %.2f seconds", len(to_index), elapsed)
    return bool(to_index or removed)
