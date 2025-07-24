"""Utilities for indexing topic outputs and occurrences."""

import json
import logging
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Tuple

import sqlite_utils
from syntok import segmenter

from think.utils import get_topics, journal_log, setup_cli

from .entities import find_day_dirs, load_cache, save_cache, scan_entities

INDEX_DIR = "index"


# Sentence indexing helpers -------------------------------------------------

TOPIC_DIR = os.path.join(os.path.dirname(__file__), "topics")
TOPIC_BASENAMES = sorted(get_topics().keys())


def split_sentences(text: str) -> List[str]:
    """Return a list of cleaned sentences from markdown text."""
    cleaned = re.sub(r"^[*-]\s*", "", text, flags=re.MULTILINE)
    sentences: List[str] = []
    for paragraph in segmenter.process(cleaned):
        for sentence in paragraph:
            joined = "".join(str(t) for t in sentence).strip()
            if joined:
                sentences.append(joined)
    return sentences


# Ponder indexing -----------------------------------------------------------


def get_index(
    journal: str | None = None, day: str | None = None
) -> Tuple[sqlite3.Connection, str]:
    """Return SQLite connection for indexes.

    If ``journal`` is not provided it will be read from the ``JOURNAL_PATH``
    environment variable. When ``day`` is supplied the database is stored under
    that day's ``index`` directory, otherwise the journal level ``index`` folder
    is used.
    """

    journal = journal or os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    db_dir = (
        os.path.join(journal, day, INDEX_DIR)
        if day
        else os.path.join(journal, INDEX_DIR)
    )
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "indexer.sqlite")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, mtime INTEGER)"
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS topics_text USING fts5(
            sentence, path UNINDEXED, day UNINDEXED, topic UNINDEXED, position UNINDEXED
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS occ_text USING fts5(
            content,
            path UNINDEXED, day UNINDEXED, idx UNINDEXED
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS occ_match(
            path TEXT,
            day TEXT,
            idx INTEGER,
            topic TEXT,
            start TEXT,
            end TEXT,
            PRIMARY KEY(path, idx)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS occ_match_day_start_end ON occ_match(day, start, end)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS occ_match_day_topic ON occ_match(day, topic)"
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS raws_text USING fts5(
            content, path UNINDEXED, day UNINDEXED, time UNINDEXED, type UNINDEXED
        )
        """
    )
    return conn, db_path


def find_topic_files(
    journal: str, exts: Tuple[str, ...] | None = None
) -> Dict[str, str]:
    """Map relative topic file path to full path filtered by ``exts``."""
    files: Dict[str, str] = {}
    exts = exts or (".md", ".json")
    for day, day_path in find_day_dirs(journal).items():
        topics_dir = os.path.join(day_path, "topics")
        if not os.path.isdir(topics_dir):
            continue
        for name in os.listdir(topics_dir):
            base, ext = os.path.splitext(name)
            if ext in exts and base in TOPIC_BASENAMES:
                rel = os.path.join(day, "topics", name)
                files[rel] = os.path.join(topics_dir, name)
    return files


# Raw file helpers -----------------------------------------------------------

AUDIO_RE = re.compile(r"^(?P<time>\d{6})_audio\.json$")
SCREEN_RE = re.compile(r"^(?P<time>\d{6})_[a-z]+_\d+_diff\.json$")


def find_raw_files(journal: str) -> Dict[str, str]:
    """Return mapping of raw JSON file paths relative to ``journal``."""
    files: Dict[str, str] = {}
    for day, day_path in find_day_dirs(journal).items():
        for name in os.listdir(day_path):
            if AUDIO_RE.match(name) or SCREEN_RE.match(name):
                rel = os.path.join(day, name)
                files[rel] = os.path.join(day_path, name)
    return files


def _scan_files(
    conn: sqlite3.Connection,
    files: Dict[str, str],
    delete_sql: str | List[str],
    index_func,
    verbose: bool = False,
) -> bool:
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


def _index_sentences(
    conn: sqlite3.Connection, rel: str, path: str, verbose: bool
) -> None:
    logger = logging.getLogger(__name__)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = split_sentences(text)
    if verbose:
        logger.info("  indexed %s sentences", len(sentences))
    day, topic = rel.split(os.sep, 1)
    for pos, sentence in enumerate(sentences):
        conn.execute(
            (
                "INSERT INTO topics_text(sentence, path, day, topic, position) VALUES (?, ?, ?, ?, ?)"
            ),
            (sentence, rel, day, topic, pos),
        )


def _index_occurrences(
    conn: sqlite3.Connection, rel: str, path: str, verbose: bool
) -> None:
    logger = logging.getLogger(__name__)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    occs = data.get("occurrences", []) if isinstance(data, dict) else data
    if verbose:
        logger.info("  indexed %s occurrences", len(occs))
    day = rel.split(os.sep, 1)[0]
    topic = os.path.splitext(os.path.basename(rel))[0]
    for idx, occ in enumerate(occs):
        conn.execute(
            ("INSERT INTO occ_text(content, path, day, idx) " "VALUES (?, ?, ?, ?)"),
            (
                json.dumps(occ, ensure_ascii=False),
                rel,
                day,
                idx,
            ),
        )
        conn.execute(
            (
                "INSERT INTO occ_match(path, day, idx, topic, start, end) VALUES (?, ?, ?, ?, ?, ?)"
            ),
            (
                rel,
                day,
                idx,
                topic,
                occ.get("start", ""),
                occ.get("end", ""),
            ),
        )


def scan_topics(journal: str, verbose: bool = False) -> bool:
    """Index sentences from topic markdown files."""
    logger = logging.getLogger(__name__)
    conn, _ = get_index(journal)
    files = find_topic_files(journal, (".md",))
    if files:
        logger.info("\nIndexing %s topic files...", len(files))
    changed = _scan_files(
        conn, files, "DELETE FROM topics_text WHERE path=?", _index_sentences, verbose
    )
    if changed:
        conn.commit()
    conn.close()
    return changed


def scan_occurrences(journal: str, verbose: bool = False) -> bool:
    """Index occurrence JSON files."""
    logger = logging.getLogger(__name__)
    conn, _ = get_index(journal)
    files = find_topic_files(journal, (".json",))
    if files:
        logger.info("\nIndexing %s occurrence files...", len(files))
    changed = _scan_files(
        conn,
        files,
        ["DELETE FROM occ_text WHERE path=?", "DELETE FROM occ_match WHERE path=?"],
        _index_occurrences,
        verbose,
    )
    if changed:
        conn.commit()
    conn.close()
    return changed


def _index_raws(conn: sqlite3.Connection, rel: str, path: str, verbose: bool) -> None:
    logger = logging.getLogger(__name__)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    name = os.path.basename(path)
    m = AUDIO_RE.match(name)
    rtype = "audio"
    if not m:
        m = SCREEN_RE.match(name)
        rtype = "screen"
    if not m:
        return
    time_part = m.group("time")
    day = rel.split(os.sep, 1)[0]
    conn.execute(
        (
            "INSERT INTO raws_text(content, path, day, time, type) VALUES (?, ?, ?, ?, ?)"
        ),
        (content, rel, day, time_part, rtype),
    )
    if verbose:
        logger.info("  indexed raw %s", rel)


def scan_raws(journal: str, verbose: bool = False) -> bool:
    """Index raw audio and screen diff JSON files on a per-day basis."""
    logger = logging.getLogger(__name__)
    files = find_raw_files(journal)
    if not files:
        return False

    grouped: Dict[str, Dict[str, str]] = {}
    for rel, path in files.items():
        day = rel.split(os.sep, 1)[0]
        grouped.setdefault(day, {})[rel] = path

    changed = False
    for day, day_files in grouped.items():
        conn, _ = get_index(journal, day=day)
        logger.info("\nIndexing %s raw files for %s...", len(day_files), day)
        if _scan_files(
            conn, day_files, "DELETE FROM raws_text WHERE path=?", _index_raws, verbose
        ):
            conn.commit()
            changed = True
        conn.close()
    return changed


def search_topics(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    day: str | None = None,
    topic: str | None = None,
) -> tuple[int, List[Dict[str, str]]]:
    """Search the topic sentence index and return total count and results."""

    conn, _ = get_index()
    db = sqlite_utils.Database(conn)
    quoted = db.quote(query)

    where_clause = f"topics_text MATCH {quoted}"
    params: List[str] = []

    if day:
        where_clause += " AND day=?"
        params.append(day)
    if topic:
        where_clause += " AND topic LIKE ?"
        params.append(f"%{topic}%")

    total = conn.execute(
        f"SELECT count(*) FROM topics_text WHERE {where_clause}", params
    ).fetchone()[0]

    cursor = conn.execute(
        f"""
        SELECT sentence, path, day, topic, position, bm25(topics_text) as rank
        FROM topics_text WHERE {where_clause} ORDER BY rank LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    )
    results: List[Dict[str, str]] = []
    for sentence, path, day, topic, pos, rank in cursor.fetchall():
        results.append(
            {
                "id": path,
                "text": sentence,
                "metadata": {
                    "day": day,
                    "topic": topic,
                    "path": path,
                    "index": pos,
                },
                "score": rank,
            }
        )
    conn.close()
    return total, results


def search_occurrences(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    day: str | None = None,
    start: str | None = None,
    end: str | None = None,
    topic: str | None = None,
) -> tuple[int, List[Dict[str, Any]]]:
    """Search the occurrences index and return total count and results."""
    conn, _ = get_index()
    db = sqlite_utils.Database(conn)
    quoted = db.quote(query)

    # Build WHERE clause and parameters
    where_clause = f"occ_text MATCH {quoted}"
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
        FROM occ_text t JOIN occ_match m ON t.path=m.path AND t.idx=m.idx
        WHERE {where_clause}
        """,
        params,
    ).fetchone()[0]

    # Get results with limit and offset
    sql = f"""
        SELECT t.content,
               m.path, m.day, m.idx, m.topic, m.start, m.end,
               bm25(occ_text) as rank
        FROM occ_text t JOIN occ_match m ON t.path=m.path AND t.idx=m.idx
        WHERE {where_clause}
        ORDER BY rank LIMIT ? OFFSET ?
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
                "occurrence": occ_obj,
            }
        )
    conn.close()
    return total, results


def search_raws(
    query: str,
    limit: int = 5,
    offset: int = 0,
    day: str | None = None,
) -> tuple[int, List[Dict[str, str]]]:
    """Search raw indexes and return total count and results.

    If ``day`` is provided only that day's index is searched. Otherwise all
    available per-day indexes are queried.
    """

    results: List[Dict[str, str]] = []
    total = 0

    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    days = [day] if day else sorted(find_day_dirs(journal))
    for d in days:
        conn, _ = get_index(day=d)
        db = sqlite_utils.Database(conn)
        quoted = db.quote(query)

        total += conn.execute(
            f"SELECT count(*) FROM raws_text WHERE raws_text MATCH {quoted}"
        ).fetchone()[0]

        cursor = conn.execute(
            f"""
            SELECT content, path, day, time, type, bm25(raws_text) as rank
            FROM raws_text WHERE raws_text MATCH {quoted} ORDER BY rank
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

    results.sort(key=lambda r: r["score"])
    start = offset
    end = offset + limit
    return total, results[start:end]


def _display_search_results(results: List[Dict[str, str]]) -> None:
    """Display search results in a consistent format."""
    for idx, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        snippet = r["text"]
        label = meta.get("topic") or meta.get("time") or ""
        print(f"{idx}. {meta.get('day')} {label}: {snippet}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Index topic markdown and occurrence files"
    )
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Scan journal and update the index before searching",
    )
    parser.add_argument(
        "--raws",
        action="store_true",
        help="Operate on raw *_audio.json and *_diff.json files",
    )
    parser.add_argument(
        "--day",
        help="Limit raw query to a specific YYYYMMDD day",
    )
    parser.add_argument(
        "-q",
        "--query",
        nargs="?",
        const="",
        help="Run query (interactive mode if no query provided)",
    )

    args = setup_cli(parser)

    # Require either --rescan or -q
    if not args.rescan and args.query is None:
        parser.print_help()
        return

    journal = os.getenv("JOURNAL_PATH")

    if args.rescan:
        if args.raws:
            changed = scan_raws(journal, verbose=args.verbose)
            if changed:
                journal_log("indexer raw rescan ok")
        else:
            cache = load_cache(journal)
            changed = scan_entities(journal, cache)
            changed |= scan_topics(journal, verbose=args.verbose)
            changed |= scan_occurrences(journal, verbose=args.verbose)
            if changed:
                save_cache(journal, cache)
                journal_log("indexer rescan ok")

    # Handle query argument
    if args.query is not None:
        search_func = search_raws if args.raws else search_topics
        query_kwargs = {}
        if args.raws:
            query_kwargs["day"] = args.day
        if args.query:
            # Single query mode - run query and exit
            _total, results = search_func(args.query, 5, **query_kwargs)
            _display_search_results(results)
        else:
            # Interactive mode
            while True:
                try:
                    query = input("search> ").strip()
                except EOFError:
                    break
                if not query:
                    break
                _total, results = search_func(query, 5, **query_kwargs)
                _display_search_results(results)


if __name__ == "__main__":
    main()
