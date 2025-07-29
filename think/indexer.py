"""Utilities for indexing summary outputs and events."""

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

from .entities import (
    find_day_dirs,
    load_cache,
    save_cache,
)
from .entities import scan_entities as scan_entities_cache

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


def _ensure_schema(conn: sqlite3.Connection, index: str) -> None:
    """Create required tables for *index* if they don't exist."""
    for statement in SCHEMAS[index]:
        conn.execute(statement)


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
    *,
    index: str,
    journal: str | None = None,
    day: str | None = None,
) -> Tuple[sqlite3.Connection, str]:
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


def find_summary_files(
    journal: str, exts: Tuple[str, ...] | None = None
) -> Dict[str, str]:
    """Map relative summary file path to full path filtered by ``exts``."""
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


# Transcript file helpers -----------------------------------------------------------

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
                "INSERT INTO summaries_text(sentence, path, day, topic, position) VALUES (?, ?, ?, ?, ?)"
            ),
            (sentence, rel, day, topic, pos),
        )


def _index_events(conn: sqlite3.Connection, rel: str, path: str, verbose: bool) -> None:
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


def scan_summaries(journal: str, verbose: bool = False) -> bool:
    """Index sentences from summary markdown files."""
    logger = logging.getLogger(__name__)
    conn, _ = get_index(index="summaries", journal=journal)
    files = find_summary_files(journal, (".md",))
    if files:
        logger.info("\nIndexing %s summary files...", len(files))
    changed = _scan_files(
        conn,
        files,
        "DELETE FROM summaries_text WHERE path=?",
        _index_sentences,
        verbose,
    )
    if changed:
        conn.commit()
    conn.close()
    return changed


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


ENTITY_ITEM_RE = re.compile(r"^\s*[-*]\s*(.*)")


def parse_entity_line(line: str) -> Tuple[str, str, str] | None:
    """Parse a single line from an ``entities.md`` file."""
    cleaned = line.replace("**", "")
    match = ENTITY_ITEM_RE.match(cleaned)
    if not match:
        return None

    text = match.group(1).strip()
    if ":" not in text:
        return None

    etype, rest = text.split(":", 1)
    rest = rest.strip()
    if " - " in rest:
        name, desc = rest.split(" - ", 1)
    else:
        name, desc = rest, ""

    return etype.strip(), name.strip(), desc.strip()


def parse_entities(path: str) -> List[Tuple[str, str, str]]:
    """Return parsed entity tuples from ``entities.md`` inside ``path``."""
    items: List[Tuple[str, str, str]] = []
    valid_types = {"Person", "Company", "Project", "Tool"}

    file_path = os.path.join(path, "entities.md")
    if not os.path.isfile(file_path):
        return items

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not ENTITY_ITEM_RE.match(line.replace("**", "")):
                continue
            parsed = parse_entity_line(line)
            if not parsed:
                continue
            etype, name, desc = parsed
            if etype not in valid_types:
                continue
            items.append((etype, name, desc))

    return items


def find_entity_files(journal: str) -> Dict[str, str]:
    """Return mapping of entity file paths relative to ``journal``."""
    files: Dict[str, str] = {}
    top_path = os.path.join(journal, "entities.md")
    if os.path.isfile(top_path):
        files["entities.md"] = top_path

    for day, path in find_day_dirs(journal).items():
        md_path = os.path.join(path, "entities.md")
        if os.path.isfile(md_path):
            files[os.path.join(day, "entities.md")] = md_path

    return files


def _index_entities(
    conn: sqlite3.Connection, rel: str, path: str, verbose: bool
) -> None:
    """Index parsed entities from ``entities.md`` file."""
    logger = logging.getLogger(__name__)
    entries = parse_entities(os.path.dirname(path))
    day = rel.split(os.sep, 1)[0] if os.sep in rel else ""
    for etype, name, desc in entries:
        conn.execute(
            (
                "INSERT INTO entity_appearances(name, desc, day, type, path) VALUES (?, ?, ?, ?, ?)"
            ),
            (name, desc, day, etype, rel),
        )
    if verbose:
        logger.info("  indexed %s entities", len(entries))


def _rebuild_entities(conn: sqlite3.Connection) -> None:
    """Rebuild the aggregate entities table from appearances."""
    data: Dict[str, Dict[str, Any]] = {}
    cursor = conn.execute(
        "SELECT day, type, name, desc, path FROM entity_appearances ORDER BY day"
    )
    for day, etype, name, desc, path in cursor.fetchall():
        entry = data.setdefault(
            name,
            {
                "first_seen": "",
                "last_seen": "",
                "days": set(),
                "top": False,
                "desc": "",
                "type": "",
            },
        )
        if path == "entities.md" and day == "":
            entry["top"] = True
            entry["type"] = etype
            if desc:
                entry["desc"] = desc
            continue

        if not entry["first_seen"] or day < entry["first_seen"]:
            entry["first_seen"] = day
        if not entry["last_seen"] or day > entry["last_seen"]:
            entry["last_seen"] = day
        entry["days"].add(day)
        if not entry["top"]:
            entry["type"] = etype
            if desc:
                entry["desc"] = desc

    conn.execute("DELETE FROM entities")
    for name, info in data.items():
        conn.execute(
            "INSERT INTO entities(name, desc, first_seen, last_seen, days, top, type) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                name,
                info.get("desc", ""),
                info.get("first_seen", ""),
                info.get("last_seen", ""),
                len(info.get("days", set())),
                1 if info.get("top") else 0,
                info.get("type", ""),
            ),
        )


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


def scan_entities(journal: str, verbose: bool = False) -> bool:
    """Index entities from ``entities.md`` files across the journal."""
    logger = logging.getLogger(__name__)
    conn, _ = get_index(index="entities", journal=journal)
    files = find_entity_files(journal)
    if files:
        logger.info("\nIndexing %s entity files...", len(files))
    changed = _scan_files(
        conn,
        files,
        "DELETE FROM entity_appearances WHERE path=?",
        _index_entities,
        verbose,
    )
    if changed:
        _rebuild_entities(conn)
        conn.commit()
    conn.close()
    return changed


def search_summaries(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    day: str | None = None,
    topic: str | None = None,
) -> tuple[int, List[Dict[str, str]]]:
    """Search the summary sentence index and return total count and results."""

    conn, _ = get_index(index="summaries")
    db = sqlite_utils.Database(conn)
    quoted = db.quote(query)

    where_clause = f"summaries_text MATCH {quoted}"
    params: List[str] = []

    if day:
        where_clause += " AND day=?"
        params.append(day)
    if topic:
        where_clause += " AND topic LIKE ?"
        params.append(f"%{topic}%")

    total = conn.execute(
        f"SELECT count(*) FROM summaries_text WHERE {where_clause}", params
    ).fetchone()[0]

    cursor = conn.execute(
        f"""
        SELECT sentence, path, day, topic, position, bm25(summaries_text) as rank
        FROM summaries_text WHERE {where_clause} ORDER BY rank LIMIT ? OFFSET ?
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


# Search events from the events index.


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


def search_entities(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    day: str | None = None,
    etype: str | None = None,
    name: str | None = None,
) -> tuple[int, List[Dict[str, Any]]]:
    """Search the entities index and return total count and results."""

    conn, _ = get_index(index="entities")
    db = sqlite_utils.Database(conn)

    fts_parts = []
    if query:
        fts_parts.append(query)
    if name:
        fts_parts.append(f"name:{name}")

    where_clause = "1"
    params: List[str] = []
    if fts_parts:
        quoted = db.quote(" AND ".join(fts_parts))
        where_clause = f"entities MATCH {quoted}"
    if day:
        where_clause += " AND first_seen<=? AND last_seen>=?"
        params.extend([day, day])
    if etype:
        where_clause += " AND type=?"
        params.append(etype)

    total_entities = conn.execute(
        f"SELECT count(*) FROM entities WHERE {where_clause}", params
    ).fetchone()[0]

    ent_cursor = conn.execute(
        f"SELECT name, desc, type, top, first_seen, last_seen, days, bm25(entities) as rank FROM entities WHERE {where_clause} ORDER BY rank LIMIT ? OFFSET ?",
        params + [limit, offset],
    )

    ent_results: List[Dict[str, Any]] = []
    for (
        name_val,
        desc_val,
        type_val,
        top_val,
        first_seen_val,
        last_seen_val,
        days_val,
        rank,
    ) in ent_cursor.fetchall():
        ent_results.append(
            {
                "id": name_val,
                "text": desc_val or name_val,
                "metadata": {
                    "name": name_val,
                    "type": type_val,
                    "top": bool(top_val),
                    "first_seen": first_seen_val,
                    "last_seen": last_seen_val,
                    "days": days_val,
                },
                "score": rank,
            }
        )

    # Determine remaining limit/offset for appearances
    remaining_limit = max(0, limit - len(ent_results))
    appearance_offset = 0
    if offset > total_entities:
        appearance_offset = offset - total_entities
    elif offset + limit > total_entities:
        appearance_offset = 0
        remaining_limit = limit - (total_entities - offset)
    else:
        remaining_limit = 0

    fts_clause = "1"
    params2: List[str] = []
    if fts_parts:
        quoted2 = db.quote(" AND ".join(fts_parts))
        fts_clause = f"entity_appearances MATCH {quoted2}"
    if day:
        fts_clause += " AND day=?"
        params2.append(day)
    if etype:
        fts_clause += " AND type=?"
        params2.append(etype)

    total_appearances = conn.execute(
        f"SELECT count(*) FROM entity_appearances WHERE {fts_clause}", params2
    ).fetchone()[0]

    app_results: List[Dict[str, Any]] = []
    if remaining_limit > 0:
        app_cursor = conn.execute(
            f"SELECT name, desc, day, type, path, bm25(entity_appearances) as rank FROM entity_appearances WHERE {fts_clause} ORDER BY rank LIMIT ? OFFSET ?",
            params2 + [remaining_limit, appearance_offset],
        )
        for (
            name_val,
            desc_val,
            day_val,
            type_val,
            path_val,
            rank,
        ) in app_cursor.fetchall():
            app_results.append(
                {
                    "id": f"{path_val}:{name_val}",
                    "text": desc_val or name_val,
                    "metadata": {
                        "day": day_val,
                        "path": path_val,
                        "type": type_val,
                        "name": name_val,
                    },
                    "score": rank,
                }
            )

    conn.close()
    total = total_entities + total_appearances
    return total, ent_results + app_results


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
        description="Index summary markdown and event files"
    )
    parser.add_argument(
        "--index",
        choices=["summaries", "events", "transcripts", "entities"],
        required=True,
        help="Which index to operate on",
    )
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Scan journal and update the index before searching",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove the selected index before optional rescan",
    )
    parser.add_argument(
        "--day",
        help="Limit transcript query to a specific YYYYMMDD day",
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

    if args.reset:
        reset_index(
            journal, args.index, day=args.day if args.index == "transcripts" else None
        )

    if args.rescan:
        if args.index == "transcripts":
            changed = scan_transcripts(journal, verbose=args.verbose)
            if changed:
                journal_log("indexer transcripts rescan ok")
        elif args.index == "events":
            changed = scan_events(journal, verbose=args.verbose)
            if changed:
                journal_log("indexer events rescan ok")
        elif args.index == "summaries":
            cache = load_cache(journal)
            changed = scan_entities_cache(journal, cache)
            changed |= scan_summaries(journal, verbose=args.verbose)
            if changed:
                save_cache(journal, cache)
                journal_log("indexer summaries rescan ok")
        elif args.index == "entities":
            changed = scan_entities(journal, verbose=args.verbose)
            if changed:
                journal_log("indexer entities rescan ok")

    # Handle query argument
    if args.query is not None:
        if args.index == "transcripts":
            search_func = search_transcripts
            query_kwargs = {"day": args.day}
        elif args.index == "events":
            search_func = search_events
            query_kwargs = {}
        elif args.index == "entities":
            search_func = search_entities
            query_kwargs = {"day": args.day}
        else:
            search_func = search_summaries
            query_kwargs = {}
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
