"""Summary indexing and search functionality."""

import logging
import os
import sqlite3
from typing import Dict, List, Tuple

import sqlite_utils
from syntok import segmenter

from think.utils import get_topics

from .core import _scan_files, find_day_dirs, get_index

# Sentence indexing helpers
TOPIC_DIR = os.path.join(os.path.dirname(__file__), "..", "topics")
TOPIC_BASENAMES = sorted(get_topics().keys())


def split_sentences(text: str) -> List[str]:
    """Return a list of cleaned sentences from markdown text."""
    import re

    cleaned = re.sub(r"^[*-]\s*", "", text, flags=re.MULTILINE)
    sentences: List[str] = []
    for paragraph in segmenter.process(cleaned):
        for sentence in paragraph:
            joined = "".join(str(t) for t in sentence).strip()
            if joined:
                sentences.append(joined)
    return sentences


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


def _index_sentences(
    conn: sqlite3.Connection, rel: str, path: str, verbose: bool
) -> None:
    """Index sentences from a summary markdown file."""
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
