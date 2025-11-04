"""News indexing and search functionality."""

import logging
import os
import sqlite3
from typing import Dict, List

import sqlite_utils

from .core import _scan_files, get_index


def find_news_files(journal: str) -> Dict[str, str]:
    """Map relative news file path to full path."""
    files: Dict[str, str] = {}
    facets_dir = os.path.join(journal, "facets")

    if not os.path.isdir(facets_dir):
        return files

    for facet_name in os.listdir(facets_dir):
        facet_path = os.path.join(facets_dir, facet_name)
        if not os.path.isdir(facet_path):
            continue

        news_dir = os.path.join(facet_path, "news")
        if not os.path.isdir(news_dir):
            continue

        for filename in os.listdir(news_dir):
            if filename.endswith(".md"):
                # Extract day from filename (YYYYMMDD.md format)
                day = filename[:-3]  # Remove .md extension
                if len(day) == 8 and day.isdigit():
                    rel = os.path.join("facets", facet_name, "news", filename)
                    files[rel] = os.path.join(news_dir, filename)

    return files


def _index_news_content(
    conn: sqlite3.Connection, rel: str, path: str, verbose: bool
) -> None:
    """Index content from a news markdown file."""
    logger = logging.getLogger(__name__)

    # Extract facet and day from relative path
    # Format: facets/{facet}/news/{YYYYMMDD}.md
    parts = rel.split(os.sep)
    facet = parts[1]  # facets/FACET/news/YYYYMMDD.md
    day = os.path.splitext(parts[3])[0]  # Remove .md extension

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if verbose:
        logger.info("  indexed news for facet %s, day %s", facet, day)

    conn.execute(
        "INSERT INTO news_text(content, facet, day) VALUES (?, ?, ?)",
        (content, facet, day),
    )


def scan_news(journal: str, verbose: bool = False) -> bool:
    """Index content from news markdown files."""
    logger = logging.getLogger(__name__)
    conn, _ = get_index(index="news", journal=journal)
    files = find_news_files(journal)

    if files:
        logger.info("\nIndexing %s news files...", len(files))

    changed = _scan_files(
        conn,
        files,
        "DELETE FROM news_text WHERE 'facets/' || facet || '/news/' || day || '.md' = ?",
        _index_news_content,
        verbose,
    )

    if changed:
        conn.commit()
    conn.close()
    return changed


def search_news(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    facet: str | None = None,
    day: str | None = None,
) -> tuple[int, List[Dict[str, str]]]:
    """Search the news index and return total count and results."""

    conn, _ = get_index(index="news")
    db = sqlite_utils.Database(conn)
    quoted = db.quote(query)

    where_clause = f"news_text MATCH {quoted}"
    params: List[str] = []

    if facet:
        where_clause += " AND facet=?"
        params.append(facet)
    if day:
        where_clause += " AND day=?"
        params.append(day)

    total = conn.execute(
        f"SELECT count(*) FROM news_text WHERE {where_clause}", params
    ).fetchone()[0]

    cursor = conn.execute(
        f"""
        SELECT content, facet, day, bm25(news_text) as rank
        FROM news_text WHERE {where_clause} ORDER BY day DESC LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    )

    results: List[Dict[str, str]] = []
    for content, facet, day, rank in cursor.fetchall():
        # Extract first non-empty line as snippet
        lines = content.split("\n")
        snippet = ""
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                snippet = line[:200]  # Limit snippet length
                break

        results.append(
            {
                "id": f"facets/{facet}/news/{day}.md",
                "text": snippet or content[:200],
                "metadata": {
                    "facet": facet,
                    "day": day,
                    "path": f"facets/{facet}/news/{day}.md",
                },
                "score": rank,
            }
        )

    conn.close()
    return total, results
