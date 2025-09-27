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
    domains_dir = os.path.join(journal, "domains")

    if not os.path.isdir(domains_dir):
        return files

    for domain_name in os.listdir(domains_dir):
        domain_path = os.path.join(domains_dir, domain_name)
        if not os.path.isdir(domain_path):
            continue

        news_dir = os.path.join(domain_path, "news")
        if not os.path.isdir(news_dir):
            continue

        for filename in os.listdir(news_dir):
            if filename.endswith(".md"):
                # Extract day from filename (YYYYMMDD.md format)
                day = filename[:-3]  # Remove .md extension
                if len(day) == 8 and day.isdigit():
                    rel = os.path.join("domains", domain_name, "news", filename)
                    files[rel] = os.path.join(news_dir, filename)

    return files


def _index_news_content(
    conn: sqlite3.Connection, rel: str, path: str, verbose: bool
) -> None:
    """Index content from a news markdown file."""
    logger = logging.getLogger(__name__)

    # Extract domain and day from relative path
    # Format: domains/{domain}/news/{YYYYMMDD}.md
    parts = rel.split(os.sep)
    domain = parts[1]  # domains/DOMAIN/news/YYYYMMDD.md
    day = os.path.splitext(parts[3])[0]  # Remove .md extension

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if verbose:
        logger.info("  indexed news for domain %s, day %s", domain, day)

    conn.execute(
        "INSERT INTO news_text(content, domain, day) VALUES (?, ?, ?)",
        (content, domain, day),
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
        "DELETE FROM news_text WHERE 'domains/' || domain || '/news/' || day || '.md' = ?",
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
    domain: str | None = None,
    day: str | None = None,
) -> tuple[int, List[Dict[str, str]]]:
    """Search the news index and return total count and results."""

    conn, _ = get_index(index="news")
    db = sqlite_utils.Database(conn)
    quoted = db.quote(query)

    where_clause = f"news_text MATCH {quoted}"
    params: List[str] = []

    if domain:
        where_clause += " AND domain=?"
        params.append(domain)
    if day:
        where_clause += " AND day=?"
        params.append(day)

    total = conn.execute(
        f"SELECT count(*) FROM news_text WHERE {where_clause}", params
    ).fetchone()[0]

    cursor = conn.execute(
        f"""
        SELECT content, domain, day, bm25(news_text) as rank
        FROM news_text WHERE {where_clause} ORDER BY rank LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    )

    results: List[Dict[str, str]] = []
    for content, domain, day, rank in cursor.fetchall():
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
                "id": f"domains/{domain}/news/{day}.md",
                "text": snippet or content[:200],
                "metadata": {
                    "domain": domain,
                    "day": day,
                    "path": f"domains/{domain}/news/{day}.md",
                },
                "score": rank,
            }
        )

    conn.close()
    return total, results
