"""Insight indexing and search functionality."""

import logging
import os
import re
import sqlite3
from typing import Dict, List, Tuple

from think.insights import chunk_markdown, render_chunk
from think.utils import day_dirs, get_insight_topic, get_insights

from .core import _scan_files, get_index, sanitize_fts_query

# Regex to match segment folder names (HHMMSS_LEN format)
SEGMENT_RE = re.compile(r"^\d{6}_\d+$")


def _get_insight_topics() -> set[str]:
    """Return set of valid insight topic filenames (filesystem names)."""
    return {get_insight_topic(key) for key in get_insights().keys()}


def find_event_files(journal: str) -> Dict[str, str]:
    """Map relative event file path to full path.

    Scans facet event directories: facets/*/events/*.jsonl
    """
    files: Dict[str, str] = {}

    facets_dir = os.path.join(journal, "facets")
    if not os.path.isdir(facets_dir):
        return files

    for facet_name in os.listdir(facets_dir):
        events_dir = os.path.join(facets_dir, facet_name, "events")
        if not os.path.isdir(events_dir):
            continue
        for name in os.listdir(events_dir):
            if not name.endswith(".jsonl"):
                continue
            rel = os.path.join("facets", facet_name, "events", name)
            files[rel] = os.path.join(events_dir, name)

    return files


def find_insight_files(
    journal: str, exts: Tuple[str, ...] | None = None
) -> Dict[str, str]:
    """Map relative insight file path to full path filtered by ``exts``.

    Scans five locations:
    - Daily insights: YYYYMMDD/insights/*.md
    - Segment insights: YYYYMMDD/HHMMSS_LEN/*.md
    - Import summaries: imports/*/summary.md
    - Facet news: facets/*/news/*.md
    - App insights: apps/*/insights/*.md
    """
    files: Dict[str, str] = {}
    exts = exts or (".md",)
    insight_topics = _get_insight_topics()

    # Scan daily and segment insights
    for day, day_path in day_dirs().items():
        # Daily insights in insights/ subdirectory
        insights_dir = os.path.join(day_path, "insights")
        if os.path.isdir(insights_dir):
            for name in os.listdir(insights_dir):
                base, ext = os.path.splitext(name)
                if ext in exts and base in insight_topics:
                    rel = os.path.join(day, "insights", name)
                    files[rel] = os.path.join(insights_dir, name)

        # Segment insights in HHMMSS_LEN/ subdirectories
        for entry in os.listdir(day_path):
            if not SEGMENT_RE.match(entry):
                continue
            segment_dir = os.path.join(day_path, entry)
            if not os.path.isdir(segment_dir):
                continue
            for name in os.listdir(segment_dir):
                base, ext = os.path.splitext(name)
                if ext in exts and base in insight_topics:
                    rel = os.path.join(day, entry, name)
                    files[rel] = os.path.join(segment_dir, name)

    # Scan import summaries
    imports_dir = os.path.join(journal, "imports")
    if os.path.isdir(imports_dir):
        for import_id in os.listdir(imports_dir):
            import_path = os.path.join(imports_dir, import_id)
            if not os.path.isdir(import_path):
                continue
            summary_path = os.path.join(import_path, "summary.md")
            if os.path.isfile(summary_path) and ".md" in exts:
                rel = os.path.join("imports", import_id, "summary.md")
                files[rel] = summary_path

    # Scan facet news
    facets_dir = os.path.join(journal, "facets")
    if os.path.isdir(facets_dir):
        for facet_name in os.listdir(facets_dir):
            news_dir = os.path.join(facets_dir, facet_name, "news")
            if not os.path.isdir(news_dir):
                continue
            for name in os.listdir(news_dir):
                base, ext = os.path.splitext(name)
                if ext in exts:
                    rel = os.path.join("facets", facet_name, "news", name)
                    files[rel] = os.path.join(news_dir, name)

    # Scan app insights (flat structure: apps/{app}/insights/*.md)
    apps_dir = os.path.join(journal, "apps")
    if os.path.isdir(apps_dir):
        for app_name in os.listdir(apps_dir):
            app_insights_dir = os.path.join(apps_dir, app_name, "insights")
            if not os.path.isdir(app_insights_dir):
                continue
            for name in os.listdir(app_insights_dir):
                base, ext = os.path.splitext(name)
                if ext in exts:
                    rel = os.path.join("apps", app_name, "insights", name)
                    files[rel] = os.path.join(app_insights_dir, name)

    return files


def _index_chunks(conn: sqlite3.Connection, rel: str, path: str, verbose: bool) -> None:
    """Index chunks from an insight markdown file."""
    logger = logging.getLogger(__name__)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [render_chunk(c) for c in chunk_markdown(text)]
    if verbose:
        logger.info("  indexed %s chunks", len(chunks))

    # Parse path to extract day and topic
    # Formats:
    #   Daily: 20240101/insights/flow.md -> day=20240101, topic=flow
    #   Segment: 20240101/143022_300/screen.md -> day=20240101, topic=screen
    #   Import: imports/20250115_093000/summary.md -> day=20250115, topic=import
    #   Facet news: facets/ml_research/news/20250118.md -> day=20250118, topic=news
    #   App insight: apps/{app}/insights/{topic}.md -> day=None, topic={app}:{topic}
    parts = rel.split(os.sep)

    if parts[0] == "imports":
        # Import summary: imports/{import_id}/summary.md
        # Extract day from import_id (format: YYYYMMDD_HHMMSS)
        import_id = parts[1]
        day = import_id.split("_")[0] if "_" in import_id else import_id[:8]
        topic = "import"
    elif parts[0] == "facets":
        # Facet news: facets/{facet_name}/news/YYYYMMDD.md
        # Extract day from filename
        filename = parts[-1]
        day = os.path.splitext(filename)[0]
        topic = "news"
    elif parts[0] == "apps":
        # App insight: apps/{app}/insights/{topic}.md
        # No day association, topic is {app}:{basename}
        app_name = parts[1]
        filename = parts[-1]
        base = os.path.splitext(filename)[0]
        day = ""  # App insights may not have a day
        topic = f"{app_name}:{base}"
    else:
        # Daily or segment insight
        day = parts[0]
        # Get basename without extension as topic
        filename = parts[-1]
        topic = os.path.splitext(filename)[0]

    for pos, chunk in enumerate(chunks):
        conn.execute(
            (
                "INSERT INTO insights_text(sentence, path, day, topic, position) VALUES (?, ?, ?, ?, ?)"
            ),
            (chunk, rel, day, topic, pos),
        )


def scan_insights(journal: str, verbose: bool = False) -> bool:
    """Index chunks from insight markdown files."""
    logger = logging.getLogger(__name__)
    conn, _ = get_index(index="insights", journal=journal)
    files = find_insight_files(journal, (".md",))
    if files:
        logger.info("\nIndexing %s insight files...", len(files))
    changed = _scan_files(
        conn,
        files,
        "DELETE FROM insights_text WHERE path=?",
        _index_chunks,
        verbose,
    )
    if changed:
        conn.commit()
    conn.close()
    return changed


def search_insights(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    day: str | None = None,
    topic: str | None = None,
) -> tuple[int, List[Dict[str, str]]]:
    """Search the insight sentence index and return total count and results."""

    conn, _ = get_index(index="insights")
    sanitized = sanitize_fts_query(query)

    where_clause = f"insights_text MATCH '{sanitized}'"
    params: List[str] = []

    if day:
        where_clause += " AND day=?"
        params.append(day)
    if topic:
        where_clause += " AND topic=?"
        params.append(topic)

    total = conn.execute(
        f"SELECT count(*) FROM insights_text WHERE {where_clause}", params
    ).fetchone()[0]

    cursor = conn.execute(
        f"""
        SELECT sentence, path, day, topic, position, bm25(insights_text) as rank
        FROM insights_text WHERE {where_clause} ORDER BY rank LIMIT ? OFFSET ?
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
