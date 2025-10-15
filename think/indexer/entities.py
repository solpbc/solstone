"""Entity indexing and search functionality."""

import logging
import os
import re
import sqlite3
from typing import Any, Dict, List, Tuple

import sqlite_utils

from .core import _scan_files, get_index

# Entity parsing helpers
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
    """Map relative entity file paths to full paths for domain-scoped entities."""
    files: Dict[str, str] = {}
    domains_dir = os.path.join(journal, "domains")

    if not os.path.isdir(domains_dir):
        return files

    # Scan all directories in domains/ (not just those with domain.json)
    for domain_name in os.listdir(domains_dir):
        domain_path = os.path.join(domains_dir, domain_name)
        if not os.path.isdir(domain_path):
            continue

        # Check for attached entities: domains/{domain}/entities.md
        attached_path = os.path.join(domain_path, "entities.md")
        if os.path.isfile(attached_path):
            rel = os.path.join(domain_name, "entities.md")
            files[rel] = attached_path

        # Check for detected entities: domains/{domain}/entities/*.md
        detected_dir = os.path.join(domain_path, "entities")
        if os.path.isdir(detected_dir):
            for filename in os.listdir(detected_dir):
                if filename.endswith(".md") and len(filename) == 11:  # YYYYMMDD.md
                    day = filename[:-3]
                    if day.isdigit() and len(day) == 8:
                        rel = os.path.join(domain_name, "entities", filename)
                        files[rel] = os.path.join(detected_dir, filename)

    return files


def _index_entities(
    conn: sqlite3.Connection, rel: str, path: str, verbose: bool
) -> None:
    """Index parsed entities from domain-scoped ``entities.md`` file."""
    logger = logging.getLogger(__name__)

    # Extract domain and day from relative path
    # Format: {domain}/entities.md or {domain}/entities/YYYYMMDD.md
    parts = rel.split(os.sep)
    domain = parts[0]
    day = None
    attached = 1

    if len(parts) == 3:  # {domain}/entities/YYYYMMDD.md
        day = os.path.splitext(parts[2])[0]  # Remove .md extension
        attached = 0

    # Parse entities from the file
    # For attached entities: path ends with entities.md, parse from parent dir
    # For detected entities: path ends with YYYYMMDD.md in entities/ subdir, parse directly
    if attached:
        entries = parse_entities(os.path.dirname(path))
    else:
        # For detected entities, parse the specific file directly
        entries = []
        valid_types = {"Person", "Company", "Project", "Tool"}

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not ENTITY_ITEM_RE.match(line.replace("**", "")):
                    continue
                parsed = parse_entity_line(line)
                if not parsed:
                    continue
                etype, name, desc = parsed
                if etype not in valid_types:
                    continue
                entries.append((etype, name, desc))

    for etype, name, desc in entries:
        conn.execute(
            "INSERT INTO entities(name, description, domain, day, type, attached) VALUES (?, ?, ?, ?, ?, ?)",
            (name, desc, domain, day, etype, attached),
        )

    if verbose:
        logger.info(
            "  indexed %s entities (domain=%s, day=%s)",
            len(entries),
            domain,
            day or "attached",
        )


def scan_entities(journal: str, verbose: bool = False) -> bool:
    """Index entities from domain-scoped ``entities.md`` files."""
    logger = logging.getLogger(__name__)
    conn, _ = get_index(index="entities", journal=journal)
    files = find_entity_files(journal)

    if files:
        logger.info("\nIndexing %s entity files...", len(files))

    # Build delete SQL based on rel path structure
    def get_delete_sql(rel: str) -> str:
        parts = rel.split(os.sep)
        domain = parts[0]
        if len(parts) == 2:  # attached: {domain}/entities.md
            return f"DELETE FROM entities WHERE domain='{domain}' AND day IS NULL"
        else:  # detected: {domain}/entities/YYYYMMDD.md
            day = os.path.splitext(parts[2])[0]
            return f"DELETE FROM entities WHERE domain='{domain}' AND day='{day}'"

    # Modified _scan_files logic to handle dynamic delete SQL per file
    # For now, use a list of possible delete patterns - we'll handle this in a custom scan
    import time

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

    for idx, (rel, path, mtime) in enumerate(to_index, 1):
        if verbose:
            logger.info("[%s/%s] %s", idx, len(to_index), rel)

        # Execute appropriate delete SQL based on file type
        delete_sql = get_delete_sql(rel)
        conn.execute(delete_sql)

        _index_entities(conn, rel, path, verbose)
        conn.execute("REPLACE INTO files(path, mtime) VALUES (?, ?)", (rel, mtime))

    # Remove files that no longer exist
    removed = set(db_mtimes) - set(files)
    for rel in removed:
        delete_sql = get_delete_sql(rel)
        conn.execute(delete_sql)
        conn.execute("DELETE FROM files WHERE path=?", (rel,))

    elapsed = time.time() - start
    logger.info("%s total indexed in %.2f seconds", len(to_index), elapsed)

    changed = bool(to_index or removed)
    if changed:
        conn.commit()
    conn.close()
    return changed


def search_entities(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    domain: str | None = None,
    day: str | None = None,
    etype: str | None = None,
    name: str | None = None,
    attached: bool | None = None,
    order: str = "rank",
) -> tuple[int, List[Dict[str, Any]]]:
    """Search the entities index and return total count and results.

    Parameters
    ----------
    query : str
        FTS search across name and description fields
    limit : int
        Maximum number of results to return
    offset : int
        Number of results to skip
    domain : str, optional
        Filter to specific domain
    day : str, optional
        Filter to specific day (YYYYMMDD format)
    etype : str, optional
        Filter by entity type (Person, Company, Project, Tool)
    name : str, optional
        Column-specific FTS search on name field
    attached : bool, optional
        Filter by attached status (True=attached only, False=detected only)
    order : str
        Sort order: "rank" (BM25 relevance) or "day" (chronological)

    Returns
    -------
    tuple[int, List[Dict[str, Any]]]
        Total count and list of result dictionaries
    """
    conn, _ = get_index(index="entities")
    db = sqlite_utils.Database(conn)

    # Build FTS query combining query and name
    fts_parts = []
    if query:
        fts_parts.append(query)
    if name:
        # FTS5 column-specific search: escape double quotes and wrap in double quotes
        escaped_name = name.replace('"', '""')
        # Add wildcard for prefix matching to support partial name searches
        fts_parts.append(f'name:"{escaped_name}"*')

    where_clause = "1"
    params: List[Any] = []

    if fts_parts:
        # Use db.quote to properly escape the entire FTS5 query
        quoted = db.quote(" AND ".join(fts_parts))
        where_clause = f"entities MATCH {quoted}"

    if domain:
        where_clause += " AND domain=?"
        params.append(domain)
    if day:
        where_clause += " AND day=?"
        params.append(day)
    if etype:
        where_clause += " AND type=?"
        params.append(etype)
    if attached is True:
        where_clause += " AND attached=1"
    elif attached is False:
        where_clause += " AND attached=0"

    total = conn.execute(
        f"SELECT count(*) FROM entities WHERE {where_clause}", params
    ).fetchone()[0]

    order = order.lower()
    order_by = "bm25(entities)"
    if order == "day":
        order_by = "day DESC"

    cursor = conn.execute(
        f"""
        SELECT name, description, domain, day, type, attached, bm25(entities) as rank
        FROM entities WHERE {where_clause} ORDER BY {order_by} LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    )

    results: List[Dict[str, Any]] = []
    for (
        name_val,
        desc_val,
        domain_val,
        day_val,
        type_val,
        attached_val,
        rank,
    ) in cursor.fetchall():
        # Build unique ID
        if day_val:
            result_id = f"{domain_val}/entities/{day_val}.md:{name_val}"
        else:
            result_id = f"{domain_val}/entities.md:{name_val}"

        results.append(
            {
                "id": result_id,
                "text": desc_val or name_val,
                "metadata": {
                    "domain": domain_val,
                    "day": day_val,
                    "type": type_val,
                    "name": name_val,
                    "attached": bool(attached_val),
                },
                "score": rank,
            }
        )

    conn.close()
    return total, results
