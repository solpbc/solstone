"""Entity indexing and search functionality."""

import logging
import os
import sqlite3
from typing import Any, Dict, List

import sqlite_utils

from .core import _scan_files, get_index


def parse_entities(path: str) -> list[dict[str, Any]]:
    """Return parsed entity dicts from ``entities.jsonl`` inside ``path``."""
    from think.entities import parse_entity_file

    file_path = os.path.join(path, "entities.jsonl")
    return parse_entity_file(file_path)


def find_entity_files(journal: str) -> Dict[str, str]:
    """Map relative entity file paths to full paths for facet-scoped entities."""
    files: Dict[str, str] = {}
    facets_dir = os.path.join(journal, "facets")

    if not os.path.isdir(facets_dir):
        return files

    # Scan all directories in facets/ (not just those with facet.json)
    for facet_name in os.listdir(facets_dir):
        facet_path = os.path.join(facets_dir, facet_name)
        if not os.path.isdir(facet_path):
            continue

        # Check for attached entities: facets/{facet}/entities.jsonl
        attached_path = os.path.join(facet_path, "entities.jsonl")
        if os.path.isfile(attached_path):
            rel = os.path.join(facet_name, "entities.jsonl")
            files[rel] = attached_path

        # Check for detected entities: facets/{facet}/entities/*.jsonl
        detected_dir = os.path.join(facet_path, "entities")
        if os.path.isdir(detected_dir):
            for filename in os.listdir(detected_dir):
                if (
                    filename.endswith(".jsonl") and len(filename) == 14
                ):  # YYYYMMDD.jsonl
                    day = filename[:-6]  # Remove .jsonl
                    if day.isdigit() and len(day) == 8:
                        rel = os.path.join(facet_name, "entities", filename)
                        files[rel] = os.path.join(detected_dir, filename)

    return files


def _index_entities(
    conn: sqlite3.Connection, rel: str, path: str, verbose: bool
) -> None:
    """Index parsed entities from facet-scoped ``entities.jsonl`` file."""
    logger = logging.getLogger(__name__)

    # Extract facet and day from relative path
    # Format: {facet}/entities.jsonl or {facet}/entities/YYYYMMDD.jsonl
    parts = rel.split(os.sep)
    facet = parts[0]
    day = None
    attached = 1

    if len(parts) == 3:  # {facet}/entities/YYYYMMDD.jsonl
        day = os.path.splitext(parts[2])[0]  # Remove .jsonl extension
        attached = 0

    # Parse entities from the file
    # For attached entities: path ends with entities.jsonl, parse from parent dir
    # For detected entities: path ends with YYYYMMDD.jsonl in entities/ subdir, parse directly
    from think.entities import parse_entity_file

    if attached:
        entries = parse_entities(os.path.dirname(path))
    else:
        # For detected entities, parse the specific file directly
        entries = parse_entity_file(path)

    for entity in entries:
        etype = entity.get("type", "")
        name = entity.get("name", "")
        desc = entity.get("description", "")
        conn.execute(
            "INSERT INTO entities(name, description, facet, day, type, attached) VALUES (?, ?, ?, ?, ?, ?)",
            (name, desc, facet, day, etype, attached),
        )

    if verbose:
        logger.info(
            "  indexed %s entities (facet=%s, day=%s)",
            len(entries),
            facet,
            day or "attached",
        )


def scan_entities(journal: str, verbose: bool = False) -> bool:
    """Index entities from facet-scoped ``entities.jsonl`` files."""
    logger = logging.getLogger(__name__)
    conn, _ = get_index(index="entities", journal=journal)
    files = find_entity_files(journal)

    if files:
        logger.info("\nIndexing %s entity files...", len(files))

    # Build delete SQL based on rel path structure
    def get_delete_sql(rel: str) -> str:
        parts = rel.split(os.sep)
        facet = parts[0]
        if len(parts) == 2:  # attached: {facet}/entities.jsonl
            return f"DELETE FROM entities WHERE facet='{facet}' AND day IS NULL"
        else:  # detected: {facet}/entities/YYYYMMDD.jsonl
            day = os.path.splitext(parts[2])[0]
            return f"DELETE FROM entities WHERE facet='{facet}' AND day='{day}'"

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
    facet: str | None = None,
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
    facet : str, optional
        Filter to specific facet
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

    if facet:
        where_clause += " AND facet=?"
        params.append(facet)
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
        SELECT name, description, facet, day, type, attached, bm25(entities) as rank
        FROM entities WHERE {where_clause} ORDER BY {order_by} LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    )

    results: List[Dict[str, Any]] = []
    for (
        name_val,
        desc_val,
        facet_val,
        day_val,
        type_val,
        attached_val,
        rank,
    ) in cursor.fetchall():
        # Build unique ID
        if day_val:
            result_id = f"{facet_val}/entities/{day_val}.jsonl:{name_val}"
        else:
            result_id = f"{facet_val}/entities.jsonl:{name_val}"

        results.append(
            {
                "id": result_id,
                "text": desc_val or name_val,
                "metadata": {
                    "facet": facet_val,
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
