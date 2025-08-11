"""Entity indexing and search functionality."""

import logging
import os
import re
import sqlite3
from typing import Any, Dict, List, Tuple

import sqlite_utils

from .core import _scan_files, find_day_dirs, get_index

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


def search_entities(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    day: str | None = None,
    etype: str | None = None,
    name: str | None = None,
    top: bool | None = None,
    order: str = "rank",
) -> tuple[int, List[Dict[str, Any]]]:
    """Search the entities index and return total count and results.

    Parameters
    ----------
    order : str, optional
        How to sort entity results. "rank" (default) sorts by FTS rank,
        "count" sorts by descending appearance count and "day" orders
        appearance rows chronologically.
    """

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
    if top is not None:
        where_clause += " AND top=?"
        params.append(1 if top else 0)

    total_entities = conn.execute(
        f"SELECT count(*) FROM entities WHERE {where_clause}", params
    ).fetchone()[0]

    order = order.lower()
    ent_order = "bm25(entities)"
    if order == "count":
        ent_order = "days DESC"
    ent_cursor = conn.execute(
        f"SELECT name, desc, type, top, first_seen, last_seen, days, bm25(entities) as rank FROM entities WHERE {where_clause} ORDER BY {ent_order} LIMIT ? OFFSET ?",
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

    # Skip entity_appearances search if top is specified
    if top is not None:
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
        app_order = "bm25(entity_appearances)"
        if order == "day":
            app_order = "day"
        app_cursor = conn.execute(
            f"SELECT name, desc, day, type, path, bm25(entity_appearances) as rank FROM entity_appearances WHERE {fts_clause} ORDER BY {app_order} LIMIT ? OFFSET ?",
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
