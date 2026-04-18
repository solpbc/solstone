# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Graph app - knowledge graph visualization."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from think.entities.core import is_noise_entity
from think.indexer.journal import (
    get_entity_intelligence,
    get_entity_strength,
    get_journal_index,
    get_principal_entity_names,
)

graph_bp = Blueprint(
    "app:graph",
    __name__,
    url_prefix="/app/graph",
)


@graph_bp.route("/")
def index():
    return render_template("app.html")


@graph_bp.route("/api/graph")
def api_graph():
    """Return nodes + edges for the knowledge graph visualization.

    Query params:
        facet: filter by facet name
        since: YYYYMMDD start date
        types: comma-separated entity types to include
        min_strength: minimum score threshold
        limit: max nodes (default 100)
    """
    facet = request.args.get("facet") or None
    since = request.args.get("since") or None
    types_param = request.args.get("types") or None
    min_strength = request.args.get("min_strength", type=float, default=0.0)
    limit = request.args.get("limit", type=int, default=100)

    # "all" means no date filter; default to 90 days if not specified
    if since == "all":
        since = None
    elif not since:
        since = (date.today() - timedelta(days=90)).strftime("%Y%m%d")

    # Get ranked entities
    ranked = get_entity_strength(facet=facet, since=since, limit=limit * 3)

    # Look up entity type and is_principal from the identity table
    conn, _ = get_journal_index()
    try:
        entity_meta = _load_entity_metadata(conn)

        # Build nodes, applying filters
        type_filter = None
        if types_param:
            type_filter = {t.strip().lower() for t in types_param.split(",")}

        nodes = []
        node_names: set[str] = set()
        for r in ranked:
            if r["score"] < min_strength:
                continue

            entity_name = r["entity_name"]
            entity_id = r.get("entity_id") or ""
            meta = entity_meta.get(entity_id, {})
            entity_type = (meta.get("type") or "unknown").lower()

            if type_filter and entity_type not in type_filter:
                continue

            is_principal = meta.get("is_principal", False)
            nodes.append(
                {
                    "id": entity_id or entity_name,
                    "name": meta.get("name") or entity_name,
                    "type": entity_type,
                    "score": r["score"],
                    "kg_edge_count": r["kg_edge_count"],
                    "co_occurrence": r["co_occurrence"],
                    "appearance": r["appearance"],
                    "recency": r["recency"],
                    "facet_breadth": r["facet_breadth"],
                    "observation_depth": r["observation_depth"],
                    "is_principal": is_principal,
                }
            )
            node_names.add(entity_name)
            if entity_id:
                node_names.add(entity_id)

            if len(nodes) >= limit:
                break

        # Build edges
        explicit_edges = _get_explicit_edges(conn, node_names, facet, since)
        co_occurrence_edges = _get_co_occurrence_edges(
            conn, node_names, explicit_edges, facet, since
        )

        # Map edge entity names to node IDs
        node_ids = {n["id"] for n in nodes}
        name_to_id = _build_name_to_node_id(conn, node_ids, since=since, facet=facet)
        edges = []
        for e in explicit_edges + co_occurrence_edges:
            from_id = name_to_id.get(e["from_name"], e["from_name"])
            to_id = name_to_id.get(e["to_name"], e["to_name"])
            # Only include edges where both endpoints are visible nodes
            if from_id in node_ids and to_id in node_ids:
                e["from"] = from_id
                e["to"] = to_id
                edges.append(e)

        # Stats
        total_entities = conn.execute(
            "SELECT COUNT(DISTINCT entity_id) FROM entities WHERE source='identity'"
        ).fetchone()[0]
        total_signals = conn.execute("SELECT COUNT(*) FROM entity_signals").fetchone()[
            0
        ]

    finally:
        conn.close()

    # Count edges by type
    explicit_edge_count = sum(1 for e in edges if e.get("edge_type") == "explicit")
    co_occurrence_edge_count = sum(
        1 for e in edges if e.get("edge_type") == "co_occurrence"
    )

    return jsonify(
        {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_entities": total_entities,
                "total_signals": total_signals,
                "explicit_edge_count": explicit_edge_count,
                "co_occurrence_edge_count": co_occurrence_edge_count,
            },
        }
    )


@graph_bp.route("/api/entity/<path:name>")
def api_entity(name: str):
    """Return full entity intelligence for the inspect panel."""
    facet = request.args.get("facet") or None
    result = get_entity_intelligence(name, facet=facet)
    if result is None:
        return jsonify({"error": "Entity not found"}), 404
    return jsonify(result)


def _load_entity_metadata(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Load identity metadata (type, name, is_principal) for all entities."""
    rows = conn.execute(
        "SELECT entity_id, name, type, is_principal FROM entities WHERE source='identity'"
    ).fetchall()
    return {
        r[0]: {"name": r[1], "type": r[2] or "unknown", "is_principal": bool(r[3])}
        for r in rows
    }


def _get_explicit_edges(
    conn: sqlite3.Connection,
    node_names: set[str],
    facet: str | None,
    since: str | None,
) -> list[dict[str, Any]]:
    """Query kg_edge signals for explicit relationship edges."""
    if not node_names:
        return []

    placeholders = ",".join("?" for _ in node_names)
    where_parts = [
        "signal_type='kg_edge'",
        f"entity_name IN ({placeholders})",
    ]
    params: list[Any] = list(node_names)

    if facet:
        where_parts.append("facet=?")
        params.append(facet.lower())
    if since:
        where_parts.append("day>=?")
        params.append(since)

    where = " AND ".join(where_parts)
    rows = conn.execute(
        f"""
        SELECT entity_name, target_name, relationship_type, COUNT(*) as freq
        FROM entity_signals
        WHERE {where}
          AND target_name IS NOT NULL AND target_name != ''
        GROUP BY entity_name, target_name, relationship_type
        """,
        params,
    ).fetchall()

    return [
        {
            "from_name": r[0],
            "to_name": r[1],
            "relationship_type": r[2] or "",
            "frequency": r[3],
            "edge_type": "explicit",
        }
        for r in rows
        if not is_noise_entity(r[0]) and not is_noise_entity(r[1] or "")
    ]


def _get_co_occurrence_edges(
    conn: sqlite3.Connection,
    node_names: set[str],
    explicit_edges: list[dict[str, Any]],
    facet: str | None,
    since: str | None,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    """Find co-occurrence edges: entity pairs sharing paths, minus explicit edges."""
    if not node_names:
        return []

    # Build set of already-explicit pairs
    explicit_pairs: set[tuple[str, str]] = set()
    for e in explicit_edges:
        explicit_pairs.add((e["from_name"], e["to_name"]))
        explicit_pairs.add((e["to_name"], e["from_name"]))

    # Exclude principal entities — they co-occur with everyone by definition
    principal_names = get_principal_entity_names(conn)

    placeholders = ",".join("?" for _ in node_names)
    where_parts = [
        f"s1.entity_name IN ({placeholders})",
        f"s2.entity_name IN ({placeholders})",
        "s1.entity_name < s2.entity_name",
    ]
    params: list[Any] = list(node_names) + list(node_names)

    if principal_names:
        ph = ",".join("?" for _ in principal_names)
        where_parts.append(f"s1.entity_name NOT IN ({ph})")
        params.extend(principal_names)
        where_parts.append(f"s2.entity_name NOT IN ({ph})")
        params.extend(principal_names)
    if facet:
        where_parts.append("s1.facet=?")
        params.append(facet.lower())
    if since:
        where_parts.append("s1.day>=?")
        params.append(since)

    where = " AND ".join(where_parts)
    params.append(limit)
    # NOTE: A composite index on entity_signals(path, entity_name) would
    # significantly improve this self-join. See schema in think/indexer/journal.py.
    rows = conn.execute(
        f"""
        SELECT s1.entity_name, s2.entity_name, COUNT(DISTINCT s1.path) as freq
        FROM entity_signals s1
        JOIN entity_signals s2
          ON s1.path = s2.path
         AND s1.entity_name != s2.entity_name
        WHERE {where}
        GROUP BY s1.entity_name, s2.entity_name
        HAVING freq >= 2
        ORDER BY freq DESC
        LIMIT ?
        """,
        params,
    ).fetchall()

    edges = []
    for r in rows:
        if (r[0], r[1]) in explicit_pairs:
            continue
        if is_noise_entity(r[0]) or is_noise_entity(r[1]):
            continue
        edges.append(
            {
                "from_name": r[0],
                "to_name": r[1],
                "frequency": r[2],
                "edge_type": "co_occurrence",
            }
        )

    return edges


def _build_name_to_node_id(
    conn: sqlite3.Connection,
    node_ids: set[str],
    since: str | None = None,
    facet: str | None = None,
) -> dict[str, str]:
    """Map signal entity_names to node IDs (entity_ids) for edge matching.

    Uses shared name resolution (build_name_resolution_map) for consistent
    matching — the same name resolves the same way everywhere.
    """
    import json as _json

    from think.entities.matching import build_name_resolution_map

    rows = conn.execute(
        "SELECT entity_id, name, aka FROM entities WHERE source='identity'"
    ).fetchall()

    # Build entity dicts filtered to entities in node_ids
    entity_dicts: list[dict[str, Any]] = []
    for entity_id, name, aka_str in rows:
        if entity_id not in node_ids:
            continue
        d: dict[str, Any] = {"id": entity_id, "name": name, "aka": []}
        if aka_str:
            try:
                aka_list = _json.loads(aka_str)
                if isinstance(aka_list, list):
                    d["aka"] = aka_list
            except (ValueError, TypeError):
                pass
        entity_dicts.append(d)

    # Collect all names that may appear as edge endpoints
    where_parts: list[str] = []
    params: list[Any] = []
    if since:
        where_parts.append("day>=?")
        params.append(since)
    if facet:
        where_parts.append("facet=?")
        params.append(facet.lower())

    entity_name_rows = conn.execute(
        f"SELECT DISTINCT entity_name FROM entity_signals"
        f"{' WHERE ' + ' AND '.join(where_parts) if where_parts else ''}",
        params,
    ).fetchall()
    target_name_rows = conn.execute(
        f"SELECT DISTINCT target_name FROM entity_signals "
        f"WHERE target_name IS NOT NULL AND target_name != ''"
        f"{' AND ' + ' AND '.join(where_parts) if where_parts else ''}",
        params,
    ).fetchall()
    all_names = list(
        {r[0] for r in entity_name_rows} | {r[0] for r in target_name_rows}
    )

    return build_name_resolution_map(all_names, entity_dicts)
