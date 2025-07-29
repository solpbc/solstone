from __future__ import annotations

import os
from typing import Any, Dict, List

from flask import Blueprint, jsonify, render_template, request

from think.entities import Entities
from think.indexer import scan_entities, search_entities

from .. import state
from ..utils import (
    format_date,
    generate_top_summary,
    modify_entity_file,
    modify_entity_in_file,
    update_top_entry,
)

bp = Blueprint("entities", __name__, template_folder="../templates")


def reload_entities() -> None:
    """Rescan entity files and rebuild the search index."""
    ent = Entities(state.journal_root)
    ent.rescan()
    state.entities_index = ent.index()
    scan_entities(state.journal_root)


@bp.route("/entities")
def entities() -> str:
    return render_template("entities.html", active="entities")


@bp.route("/entities/api/types")
def entities_types() -> Any:
    """Return available entity types and their counts."""
    types = ["Person", "Company", "Project", "Tool"]
    data: Dict[str, int] = {}
    for t in types:
        total, _ = search_entities("", limit=0, etype=t)
        data[t] = total
    return jsonify(data)


@bp.route("/entities/api/list")
def entities_list() -> Any:
    """Return entities for a specific type ordered by count."""
    etype = request.args.get("type")
    if not etype:
        return jsonify([])
    _total_top, top_results = search_entities(
        "",
        limit=500,
        etype=etype,
        top=True,
        order="count",
    )
    _total_other, other_results = search_entities(
        "",
        limit=500,
        etype=etype,
        top=False,
        order="count",
    )
    results = []
    for r in top_results + other_results:
        meta = r["metadata"]
        results.append(
            {
                "name": meta["name"],
                "desc": r["text"],
                "top": meta.get("top", False),
                "count": meta.get("days", 0),
            }
        )
    return jsonify(results)


@bp.route("/entities/api/details")
def entities_details() -> Any:
    """Return detailed info for a single entity."""
    etype = request.args.get("type")
    name = request.args.get("name")
    if not etype or not name:
        return jsonify({})
    _total, results = search_entities(
        "",
        limit=1000,
        etype=etype,
        name=name,
        order="day",
    )
    if not results:
        return jsonify({})
    aggregated = results[0]
    top_flag = aggregated["metadata"].get("top", False)
    desc = aggregated["text"]
    descriptions: Dict[str, str] = {}
    raw_dates: List[str] = []
    for r in results[1:]:
        meta = r["metadata"]
        day = meta.get("day")
        if day:
            if day not in raw_dates:
                raw_dates.append(day)
            if r["text"]:
                descriptions[format_date(day)] = r["text"]
    raw_dates.sort()
    return jsonify(
        {
            "name": name,
            "top": top_flag,
            "desc": desc,
            "dates": [format_date(d) for d in raw_dates],
            "raw_dates": raw_dates,
            "descriptions": descriptions,
        }
    )


@bp.route("/entities/api/top_generate", methods=["POST"])
def api_top_generate() -> Any:
    payload = request.get_json(force=True)
    etype = payload.get("type")
    name = payload.get("name")
    info = state.entities_index.get(etype, {}).get(name)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or info is None:
        return ("", 400)
    try:
        desc = generate_top_summary(info, api_key)
        return jsonify({"desc": desc})
    except Exception as e:  # pragma: no cover - network errors
        return jsonify({"error": str(e)}), 500


@bp.route("/entities/api/top_update", methods=["POST"])
def api_top_update() -> Any:
    payload = request.get_json(force=True)
    etype = payload.get("type")
    name = payload.get("name")
    desc = (payload.get("desc") or "").replace("\n", " ").replace("\r", " ").strip()
    update_top_entry(state.journal_root, etype, name, desc)
    reload_entities()
    return jsonify({"status": "ok"})


@bp.route("/entities/api/remove", methods=["POST"])
@bp.route("/entities/api/rename", methods=["POST"])
def api_modify_entity() -> Any:
    payload = request.get_json(force=True)
    action = "remove" if request.path.endswith("remove") else "rename"
    days = payload.get("days", [])
    etype = payload.get("type")
    name = payload.get("name")
    new_name = payload.get("new_name") if action == "rename" else None
    for day in days:
        modify_entity_file(state.journal_root, day, etype, name, new_name, action)
    if action == "rename" and new_name:
        top_file = os.path.join(state.journal_root, "entities.md")
        try:
            modify_entity_in_file(
                top_file, etype, name, new_name, "rename", require_match=False
            )
        except Exception:
            pass
    reload_entities()
    return jsonify({"status": "ok"})
