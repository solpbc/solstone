from __future__ import annotations

import os
from typing import Any, Dict, List

from flask import Blueprint, jsonify, render_template, request

from think.entities import Entities

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
    ent = Entities(state.journal_root)
    ent.rescan()
    state.entities_index = ent.index()


@bp.route("/entities")
def entities() -> str:
    return render_template("entities.html", active="entities")


@bp.route("/entities/api/data")
def entities_data() -> Any:
    data: Dict[str, List[Dict[str, object]]] = {}
    for etype, names in state.entities_index.items():
        data[etype] = []
        for name, info in names.items():
            formatted_descriptions = {
                format_date(date): text for date, text in info.get("descriptions", {}).items()
            }
            data[etype].append(
                {
                    "name": name,
                    "dates": [format_date(d) for d in sorted(info.get("dates", []))],
                    "raw_dates": sorted(info.get("dates", [])),
                    "desc": info.get("primary", ""),
                    "top": info.get("top", False),
                    "descriptions": formatted_descriptions,
                }
            )
    return jsonify(data)


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
            modify_entity_in_file(top_file, etype, name, new_name, "rename", require_match=False)
        except Exception:
            pass
    reload_entities()
    return jsonify({"status": "ok"})
