"""Combined web service for dream review apps."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request

from dream.entity_review import (
    format_date,
    generate_top_summary,
    modify_entity_file,
    modify_entity_in_file,
    update_top_entry,
)
from dream.meeting_calendar import build_index
from think.indexer import get_entities

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "review", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "review", "static"),
)

journal_root = ""
entities_index: Dict[str, Dict[str, dict]] = {}
meetings_index: Dict[str, List[Dict[str, Any]]] = {}


def reload_entities() -> None:
    global entities_index
    entities_index = get_entities(journal_root)


@app.route("/")
def home() -> str:
    return render_template("home.html", active="home")


@app.route("/entities")
def entities() -> str:
    return render_template("entities.html", active="entities")


@app.route("/calendar")
def calendar() -> str:
    return render_template("calendar.html", active="calendar")


@app.route("/entities/api/data")
def entities_data() -> Any:
    data: Dict[str, List[Dict[str, object]]] = {}
    for etype, names in entities_index.items():
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


@app.route("/entities/api/top_generate", methods=["POST"])
def api_top_generate() -> Any:
    payload = request.get_json(force=True)
    etype = payload.get("type")
    name = payload.get("name")
    info = entities_index.get(etype, {}).get(name)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or info is None:
        return ("", 400)
    try:
        desc = generate_top_summary(info, api_key)
        return jsonify({"desc": desc})
    except Exception as e:  # pragma: no cover - network errors
        return jsonify({"error": str(e)}), 500


@app.route("/entities/api/top_update", methods=["POST"])
def api_top_update() -> Any:
    payload = request.get_json(force=True)
    etype = payload.get("type")
    name = payload.get("name")
    desc = (payload.get("desc") or "").replace("\n", " ").replace("\r", " ").strip()
    update_top_entry(journal_root, etype, name, desc)
    reload_entities()
    return jsonify({"status": "ok"})


@app.route("/entities/api/remove", methods=["POST"])
@app.route("/entities/api/rename", methods=["POST"])
def api_modify_entity() -> Any:
    payload = request.get_json(force=True)
    action = "remove" if request.path.endswith("remove") else "rename"
    days = payload.get("days", [])
    etype = payload.get("type")
    name = payload.get("name")
    new_name = payload.get("new_name") if action == "rename" else None
    for day in days:
        modify_entity_file(journal_root, day, etype, name, new_name, action)
    if action == "rename" and new_name:
        top_file = os.path.join(journal_root, "entities.md")
        try:
            modify_entity_in_file(top_file, etype, name, new_name, "rename", require_match=False)
        except Exception:
            pass
    reload_entities()
    return jsonify({"status": "ok"})


@app.route("/calendar/api/meetings")
def calendar_meetings() -> Any:
    return jsonify(meetings_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined review web service")
    parser.add_argument("journal", help="Journal directory containing YYYYMMDD folders")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    args = parser.parse_args()

    global journal_root, meetings_index
    journal_root = args.journal
    reload_entities()
    meetings_index = build_index(journal_root)

    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
