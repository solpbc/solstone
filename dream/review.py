"""Combined web service for dream review apps."""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict, List

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

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
app.secret_key = os.getenv("DREAM_SECRET", "sunstone-secret")
app.config["PASSWORD"] = ""

journal_root = ""
entities_index: Dict[str, Dict[str, dict]] = {}
meetings_index: Dict[str, List[Dict[str, Any]]] = {}


def reload_entities() -> None:
    global entities_index
    entities_index = get_entities(journal_root)


@app.before_request
def require_login() -> Any:
    if request.endpoint in {"login", "static"}:
        return None
    if not session.get("logged_in"):
        return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login() -> Any:
    error = None
    if request.method == "POST":
        if request.form.get("password") == app.config.get("PASSWORD"):
            session["logged_in"] = True
            return redirect(url_for("home"))
        error = "Invalid password"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout() -> Any:
    session.pop("logged_in", None)
    return redirect(url_for("login"))


@app.route("/")
def home() -> str:
    summary_path = os.path.join(journal_root, "summary.md")
    summary_html = ""
    if os.path.isfile(summary_path):
        try:
            import markdown  # type: ignore

            with open(summary_path, "r", encoding="utf-8") as f:
                summary_html = markdown.markdown(f.read())
        except Exception:
            summary_html = "<p>Error loading summary.</p>"
    return render_template("home.html", active="home", summary_html=summary_html)


@app.route("/entities")
def entities() -> str:
    return render_template("entities.html", active="entities")


@app.route("/calendar")
def calendar() -> str:
    return render_template("calendar.html", active="calendar")


@app.route("/calendar/<day>")
def calendar_day(day: str) -> str:
    if not re.fullmatch(r"\d{8}", day):
        return "", 404
    day_dir = os.path.join(journal_root, day)
    if not os.path.isdir(day_dir):
        return "", 404
    files = []
    for name in sorted(os.listdir(day_dir)):
        if name.startswith("ponder_") and name.endswith(".md"):
            path = os.path.join(day_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                continue
            try:
                import markdown  # type: ignore

                html = markdown.markdown(text)
            except Exception:
                html = "<p>Error loading file.</p>"
            label = name[7:-3].replace("_", " ").title()
            files.append({"label": label, "html": html})
    title = format_date(day)
    days = sorted(d for d in os.listdir(journal_root) if re.fullmatch(r"\d{8}", d))
    prev_day = next_day = None
    if day in days:
        idx = days.index(day)
        if idx > 0:
            prev_day = days[idx - 1]
        if idx < len(days) - 1:
            next_day = days[idx + 1]
    return render_template(
        "day.html",
        active="calendar",
        title=title,
        files=files,
        prev_day=prev_day,
        next_day=next_day,
    )


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
    parser.add_argument(
        "--password",
        help="Password required for login (can also set DREAM_PASSWORD)",
        default=os.getenv("DREAM_PASSWORD"),
    )
    args = parser.parse_args()

    global journal_root, meetings_index
    journal_root = args.journal
    app.config["PASSWORD"] = args.password
    reload_entities()
    meetings_index = build_index(journal_root)

    if not app.config["PASSWORD"]:
        raise ValueError("Password must be provided via --password or DREAM_PASSWORD")

    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
