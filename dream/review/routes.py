from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from flask import (
    Blueprint,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from think.entities import get_entities

from . import state
from .utils import (
    DATE_RE,
    format_date,
    generate_top_summary,
    modify_entity_file,
    modify_entity_in_file,
    update_top_entry,
)

bp = Blueprint(
    "review",
    __name__,
    template_folder="templates",
    static_folder="static",
)


def reload_entities() -> None:
    from . import state

    state.entities_index = get_entities(state.journal_root)


@bp.before_app_request
def require_login() -> Any:
    if request.endpoint in {"review.login", "review.static"}:
        return None
    if not session.get("logged_in"):
        return redirect(url_for("review.login"))


@bp.route("/login", methods=["GET", "POST"])
def login() -> Any:
    error = None
    if request.method == "POST":
        if request.form.get("password") == bp.app.config.get("PASSWORD"):
            session["logged_in"] = True
            return redirect(url_for("review.home"))
        error = "Invalid password"
    return render_template("login.html", error=error)


@bp.route("/logout")
def logout() -> Any:
    session.pop("logged_in", None)
    return redirect(url_for("review.login"))


@bp.route("/")
def home() -> str:
    summary_path = os.path.join(state.journal_root, "summary.md")
    summary_html = ""
    if os.path.isfile(summary_path):
        try:
            import markdown  # type: ignore

            with open(summary_path, "r", encoding="utf-8") as f:
                summary_html = markdown.markdown(f.read())
        except Exception:
            summary_html = "<p>Error loading summary.</p>"
    return render_template("home.html", active="home", summary_html=summary_html)


@bp.route("/entities")
def entities() -> str:
    return render_template("entities.html", active="entities")


@bp.route("/calendar")
def calendar() -> str:
    return render_template("calendar.html", active="calendar")


@bp.route("/calendar/<day>")
def calendar_day(day: str) -> str:
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    day_dir = os.path.join(state.journal_root, day)
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
    days = sorted(d for d in os.listdir(state.journal_root) if re.fullmatch(DATE_RE, d))
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


@bp.route("/entities/api/data")
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


@bp.route("/entities/api/top_generate", methods=["POST"])
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


@bp.route("/calendar/api/meetings")
def calendar_meetings() -> Any:
    return jsonify(meetings_index)
