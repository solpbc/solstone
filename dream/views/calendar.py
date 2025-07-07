from __future__ import annotations

import os
import re
from typing import Any

from flask import Blueprint, jsonify, render_template

from .. import state
from ..utils import DATE_RE, adjacent_days, build_occurrence_index, format_date

bp = Blueprint("calendar", __name__, template_folder="../templates")


@bp.route("/calendar")
def calendar_page() -> str:
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
            base = name[7:-3]
            label = base.replace("_", " ").title()
            files.append({"label": label, "html": html, "slug": base})
    title = format_date(day)
    prev_day, next_day = adjacent_days(state.journal_root, day)
    return render_template(
        "day.html",
        active="calendar",
        title=title,
        files=files,
        prev_day=prev_day,
        next_day=next_day,
        day=day,
    )


@bp.route("/calendar/api/occurrences")
def calendar_occurrences() -> Any:
    if not state.occurrences_index and state.journal_root:
        state.occurrences_index = build_occurrence_index(state.journal_root)
    return jsonify(state.occurrences_index)
