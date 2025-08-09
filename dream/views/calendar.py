from __future__ import annotations

import os
import re
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from .. import state
from ..utils import (
    DATE_RE,
    adjacent_days,
    build_occurrence_index,
    format_date,
    list_day_folders,
)

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
    from think.utils import get_topics

    topics = get_topics()
    files = []
    topics_dir = os.path.join(day_dir, "topics")
    if os.path.isdir(topics_dir):
        for name in sorted(os.listdir(topics_dir)):
            base, ext = os.path.splitext(name)
            if ext != ".md" or base not in topics:
                continue
            path = os.path.join(topics_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                continue
            try:
                import markdown  # type: ignore

                html = markdown.markdown(text, extensions=["extra"])
            except Exception:
                html = "<p>Error loading file.</p>"
            label = base.replace("_", " ").title()
            files.append(
                {
                    "label": label,
                    "html": html,
                    "topic": base,
                    "color": topics[base]["color"],
                }
            )
    title = format_date(day)
    prev_day, next_day = adjacent_days(state.journal_root, day)
    return render_template(
        "calendar_day.html",
        active="calendar",
        title=title,
        files=files,
        prev_day=prev_day,
        next_day=next_day,
        day=day,
    )


@bp.route("/calendar/<day>/transcript")
def calendar_transcript_page(day: str) -> str:
    """Render transcript viewer for a specific day."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    title = format_date(day)
    prev_day, next_day = adjacent_days(state.journal_root, day)
    return render_template(
        "calendar_transcript.html",
        active="calendar",
        title=title,
        day=day,
        prev_day=prev_day,
        next_day=next_day,
    )


@bp.route("/calendar/api/transcript_ranges/<day>")
def calendar_transcript_ranges(day: str) -> Any:
    """Return available transcript ranges for ``day``."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    from think.cluster import cluster_scan

    audio_ranges, screen_ranges = cluster_scan(day)
    return jsonify({"audio": audio_ranges, "screen": screen_ranges})


@bp.route("/calendar/api/transcript/<day>")
def calendar_transcript_range(day: str) -> Any:
    """Return transcript markdown HTML for the selected range."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not re.fullmatch(r"\d{6}", start) or not re.fullmatch(r"\d{6}", end):
        return "", 400
    from think.cluster import cluster_range

    markdown_text = cluster_range(day, start, end, audio=True, screen="summary")
    try:
        import markdown  # type: ignore

        html_output = markdown.markdown(markdown_text, extensions=["extra"])
    except Exception:  # pragma: no cover - fallback
        import html as html_mod

        html_output = f"<pre>{html_mod.escape(markdown_text)}</pre>"
    return jsonify({"html": html_output})


@bp.route("/calendar/api/occurrences")
def calendar_occurrences() -> Any:
    if not state.occurrences_index and state.journal_root:
        state.occurrences_index = build_occurrence_index(state.journal_root)
    return jsonify(state.occurrences_index)


@bp.route("/calendar/api/days")
def calendar_days() -> Any:
    """Return list of available day folders."""

    days = list_day_folders(state.journal_root)
    return jsonify(days)
