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

    # Get checkbox states from query params
    audio_enabled = request.args.get("audio", "true").lower() == "true"
    screen_enabled = request.args.get("screen", "true").lower() == "true"

    if not audio_enabled and not screen_enabled:
        markdown_text = "*Please select at least one source (Audio or Screen)*"
    else:
        from think.cluster import cluster_range

        # Call cluster_range with appropriate parameters based on checkboxes
        if audio_enabled and screen_enabled:
            # Both sources selected - get full content
            markdown_text = cluster_range(day, start, end, audio=True, screen="summary")
        elif audio_enabled and not screen_enabled:
            # Only audio selected - custom logic to exclude screen content
            from datetime import datetime

            from think.cluster import (
                _date_str,
                _group_entries,
                _groups_to_markdown,
                _load_entries,
                day_path,
            )

            day_dir = day_path(day)
            date_str = _date_str(day_dir)
            start_dt = datetime.strptime(date_str + start, "%Y%m%d%H%M%S")
            end_dt = datetime.strptime(date_str + end, "%Y%m%d%H%M%S")

            # Load with audio=True and screen="raw" to get entries
            entries = _load_entries(day_dir, True, "raw")
            # Filter to only audio entries within time range
            entries = [
                e
                for e in entries
                if e.get("prefix") == "audio" and start_dt <= e["timestamp"] < end_dt
            ]
            groups = _group_entries(entries)
            markdown_text = _groups_to_markdown(groups)
        elif not audio_enabled and screen_enabled:
            # Only screen selected - get screen without audio
            markdown_text = cluster_range(
                day, start, end, audio=False, screen="summary"
            )
        else:
            # This case is already handled above
            markdown_text = ""
    try:
        import markdown  # type: ignore

        html_output = markdown.markdown(markdown_text, extensions=["extra"])
    except Exception:  # pragma: no cover - fallback
        import html as html_mod

        html_output = f"<pre>{html_mod.escape(markdown_text)}</pre>"
    return jsonify({"html": html_output})


@bp.route("/calendar/api/raw_files/<day>")
def calendar_raw_files(day: str) -> Any:
    """Return raw file timestamps for the selected range."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not re.fullmatch(r"\d{6}", start) or not re.fullmatch(r"\d{6}", end):
        return "", 400

    file_type = request.args.get("type", None)  # 'audio', 'screen', or None for both

    from datetime import datetime

    from think.cluster import _date_str, _load_entries, day_path

    day_dir = day_path(day)
    if not os.path.isdir(day_dir):
        return jsonify({"files": []})

    date_str = _date_str(day_dir)
    start_dt = datetime.strptime(date_str + start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(date_str + end, "%Y%m%d%H%M%S")

    # Load entries based on type filter
    if file_type == "audio":
        entries = _load_entries(day_dir, audio=True, screen_mode=None)
    elif file_type == "screen":
        entries = _load_entries(day_dir, audio=False, screen_mode="raw")
    else:  # Load both
        entries = _load_entries(day_dir, audio=True, screen_mode="raw")

    # Filter to time range and extract timestamps
    files = []
    for e in entries:
        if start_dt <= e["timestamp"] < end_dt:
            # Convert timestamp to minutes since midnight for easier rendering
            minutes = e["timestamp"].hour * 60 + e["timestamp"].minute
            # For screen files, prefix could be "screen" (summary) or source name (raw)
            file_type_str = "audio" if e["prefix"] == "audio" else "screen"
            files.append(
                {
                    "minute": minutes,
                    "type": file_type_str,
                    "time": e["timestamp"].strftime("%H:%M:%S"),
                }
            )

    return jsonify({"files": files})


@bp.route("/calendar/api/media_files/<day>")
def calendar_media_files(day: str) -> Any:
    """Return actual media files for embedding in the selected range."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not re.fullmatch(r"\d{6}", start) or not re.fullmatch(r"\d{6}", end):
        return "", 400

    file_type = request.args.get("type", None)  # 'audio', 'screen', or None for both

    from datetime import datetime

    from think.cluster import _date_str, _load_entries, day_path
    from think.utils import get_raw_file

    day_dir = day_path(day)
    if not os.path.isdir(day_dir):
        return jsonify({"media": []})

    date_str = _date_str(day_dir)
    start_dt = datetime.strptime(date_str + start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(date_str + end, "%Y%m%d%H%M%S")

    # Load entries based on type filter
    if file_type == "audio":
        entries = _load_entries(day_dir, audio=True, screen_mode=None)
    elif file_type == "screen":
        entries = _load_entries(day_dir, audio=False, screen_mode="raw")
    else:  # Load both
        entries = _load_entries(day_dir, audio=True, screen_mode="raw")

    # Filter to time range and get raw file info
    media = []
    for e in entries:
        if start_dt <= e["timestamp"] < end_dt:
            # Get the raw file info using get_raw_file
            try:
                rel_path, mime_type, metadata = get_raw_file(day, e["name"])

                # Create a URL path for serving the file
                file_url = (
                    f"/calendar/api/serve_file/{day}/{rel_path.replace('/', '__')}"
                )

                # For screen files, prefix could be "screen" (summary) or source name (raw)
                file_type_str = "audio" if e["prefix"] == "audio" else "screen"
                human_time = e["timestamp"].strftime("%I:%M:%S %p").lstrip("0")

                media.append(
                    {
                        "url": file_url,
                        "type": file_type_str,
                        "mime_type": mime_type,
                        "time": e["timestamp"].strftime("%H:%M:%S"),
                        "human_time": human_time,
                        "timestamp": e["timestamp"].isoformat(),
                        "metadata": metadata,
                    }
                )
            except Exception:
                # Skip files that can't be processed
                continue

    # Sort by timestamp
    media.sort(key=lambda x: x["timestamp"])

    return jsonify({"media": media})


@bp.route("/calendar/api/serve_file/<day>/<path:encoded_path>")
def serve_media_file(day: str, encoded_path: str) -> Any:
    """Serve actual media files for embedding."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    try:
        # Decode the path (replace '__' back to '/')
        rel_path = encoded_path.replace("__", "/")

        # Construct the full file path
        full_path = os.path.join(state.journal_root, day, rel_path)

        # Security check: ensure the path is within the day directory
        day_dir = os.path.join(state.journal_root, day)
        if not os.path.commonpath([full_path, day_dir]) == day_dir:
            return "", 403

        # Check if file exists
        if not os.path.isfile(full_path):
            return "", 404

        from flask import send_file

        return send_file(full_path)

    except Exception:
        return "", 404


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
