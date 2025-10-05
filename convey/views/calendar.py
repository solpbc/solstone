from __future__ import annotations

import json
import os
import re
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from think.utils import day_dirs

from .. import state
from ..utils import (
    DATE_RE,
    adjacent_days,
    build_occurrence_index,
    format_date,
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

            day_dir = str(day_path(day))
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

    from think.cluster import _date_str, _load_entries
    from think.utils import day_path

    day_dir = str(day_path(day))
    # day_path now ensures dir exists, but check anyway
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

    from think.cluster import _date_str, _load_entries
    from think.utils import day_path, get_raw_file

    day_dir = str(day_path(day))
    # day_path now ensures dir exists, but check anyway
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


@bp.route("/calendar/api/download_audio/<day>")
def download_audio(day: str) -> Any:
    """Download concatenated MP3 of audio files for a time range."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not re.fullmatch(r"\d{6}", start) or not re.fullmatch(r"\d{6}", end):
        return "", 400

    import subprocess
    import tempfile
    from datetime import datetime
    from pathlib import Path

    from flask import send_file

    from think.cluster import _date_str, _load_entries
    from think.utils import day_path

    day_dir = str(day_path(day))
    # day_path now ensures dir exists, but check anyway
    if not os.path.isdir(day_dir):
        return jsonify({"error": "Day directory not found"}), 404

    date_str = _date_str(day_dir)
    start_dt = datetime.strptime(date_str + start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(date_str + end, "%Y%m%d%H%M%S")

    # Load audio entries only
    entries = _load_entries(day_dir, audio=True, screen_mode=None)

    # Filter to time range
    audio_files = []
    for e in entries:
        if e.get("prefix") == "audio" and start_dt <= e["timestamp"] < end_dt:
            # Get the raw FLAC file path
            raw_name = e["name"].replace("_audio.json", "_raw.flac")
            flac_path = os.path.join(day_dir, "heard", raw_name)
            if os.path.isfile(flac_path):
                audio_files.append(flac_path)

    if not audio_files:
        return jsonify({"error": "No audio files found in the selected range"}), 404

    # Generate filename
    start_hhmm = start_dt.strftime("%H%M")
    end_hhmm = end_dt.strftime("%H%M")
    filename = f"sunstone_{day}_{start_hhmm}-{end_hhmm}.mp3"

    # Format date for metadata
    date_obj = datetime.strptime(day, "%Y%m%d")
    date_formatted = date_obj.strftime("%B %d, %Y")
    time_range = f"{start_dt.strftime('%I:%M %p')} - {end_dt.strftime('%I:%M %p')}"

    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            if len(audio_files) == 1:
                # Single file - convert directly to MP3
                output_mp3 = temp_path / filename
                cmd = [
                    "ffmpeg",
                    "-i",
                    audio_files[0],
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    "-metadata",
                    f"title=Sunstone Recording - {date_formatted} {time_range}",
                    "-metadata",
                    "album=Sunstone Journal",
                    "-metadata",
                    f"date={date_obj.year}",
                    "-metadata",
                    f"comment=Time range: {time_range}",
                    "-y",
                    str(output_mp3),
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            else:
                # Multiple files - create concat list and merge
                concat_file = temp_path / "concat.txt"
                with open(concat_file, "w") as f:
                    for flac_file in audio_files:
                        # FFmpeg concat format requires escaping
                        escaped_path = str(flac_file).replace("'", "'\\''")
                        f.write(f"file '{escaped_path}'\n")

                output_mp3 = temp_path / filename
                cmd = [
                    "ffmpeg",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_file),
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    "-metadata",
                    f"title=Sunstone Recording - {date_formatted} {time_range}",
                    "-metadata",
                    "album=Sunstone Journal",
                    "-metadata",
                    f"date={date_obj.year}",
                    "-metadata",
                    f"comment=Time range: {time_range}",
                    "-y",
                    str(output_mp3),
                ]
                subprocess.run(cmd, check=True, capture_output=True)

            # Send the file
            return send_file(
                str(output_mp3),
                mimetype="audio/mpeg",
                as_attachment=True,
                download_name=filename,
            )

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@bp.route("/calendar/api/occurrences")
def calendar_occurrences() -> Any:
    from datetime import date

    today_str = date.today().strftime("%Y%m%d")

    # Check if we need to rebuild or update the index
    needs_rebuild = False
    if not state.occurrences_index:
        # No index exists yet
        needs_rebuild = True
    elif state.occurrences_index_date != today_str:
        # Date has changed since last build - rebuild for simplicity
        # Could optimize this to only add new days, but full rebuild ensures consistency
        needs_rebuild = True

    if needs_rebuild and state.journal_root:
        state.occurrences_index = build_occurrence_index(state.journal_root)
        state.occurrences_index_date = today_str
        state.occurrences_index_days = set(state.occurrences_index.keys())
    elif state.journal_root:
        # Check for new days that aren't in the index yet (incremental update)
        current_days = set(day_dirs().keys())

        # Find new days not in the index
        new_days = current_days - state.occurrences_index_days
        if new_days:
            # Load only the new days
            from think.utils import get_topics

            topics = get_topics()

            for day in new_days:
                day_path = os.path.join(state.journal_root, day)
                topics_dir = os.path.join(day_path, "topics")

                if os.path.isdir(topics_dir):
                    occs = []
                    for fname in os.listdir(topics_dir):
                        base, ext = os.path.splitext(fname)
                        if ext != ".json" or base not in topics:
                            continue
                        file_path = os.path.join(topics_dir, fname)
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            items = (
                                data.get("occurrences", [])
                                if isinstance(data, dict)
                                else data
                            )
                        except Exception:
                            continue
                        if not isinstance(items, list):
                            continue

                        topic = base
                        for occ in items:
                            occs.append(
                                {
                                    "title": occ.get("title", ""),
                                    "summary": occ.get("summary", ""),
                                    "subject": occ.get("subject", ""),
                                    "details": occ.get(
                                        "details", occ.get("description", "")
                                    ),
                                    "participants": occ.get("participants", []),
                                    "topic": topic,
                                    "color": topics[topic]["color"],
                                    "path": os.path.join(day, "topics", fname),
                                }
                            )

                    if occs:
                        state.occurrences_index[day] = occs

                # Add to tracked days even if no occurrences (to avoid rechecking)
                state.occurrences_index_days.add(day)

    return jsonify(state.occurrences_index)


@bp.route("/calendar/api/days")
def calendar_days() -> Any:
    """Return list of available day folders."""

    days = sorted(day_dirs().keys())
    return jsonify(days)


@bp.route("/calendar/api/stats")
def calendar_stats() -> Any:
    """Return lightweight stats for calendar display."""
    import os
    from datetime import datetime

    if not state.journal_root:
        return jsonify({})

    today = datetime.now().strftime("%Y%m%d")
    stats = {}

    # Get all days and their occurrence counts
    all_days = []
    for name, path in day_dirs().items():
        day_stats = {
            "day": name,
            "has_transcripts": False,
            "has_todos": False,
            "has_topics": False,
            "occurrence_count": 0,
        }

        # Check for transcripts (audio json files)
        for fname in os.listdir(path):
            if fname.endswith("_audio.json"):
                day_stats["has_transcripts"] = True
                break

        # Check for todos
        todos_path = os.path.join(path, "todos", "today.md")
        if os.path.exists(todos_path):
            day_stats["has_todos"] = True

        # Check for topics and count occurrences
        topics_dir = os.path.join(path, "topics")
        if os.path.isdir(topics_dir):
            # Check if any topic files exist (json or md)
            topic_files = [
                f
                for f in os.listdir(topics_dir)
                if (f.endswith(".json") or f.endswith(".md"))
                and not f.endswith(".crumb")
            ]
            if topic_files:
                day_stats["has_topics"] = True

            # Count occurrences from JSON files
            for fname in os.listdir(topics_dir):
                if fname.endswith(".json") and not fname.endswith(".crumb"):
                    try:
                        file_path = os.path.join(topics_dir, fname)
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        items = (
                            data.get("occurrences", [])
                            if isinstance(data, dict)
                            else data if isinstance(data, list) else []
                        )
                        day_stats["occurrence_count"] += len(items)
                    except Exception:
                        continue

        all_days.append(day_stats)
        stats[name] = day_stats

    # Calculate percentiles for past days only
    past_days_with_data = [
        d for d in all_days if d["day"] < today and d["occurrence_count"] > 0
    ]

    if past_days_with_data:
        # Sort by occurrence count
        sorted_days = sorted(past_days_with_data, key=lambda x: x["occurrence_count"])
        total = len(sorted_days)

        # Calculate thresholds
        bottom_20_idx = int(total * 0.2)
        top_20_idx = int(total * 0.8)

        for day in sorted_days[:bottom_20_idx]:
            stats[day["day"]]["activity_level"] = "low"

        for day in sorted_days[bottom_20_idx:top_20_idx]:
            stats[day["day"]]["activity_level"] = "medium"

        for day in sorted_days[top_20_idx:]:
            stats[day["day"]]["activity_level"] = "high"

    # Mark days with no data as "none" (but only past days)
    for day_name, day_stats in stats.items():
        if day_name < today:
            if day_stats["occurrence_count"] == 0:
                day_stats["activity_level"] = "none"
        # Today and future days get no activity_level

    return jsonify(stats)
