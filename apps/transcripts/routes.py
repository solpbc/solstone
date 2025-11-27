"""Transcript viewer app - browse and playback daily transcripts."""

from __future__ import annotations

import os
import re
from datetime import date, datetime
from typing import Any

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from convey import state
from convey.utils import DATE_RE, adjacent_days, format_date
from think.utils import day_path

transcripts_bp = Blueprint(
    "app:transcripts",
    __name__,
    url_prefix="/app/transcripts",
)


@transcripts_bp.route("/")
def index() -> Any:
    """Redirect to today's transcripts."""
    today = date.today().strftime("%Y%m%d")
    return redirect(url_for("app:transcripts.transcripts_day", day=today))


@transcripts_bp.route("/<day>")
def transcripts_day(day: str) -> str:
    """Render transcript viewer for a specific day."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    title = format_date(day)
    prev_day, next_day = adjacent_days(state.journal_root, day)

    return render_template(
        "app.html",
        app="transcripts",
        title=title,
        day=day,
        prev_day=prev_day,
        next_day=next_day,
    )


@transcripts_bp.route("/api/ranges/<day>")
def transcript_ranges(day: str) -> Any:
    """Return available transcript ranges for a day."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    from think.cluster import cluster_scan

    audio_ranges, screen_ranges = cluster_scan(day)
    return jsonify({"audio": audio_ranges, "screen": screen_ranges})


@transcripts_bp.route("/api/content/<day>")
def transcript_content(day: str) -> Any:
    """Return transcript markdown HTML for the selected range."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    from think.utils import segment_key

    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not segment_key(start) or not segment_key(end):
        return "", 400

    audio_enabled = request.args.get("audio", "true").lower() == "true"
    screen_enabled = request.args.get("screen", "true").lower() == "true"

    if not audio_enabled and not screen_enabled:
        markdown_text = "*Please select at least one source (Audio or Screen)*"
    elif audio_enabled and screen_enabled:
        from think.cluster import cluster_range

        markdown_text = cluster_range(day, start, end, audio=True, screen="summary")
    elif audio_enabled:
        # Audio only - exclude screen content
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

        entries = _load_entries(day_dir, True, "raw")
        entries = [
            e
            for e in entries
            if e.get("prefix") == "audio" and start_dt <= e["timestamp"] < end_dt
        ]
        groups = _group_entries(entries)
        markdown_text = _groups_to_markdown(groups)
    else:
        # Screen only
        from think.cluster import cluster_range

        markdown_text = cluster_range(day, start, end, audio=False, screen="summary")

    try:
        import markdown

        html_output = markdown.markdown(markdown_text, extensions=["extra", "nl2br"])
    except Exception:
        import html as html_mod

        html_output = f"<pre>{html_mod.escape(markdown_text)}</pre>"

    return jsonify({"html": html_output})


@transcripts_bp.route("/api/raw_files/<day>")
def raw_files(day: str) -> Any:
    """Return raw file timestamps for the selected range."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    from think.utils import segment_key

    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not segment_key(start) or not segment_key(end):
        return "", 400

    file_type = request.args.get("type", None)

    from think.cluster import _date_str, _load_entries

    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return jsonify({"files": []})

    date_str = _date_str(day_dir)
    start_dt = datetime.strptime(date_str + start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(date_str + end, "%Y%m%d%H%M%S")

    if file_type == "audio":
        entries = _load_entries(day_dir, audio=True, screen_mode=None)
    elif file_type == "screen":
        entries = _load_entries(day_dir, audio=False, screen_mode="raw")
    else:
        entries = _load_entries(day_dir, audio=True, screen_mode="raw")

    files = []
    for e in entries:
        if start_dt <= e["timestamp"] < end_dt:
            minutes = e["timestamp"].hour * 60 + e["timestamp"].minute
            file_type_str = "audio" if e["prefix"] == "audio" else "screen"
            files.append(
                {
                    "minute": minutes,
                    "type": file_type_str,
                    "time": e["timestamp"].strftime("%H:%M:%S"),
                }
            )

    return jsonify({"files": files})


@transcripts_bp.route("/api/media_files/<day>")
def media_files(day: str) -> Any:
    """Return actual media files for embedding in the selected range."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    from think.utils import get_raw_file, segment_key

    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not segment_key(start) or not segment_key(end):
        return "", 400

    file_type = request.args.get("type", None)

    from think.cluster import _date_str, _load_entries

    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return jsonify({"media": []})

    date_str = _date_str(day_dir)
    start_dt = datetime.strptime(date_str + start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(date_str + end, "%Y%m%d%H%M%S")

    if file_type == "audio":
        entries = _load_entries(day_dir, audio=True, screen_mode=None)
    elif file_type == "screen":
        entries = _load_entries(day_dir, audio=False, screen_mode="raw")
    else:
        entries = _load_entries(day_dir, audio=True, screen_mode="raw")

    media = []
    for e in entries:
        if start_dt <= e["timestamp"] < end_dt:
            try:
                rel_path, mime_type, metadata = get_raw_file(day, e["name"])
                file_url = f"/app/transcripts/api/serve_file/{day}/{rel_path.replace('/', '__')}"
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
                continue

    media.sort(key=lambda x: x["timestamp"])
    return jsonify({"media": media})


@transcripts_bp.route("/api/serve_file/<day>/<path:encoded_path>")
def serve_file(day: str, encoded_path: str) -> Any:
    """Serve actual media files for embedding."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    try:
        rel_path = encoded_path.replace("__", "/")
        full_path = os.path.join(state.journal_root, day, rel_path)

        day_dir = str(day_path(day))
        if not os.path.commonpath([full_path, day_dir]) == day_dir:
            return "", 403

        if not os.path.isfile(full_path):
            return "", 404

        from flask import send_file

        return send_file(full_path)

    except Exception:
        return "", 404


@transcripts_bp.route("/api/download_audio/<day>")
def download_audio(day: str) -> Any:
    """Download concatenated MP3 of audio files for a time range."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    from think.utils import segment_key

    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not segment_key(start) or not segment_key(end):
        return "", 400

    import subprocess
    import tempfile
    from pathlib import Path

    from flask import send_file

    from think.cluster import _date_str, _load_entries
    from think.utils import get_raw_file

    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return jsonify({"error": "Day directory not found"}), 404

    date_str = _date_str(day_dir)
    start_dt = datetime.strptime(date_str + start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(date_str + end, "%Y%m%d%H%M%S")

    entries = _load_entries(day_dir, audio=True, screen_mode=None)

    audio_files = []
    for e in entries:
        if e.get("prefix") == "audio" and start_dt <= e["timestamp"] < end_dt:
            try:
                rel_path, mime_type, metadata = get_raw_file(day, e["name"])
                flac_path = os.path.join(day_dir, rel_path)
                if os.path.isfile(flac_path):
                    audio_files.append(flac_path)
            except (ValueError, Exception):
                continue

    if not audio_files:
        return jsonify({"error": "No audio files found in the selected range"}), 404

    start_hhmm = start_dt.strftime("%H%M")
    end_hhmm = end_dt.strftime("%H%M")
    filename = f"sunstone_{day}_{start_hhmm}-{end_hhmm}.mp3"

    date_obj = datetime.strptime(day, "%Y%m%d")
    date_formatted = date_obj.strftime("%B %d, %Y")
    time_range = f"{start_dt.strftime('%I:%M %p')} - {end_dt.strftime('%I:%M %p')}"

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_mp3 = temp_path / filename

            if len(audio_files) == 1:
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
                concat_file = temp_path / "concat.txt"
                with open(concat_file, "w") as f:
                    for flac_file in audio_files:
                        escaped_path = str(flac_file).replace("'", "'\\''")
                        f.write(f"file '{escaped_path}'\n")

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
