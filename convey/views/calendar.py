from __future__ import annotations

import json
import os
import re
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from think.utils import day_dirs, day_path

from .. import state
from ..utils import DATE_RE, adjacent_days, format_date

bp = Blueprint("calendar", __name__, template_folder="../templates")


@bp.route("/calendar")
def calendar_page() -> str:
    return render_template("calendar.html", active="calendar")


@bp.route("/calendar/<day>")
def calendar_day(day: str) -> str:
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    day_dir = str(day_path(day))
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


@bp.route("/calendar/api/day/<day>/occurrences")
def calendar_day_occurrences(day: str) -> Any:
    """Return occurrences for a specific day using the events index."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    from think.indexer import search_events
    from think.utils import get_topics

    topics = get_topics()

    # Use search_events to get all events for this day
    _, results = search_events(query="", day=day, limit=1000)

    # Transform search results into timeline format (same as facet_day)
    occurrences = []
    for result in results:
        event = result.get("event", {})
        metadata = result.get("metadata", {})
        topic = metadata.get("topic", "other")

        # Add topic color
        topic_color = topics.get(topic, {}).get("color", "#6c757d")

        occurrence = {
            "title": event.get("title", ""),
            "summary": event.get("summary", ""),
            "subject": event.get("subject", ""),
            "details": event.get("details", event.get("description", "")),
            "participants": event.get("participants", []),
            "topic": topic,
            "color": topic_color,
        }

        # Convert time strings to ISO timestamps
        if event.get("start"):
            occurrence["startTime"] = f"{day[:4]}-{day[4:6]}-{day[6:]}T{event['start']}"
        if event.get("end"):
            occurrence["endTime"] = f"{day[:4]}-{day[4:6]}-{day[6:]}T{event['end']}"

        occurrences.append(occurrence)

    return jsonify(occurrences)


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
    from think.utils import period_key

    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not period_key(start) or not period_key(end):
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

        html_output = markdown.markdown(markdown_text, extensions=["extra", "nl2br"])
    except Exception:  # pragma: no cover - fallback
        import html as html_mod

        html_output = f"<pre>{html_mod.escape(markdown_text)}</pre>"
    return jsonify({"html": html_output})


@bp.route("/calendar/api/raw_files/<day>")
def calendar_raw_files(day: str) -> Any:
    """Return raw file timestamps for the selected range."""

    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    from think.utils import period_key

    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not period_key(start) or not period_key(end):
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
    from think.utils import period_key

    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not period_key(start) or not period_key(end):
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
        day_dir = str(day_path(day))
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

    from think.utils import period_key

    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not period_key(start) or not period_key(end):
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
            # Get the raw FLAC file path from metadata
            try:
                from think.utils import get_raw_file

                rel_path, mime_type, metadata = get_raw_file(day, e["name"])
                flac_path = os.path.join(day_dir, rel_path)
                if os.path.isfile(flac_path):
                    audio_files.append(flac_path)
            except (ValueError, Exception):
                # Skip files without valid raw field
                continue

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


@bp.route("/calendar/api/days")
def calendar_days() -> Any:
    """Return list of available day folders."""

    days = sorted(day_dirs().keys())
    return jsonify(days)


@bp.route("/calendar/api/facets/active")
def calendar_active_facets() -> Any:
    """Return list of active (non-disabled) facets for calendar filtering."""
    from think.facets import get_facets

    all_facets = get_facets()
    active_facets = []

    for name, data in all_facets.items():
        if not data.get("disabled", False):
            active_facets.append(
                {
                    "name": name,
                    "title": data.get("title", name),
                    "color": data.get("color", ""),
                    "emoji": data.get("emoji", ""),
                }
            )

    return jsonify(active_facets)


@bp.route("/calendar/api/stats")
def calendar_stats() -> Any:
    """Return lightweight stats for calendar display."""
    import os

    if not state.journal_root:
        return jsonify({})

    # Get optional facet filter from query params
    facet_filter = request.args.get("facet", "")

    stats = {}

    for name, path in day_dirs().items():
        day_stats = {
            "day": name,
            "has_transcripts": False,
            "has_todos": False,
            "has_topics": False,
            "occurrence_count": 0,
        }

        # Try to load stats.json from day directory
        stats_file = os.path.join(path, "stats.json")
        if os.path.isfile(stats_file):
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    day_data = json.load(f)

                # Extract stats
                stats_obj = day_data.get("stats", {})
                topic_data = day_data.get("topic_data", {})

                # has_transcripts: check if any audio sessions exist
                day_stats["has_transcripts"] = stats_obj.get("audio_sessions", 0) > 0

                # has_topics: check if topics were processed or topic_data exists
                day_stats["has_topics"] = (
                    stats_obj.get("topics_processed", 0) > 0 or len(topic_data) > 0
                )

                # occurrence_count: sum all topic occurrence counts
                day_stats["occurrence_count"] = sum(
                    topic.get("count", 0) for topic in topic_data.values()
                )

            except Exception:
                # If stats.json can't be read, leave defaults (all False/0)
                pass

        # Check for todos - filter by facet if specified
        from think.todo import get_facets_with_todos

        facets_with_todos = get_facets_with_todos(name)
        if facet_filter:
            # Filter to specific facet
            if facet_filter in facets_with_todos:
                day_stats["has_todos"] = True
        else:
            # Show if any facet has todos
            if facets_with_todos:
                day_stats["has_todos"] = True

        stats[name] = day_stats

    return jsonify(stats)


# ============================================================================
# DEVELOPER DEBUG VIEWS - Screen JSONL Viewer
# These routes are for debugging screen.jsonl files and can be removed when
# no longer needed. Look for _dev_ prefix to identify all related code.
# ============================================================================

# In-memory cache for decoded frames: {(day, timestamp): {frame_id: jpeg_bytes}}
_frame_cache: dict = {}


@bp.route("/calendar/<day>/screens")
def _dev_calendar_screens_list(day: str) -> str:
    """Render list of screen.jsonl files for a specific day."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return "", 404

    title = format_date(day)
    prev_day, next_day = adjacent_days(state.journal_root, day)

    return render_template(
        "_dev_calendar_screens_list.html",
        active="calendar",
        title=title,
        day=day,
        prev_day=prev_day,
        next_day=next_day,
    )


@bp.route("/calendar/<day>/screens/<timestamp>")
def _dev_calendar_screens_detail(day: str, timestamp: str) -> str:
    """Render detail view for a specific screen.jsonl file."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    from think.utils import period_key

    if not period_key(timestamp):
        return "", 404

    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return "", 404

    # Check if the screen.jsonl file exists in period
    period_dir = os.path.join(day_dir, timestamp)
    jsonl_path = os.path.join(period_dir, "screen.jsonl")
    if not os.path.isfile(jsonl_path):
        return "", 404

    title = format_date(day)
    prev_day, next_day = adjacent_days(state.journal_root, day)

    return render_template(
        "_dev_calendar_screens_detail.html",
        active="calendar",
        title=title,
        day=day,
        timestamp=timestamp,
        prev_day=prev_day,
        next_day=next_day,
    )


@bp.route("/calendar/api/screen_files/<day>")
def _dev_screen_files(day: str) -> Any:
    """Return list of screen.jsonl files for a day."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return jsonify({"files": []})

    from think.utils import period_key

    files = []
    # Look for periods (HHMMSS/)
    for item in sorted(os.listdir(day_dir)):
        item_path = os.path.join(day_dir, item)
        if os.path.isdir(item_path) and period_key(item):
            # Found period, check for screen.jsonl
            jsonl_path = os.path.join(item_path, "screen.jsonl")
            if os.path.isfile(jsonl_path):
                timestamp = item

                # Count frames (excluding header line)
                frame_count = 0
                file_size = 0
                try:
                    file_size = os.path.getsize(jsonl_path)
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            if line_num > 1:  # Skip header
                                frame_count += 1
                except Exception:
                    continue

                # Format timestamp as human-readable time
                from datetime import datetime

                try:
                    time_obj = datetime.strptime(timestamp, "%H%M%S")
                    human_time = time_obj.strftime("%I:%M:%S %p").lstrip("0")
                except Exception:
                    human_time = timestamp

                files.append(
                    {
                        "timestamp": timestamp,
                        "human_time": human_time,
                        "frame_count": frame_count,
                        "file_size": file_size,
                    }
                )

    return jsonify({"files": files})


@bp.route("/calendar/api/screen_frames/<day>/<timestamp>")
def _dev_screen_frames(day: str, timestamp: str) -> Any:
    """Return all frame records and pre-cache decoded frames from video."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    from think.utils import period_key

    if not period_key(timestamp):
        return "", 404

    day_dir = str(day_path(day))
    period_dir = os.path.join(day_dir, timestamp)
    jsonl_path = os.path.join(period_dir, "screen.jsonl")

    if not os.path.isfile(jsonl_path):
        return "", 404

    try:
        from observe.see import decode_frames, image_to_jpeg_bytes
        from observe.utils import load_analysis_frames

        all_frames = load_analysis_frames(jsonl_path)

        # The first line is a header with only {"raw": "path"}, filter it out
        # Real frames have frame_id field
        frames = [f for f in all_frames if "frame_id" in f]

        # Extract raw video path from header (first item if it only has "raw" key)
        raw_video_path = None
        if all_frames and "raw" in all_frames[0] and "frame_id" not in all_frames[0]:
            raw_video_path = all_frames[0].get("raw")

        # Decode and cache all frames from the video
        cache_key = (day, timestamp)
        if cache_key not in _frame_cache and raw_video_path:
            video_path = os.path.join(day_dir, raw_video_path)
            if os.path.isfile(video_path):
                # Use the new decode_frames utility
                images = decode_frames(video_path, frames, annotate_boxes=True)

                # Convert images to JPEG bytes and cache
                _frame_cache[cache_key] = {}
                for frame, img in zip(frames, images):
                    if img is not None:
                        frame_id = frame["frame_id"]
                        jpeg_bytes = image_to_jpeg_bytes(img)
                        _frame_cache[cache_key][frame_id] = jpeg_bytes
                        img.close()

        return jsonify({"frames": frames, "raw_video_path": raw_video_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/calendar/api/screen_frame_image/<day>/<timestamp>/<int:frame_id>")
def _dev_screen_frame_image(day: str, timestamp: str, frame_id: int) -> Any:
    """Serve a cached frame image as JPEG."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    from think.utils import period_key

    if not period_key(timestamp):
        return "", 404

    try:
        import io

        from flask import send_file

        # Check cache
        cache_key = (day, timestamp)
        if cache_key in _frame_cache and frame_id in _frame_cache[cache_key]:
            jpeg_bytes = _frame_cache[cache_key][frame_id]
            buffer = io.BytesIO(jpeg_bytes)
            buffer.seek(0)
            return send_file(buffer, mimetype="image/jpeg")

        # Frame not in cache
        return jsonify({"error": "Frame not cached. Load the frames list first."}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500
