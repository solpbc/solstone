from __future__ import annotations

import json
import os
import re
from typing import Any

from flask import Blueprint, jsonify, render_template

from convey import state
from convey.utils import DATE_RE, adjacent_days, format_date
from think.utils import day_dirs, day_path

calendar_bp = Blueprint(
    "app:calendar",
    __name__,
    url_prefix="/app/calendar",
)


@calendar_bp.route("/<day>")
def calendar_day(day: str) -> str:
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return "", 404
    from think.utils import get_insights

    insights = get_insights()
    files = []
    insights_dir = os.path.join(day_dir, "insights")
    if os.path.isdir(insights_dir):
        for name in sorted(os.listdir(insights_dir)):
            base, ext = os.path.splitext(name)
            if ext != ".md" or base not in insights:
                continue
            path = os.path.join(insights_dir, name)
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
                    "color": insights[base]["color"],
                }
            )
    title = format_date(day)
    prev_day, next_day = adjacent_days(state.journal_root, day)
    return render_template(
        "app.html",
        app="calendar",
        view="day",
        title=title,
        files=files,
        prev_day=prev_day,
        next_day=next_day,
        day=day,
    )


@calendar_bp.route("/api/day/<day>/occurrences")
def calendar_day_occurrences(day: str) -> Any:
    """Return occurrences for a specific day using the events index."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    from think.indexer import search_events
    from think.utils import get_insights

    insights = get_insights()

    # Use search_events to get all events for this day
    _, results = search_events(query="", day=day, limit=1000)

    # Transform search results into timeline format (same as facet_day)
    occurrences = []
    for result in results:
        event = result.get("event", {})
        metadata = result.get("metadata", {})
        topic = metadata.get("topic", "other")

        # Add topic color
        topic_color = insights.get(topic, {}).get("color", "#6c757d")

        occurrence = {
            "title": event.get("title", ""),
            "summary": event.get("summary", ""),
            "subject": event.get("subject", ""),
            "details": event.get("details", event.get("description", "")),
            "participants": event.get("participants", []),
            "topic": topic,
            "color": topic_color,
            "facet": metadata.get("facet", ""),
        }

        # Convert time strings to ISO timestamps
        if event.get("start"):
            occurrence["startTime"] = f"{day[:4]}-{day[4:6]}-{day[6:]}T{event['start']}"
        if event.get("end"):
            occurrence["endTime"] = f"{day[:4]}-{day[4:6]}-{day[6:]}T{event['end']}"

        occurrences.append(occurrence)

    return jsonify(occurrences)


@calendar_bp.route("/api/days")
def calendar_days() -> Any:
    """Return list of available day folders."""

    days = sorted(day_dirs().keys())
    return jsonify(days)


@calendar_bp.route("/api/stats/<month>")
def calendar_stats(month: str) -> Any:
    """Return event counts per facet for a specific month.

    Args:
        month: YYYYMM format month string

    Returns:
        JSON dict mapping day (YYYYMMDD) to facet counts dict.
        Frontend handles filtering by selected facet or summing for all-facet mode.
    """
    # Validate month format (YYYYMM)
    if not re.fullmatch(r"\d{6}", month):
        return jsonify({"error": "Invalid month format, expected YYYYMM"}), 400

    stats = {}

    for name, path in day_dirs().items():
        # Filter to only days in requested month
        if not name.startswith(month):
            continue

        # Try to load stats.json from day directory
        stats_file = os.path.join(path, "stats.json")
        if os.path.isfile(stats_file):
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    day_data = json.load(f)

                facet_data = day_data.get("facet_data", {})
                # Extract just the counts per facet
                stats[name] = {
                    facet: data.get("count", 0) for facet, data in facet_data.items()
                }

            except Exception:
                stats[name] = {}
        else:
            stats[name] = {}

    return jsonify(stats)


# ============================================================================
# DEVELOPER DEBUG VIEWS - Screen JSONL Viewer
# These routes are for debugging screen.jsonl files and can be removed when
# no longer needed. Look for _dev_ prefix to identify all related code.
# ============================================================================

# In-memory cache for decoded frames: {(day, timestamp): {frame_id: jpeg_bytes}}
_frame_cache: dict = {}


@calendar_bp.route("/<day>/screens")
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
        "app.html",
        app="calendar",
        view="_dev_screens_list",
        title=title,
        day=day,
        prev_day=prev_day,
        next_day=next_day,
    )


@calendar_bp.route("/<day>/screens/<timestamp>")
def _dev_calendar_screens_detail(day: str, timestamp: str) -> str:
    """Render detail view for a specific screen.jsonl file."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    from think.utils import segment_key

    if not segment_key(timestamp):
        return "", 404

    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return "", 404

    # Check if the screen.jsonl file exists in segment
    segment_dir = os.path.join(day_dir, timestamp)
    jsonl_path = os.path.join(segment_dir, "screen.jsonl")
    if not os.path.isfile(jsonl_path):
        return "", 404

    title = format_date(day)
    prev_day, next_day = adjacent_days(state.journal_root, day)

    return render_template(
        "app.html",
        app="calendar",
        view="_dev_screens_detail",
        title=title,
        day=day,
        timestamp=timestamp,
        prev_day=prev_day,
        next_day=next_day,
    )


@calendar_bp.route("/api/screen_files/<day>")
def _dev_screen_files(day: str) -> Any:
    """Return list of screen.jsonl files for a day."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return jsonify({"files": []})

    from think.utils import segment_key

    files = []
    # Look for segments (HHMMSS/)
    for item in sorted(os.listdir(day_dir)):
        item_path = os.path.join(day_dir, item)
        if os.path.isdir(item_path) and segment_key(item):
            # Found segment, check for screen.jsonl
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


@calendar_bp.route("/api/screen_frames/<day>/<timestamp>")
def _dev_screen_frames(day: str, timestamp: str) -> Any:
    """Return all frame records and pre-cache decoded frames from video."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    from think.utils import segment_key

    if not segment_key(timestamp):
        return "", 404

    day_dir = str(day_path(day))
    segment_dir = os.path.join(day_dir, timestamp)
    jsonl_path = os.path.join(segment_dir, "screen.jsonl")

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
            # Video path is relative to segment directory (e.g., "screen.webm")
            video_path = os.path.join(segment_dir, raw_video_path)
            if os.path.isfile(video_path):
                # Use the new decode_frames utility
                images = decode_frames(video_path, frames, annotate_boxes=True)

                # Setup ArUco detector for tag overlay
                import cv2
                import numpy as np
                from PIL import ImageDraw

                dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                params = cv2.aruco.DetectorParameters()
                params.minMarkerPerimeterRate = 0.002
                params.maxMarkerPerimeterRate = 8.0
                params.adaptiveThreshWinSizeMin = 3
                params.adaptiveThreshWinSizeMax = 23
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                aruco_detector = cv2.aruco.ArucoDetector(dictionary, params)

                # Convert images to JPEG bytes and cache
                _frame_cache[cache_key] = {}
                for frame, img in zip(frames, images):
                    if img is not None:
                        frame_id = frame["frame_id"]

                        # Detect ArUco tags and draw overlays
                        img_array = np.array(img)
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        corners, ids, _ = aruco_detector.detectMarkers(gray)

                        id_to_corners = {}
                        if ids is not None:
                            for cid, pts in zip(ids.flatten().tolist(), corners):
                                id_to_corners[cid] = pts

                        # Draw green boxes and ID labels
                        if id_to_corners:
                            draw = ImageDraw.Draw(img)
                            for tag_id, pts in id_to_corners.items():
                                # pts shape is (1, 4, 2) -> reshape to (4, 2)
                                corners_2d = pts.reshape(4, 2).astype(int)
                                # Draw polygon
                                poly = [tuple(p) for p in corners_2d]
                                poly.append(poly[0])  # close the polygon
                                draw.line(poly, fill=(0, 255, 0), width=2)
                                # Draw ID label
                                cx = int(corners_2d[:, 0].mean())
                                cy = int(corners_2d[:, 1].mean())
                                draw.text(
                                    (cx + 5, cy - 10), str(tag_id), fill=(0, 255, 0)
                                )

                        jpeg_bytes = image_to_jpeg_bytes(img)
                        _frame_cache[cache_key][frame_id] = jpeg_bytes
                        img.close()

        return jsonify({"frames": frames, "raw_video_path": raw_video_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@calendar_bp.route("/api/screen_frame_image/<day>/<timestamp>/<int:frame_id>")
def _dev_screen_frame_image(day: str, timestamp: str, frame_id: int) -> Any:
    """Serve a cached frame image as JPEG."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404
    from think.utils import segment_key

    if not segment_key(timestamp):
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
