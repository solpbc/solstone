# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

from flask import Blueprint, jsonify, redirect, render_template, url_for

from convey import state
from convey.utils import DATE_RE, format_date
from observe.utils import VIDEO_EXTENSIONS
from think.utils import day_path

calendar_bp = Blueprint(
    "app:calendar",
    __name__,
    url_prefix="/app/calendar",
)


@calendar_bp.route("/")
def index():
    """Redirect to yesterday's calendar view."""
    from datetime import timedelta

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    return redirect(url_for("app:calendar.calendar_day", day=yesterday))


@calendar_bp.route("/<day>")
def calendar_day(day: str) -> str:
    """Render events timeline for a specific day."""
    if not DATE_RE.fullmatch(day):
        return "", 404
    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return "", 404

    title = format_date(day)

    return render_template(
        "app.html",
        view="day",
        title=title,
    )


@calendar_bp.route("/api/day/<day>/events")
def calendar_day_events(day: str) -> Any:
    """Return events for a specific day from facet event logs."""
    if not DATE_RE.fullmatch(day):
        return "", 404

    from think.indexer.journal import get_events
    from think.utils import get_muse_configs

    generators = get_muse_configs(has_tools=False, has_output=True)

    # Get full event objects from source files
    raw_events = get_events(day)

    # Transform events into timeline format
    result = []
    for event in raw_events:
        topic = event.get("topic", "other")
        topic_color = generators.get(topic, {}).get("color", "#6c757d")

        formatted = {
            "title": event.get("title", ""),
            "summary": event.get("summary", ""),
            "subject": event.get("subject", ""),
            "details": event.get("details", event.get("description", "")),
            "participants": event.get("participants", []),
            "topic": topic,
            "color": topic_color,
            "facet": event.get("facet", ""),
            "occurred": event.get("occurred", True),
            "source": event.get("source", ""),
        }

        # Convert time strings to ISO timestamps
        if event.get("start"):
            formatted["startTime"] = f"{day[:4]}-{day[4:6]}-{day[6:]}T{event['start']}"
        if event.get("end"):
            formatted["endTime"] = f"{day[:4]}-{day[4:6]}-{day[6:]}T{event['end']}"

        result.append(formatted)

    return jsonify(result)


@calendar_bp.route("/api/stats/<month>")
def calendar_stats(month: str) -> Any:
    """Return event counts per facet for a specific month.

    Scans event files directly (including future dates) rather than relying
    on cached stats.json files which only exist for past days.

    Args:
        month: YYYYMM format month string

    Returns:
        JSON dict mapping day (YYYYMMDD) to facet counts dict.
        Frontend handles filtering by selected facet or summing for all-facet mode.
    """
    from think.events import get_month_event_counts

    # Validate month format (YYYYMM)
    if not re.fullmatch(r"\d{6}", month):
        return jsonify({"error": "Invalid month format, expected YYYYMM"}), 400

    stats = get_month_event_counts(month)
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
    if not DATE_RE.fullmatch(day):
        return "", 404

    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return "", 404

    title = format_date(day)

    return render_template(
        "app.html",
        view="_dev_screens_list",
        title=title,
    )


@calendar_bp.route("/<day>/screens/<timestamp>")
@calendar_bp.route("/<day>/screens/<timestamp>/<filename>")
def _dev_calendar_screens_detail(
    day: str, timestamp: str, filename: str = "screen.jsonl"
) -> str:
    """Render detail view for a specific screen.jsonl file."""
    if not DATE_RE.fullmatch(day):
        return "", 404
    from think.utils import segment_key

    if not segment_key(timestamp):
        return "", 404

    # Validate filename matches *screen.jsonl pattern
    if not filename.endswith("screen.jsonl"):
        return "", 404

    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return "", 404

    # Check if the screen.jsonl file exists in segment
    segment_dir = os.path.join(day_dir, timestamp)
    jsonl_path = os.path.join(segment_dir, filename)
    if not os.path.isfile(jsonl_path):
        return "", 404

    title = format_date(day)

    return render_template(
        "app.html",
        view="_dev_screens_detail",
        title=title,
        timestamp=timestamp,
        filename=filename,
    )


@calendar_bp.route("/api/screen_files/<day>")
def _dev_screen_files(day: str) -> Any:
    """Return list of *screen.jsonl files for a day."""
    if not DATE_RE.fullmatch(day):
        return "", 404

    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return jsonify({"files": []})

    from glob import glob

    from think.utils import segment_key

    files = []
    # Look for segments (HHMMSS_LEN/)
    for item in sorted(os.listdir(day_dir)):
        item_path = os.path.join(day_dir, item)
        if os.path.isdir(item_path) and segment_key(item):
            # Found segment, check for *screen.jsonl files
            screen_files = glob(os.path.join(item_path, "*screen.jsonl"))
            for jsonl_path in sorted(screen_files):
                filename = os.path.basename(jsonl_path)
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
                    time_obj = datetime.strptime(timestamp[:6], "%H%M%S")
                    human_time = time_obj.strftime("%I:%M:%S %p").lstrip("0")
                except Exception:
                    human_time = timestamp

                files.append(
                    {
                        "timestamp": timestamp,
                        "filename": filename,
                        "human_time": human_time,
                        "frame_count": frame_count,
                        "file_size": file_size,
                    }
                )

    return jsonify({"files": files})


@calendar_bp.route("/api/screen_frames/<day>/<timestamp>")
@calendar_bp.route("/api/screen_frames/<day>/<timestamp>/<filename>")
def _dev_screen_frames(day: str, timestamp: str, filename: str = "screen.jsonl") -> Any:
    """Return all frame records and pre-cache decoded frames from video."""
    if not DATE_RE.fullmatch(day):
        return "", 404
    from think.utils import segment_key

    if not segment_key(timestamp):
        return "", 404

    # Validate filename matches *screen.jsonl pattern
    if not filename.endswith("screen.jsonl"):
        return "", 404

    day_dir = str(day_path(day))
    segment_dir = os.path.join(day_dir, timestamp)
    jsonl_path = os.path.join(segment_dir, filename)

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
            raw_path = all_frames[0].get("raw")
            # Validate raw points to a video file (skip if not, e.g. tmux)
            if raw_path and raw_path.endswith(VIDEO_EXTENSIONS):
                raw_video_path = raw_path

        # Decode and cache all frames from the video
        cache_key = (day, timestamp, filename)
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

                        # Draw black rectangle if all 4 corner tags detected
                        # Tag IDs: 6=TL, 7=TR, 4=BL, 2=BR
                        # ArUco corner order: [TL, TR, BR, BL]
                        corner_tags = {6, 7, 4, 2}
                        if corner_tags.issubset(id_to_corners.keys()):
                            draw = ImageDraw.Draw(img)
                            # Extract outer corners from each tag
                            tl = id_to_corners[6].reshape(4, 2)[0]  # TL tag, TL corner
                            tr = id_to_corners[7].reshape(4, 2)[1]  # TR tag, TR corner
                            br = id_to_corners[2].reshape(4, 2)[2]  # BR tag, BR corner
                            bl = id_to_corners[4].reshape(4, 2)[3]  # BL tag, BL corner
                            poly = [tuple(tl), tuple(tr), tuple(br), tuple(bl)]
                            draw.polygon(poly, fill=(0, 0, 0))

                        jpeg_bytes = image_to_jpeg_bytes(img)
                        _frame_cache[cache_key][frame_id] = jpeg_bytes
                        img.close()

        return jsonify({"frames": frames, "raw_video_path": raw_video_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@calendar_bp.route("/api/screen_frame_image/<day>/<timestamp>/<int:frame_id>")
def _dev_screen_frame_image(day: str, timestamp: str, frame_id: int) -> Any:
    """Serve a cached frame image as JPEG."""
    if not DATE_RE.fullmatch(day):
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
