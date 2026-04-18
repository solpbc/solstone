# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import calendar
import os
from datetime import date, datetime
from typing import Any

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from convey.utils import DATE_RE, format_date
from observe.utils import VIDEO_EXTENSIONS
from think.utils import day_path, iter_segments, segment_parse
from think.utils import segment_path as get_segment_path

activities_bp = Blueprint(
    "app:activities",
    __name__,
    url_prefix="/app/activities",
)


@activities_bp.route("/")
def index():
    """Redirect to today's activities view."""
    today = date.today().strftime("%Y%m%d")
    return redirect(url_for("app:activities.activities_day", day=today))


@activities_bp.route("/<day>")
def activities_day(day: str) -> str:
    """Render the day view for a specific day."""
    if not DATE_RE.fullmatch(day):
        return "", 404

    title = format_date(day)

    return render_template(
        "app.html",
        view="day",
        title=title,
    )


def _month_activity_counts(month: str) -> dict[str, dict[str, int]]:
    from think.activities import load_activity_records
    from think.facets import get_facets

    year = int(month[:4])
    month_num = int(month[4:6])
    _, days_in_month = calendar.monthrange(year, month_num)
    facet_names = list(get_facets().keys())
    stats: dict[str, dict[str, int]] = {}

    for day_num in range(1, days_in_month + 1):
        day = f"{month}{day_num:02d}"
        day_counts: dict[str, int] = {}
        for facet in facet_names:
            count = len(load_activity_records(facet, day))
            if count:
                day_counts[facet] = count
        if day_counts:
            stats[day] = day_counts

    return stats


@activities_bp.route("/api/stats/<month>")
def activities_stats(month: str) -> Any:
    """Return activity counts per facet for a specific month."""
    if len(month) != 6 or not month.isdigit():
        return jsonify({"error": "Invalid month format, expected YYYYMM"}), 400

    try:
        return jsonify(_month_activity_counts(month))
    except ValueError:
        return jsonify({"error": "Invalid month format, expected YYYYMM"}), 400


@activities_bp.route("/api/day/<day>/activities")
def activities_day_activities(day: str) -> Any:
    """Return enriched activity records for a specific day.

    Loads activity records from all facets (or a single facet if ``facet``
    query param is set), enriches each with activity metadata (name, icon),
    computed timing from segment keys, and lists any output files.

    Returns JSON array of activity objects.
    """
    if not DATE_RE.fullmatch(day):
        return jsonify({"error": "Invalid day format"}), 400

    from pathlib import Path

    from convey import state
    from think.activities import (
        estimate_duration_minutes,
        get_activity_by_id,
        load_activity_records,
    )
    from think.facets import get_facets

    journal_root = state.journal_root
    facet_filter = request.args.get("facet")

    if facet_filter:
        facet_names = [facet_filter]
    else:
        facet_names = list(get_facets().keys())

    result = []
    for facet in facet_names:
        records = load_activity_records(facet, day)
        for record in records:
            activity_type = record.get("activity", "")
            activity_def = get_activity_by_id(facet, activity_type)

            # Derive start/end times from first/last segment keys
            segments = record.get("segments", [])
            start_time = end_time = None
            if segments:
                first_start, _ = segment_parse(segments[0])
                _, last_end = segment_parse(segments[-1])
                if first_start:
                    start_time = (
                        f"{day[:4]}-{day[4:6]}-{day[6:]}T"
                        f"{first_start.strftime('%H:%M:%S')}"
                    )
                if last_end:
                    end_time = (
                        f"{day[:4]}-{day[4:6]}-{day[6:]}T"
                        f"{last_end.strftime('%H:%M:%S')}"
                    )

            # Scan for output files
            outputs = []
            output_dir = (
                Path(journal_root)
                / "facets"
                / facet
                / "activities"
                / day
                / record["id"]
            )
            if output_dir.is_dir():
                for f in sorted(output_dir.iterdir()):
                    if f.is_file():
                        rel = f.relative_to(journal_root)
                        outputs.append({"filename": f.name, "path": str(rel)})

            enriched = {
                "id": record["id"],
                "activity": activity_type,
                "name": (
                    activity_def.get("name", activity_type)
                    if activity_def
                    else activity_type
                ),
                "icon": activity_def.get("icon", "") if activity_def else "",
                "facet": facet,
                "description": record.get("description", ""),
                "level_avg": record.get("level_avg", 0.5),
                "duration_minutes": estimate_duration_minutes(segments),
                "segments": segments,
                "active_entities": record.get("active_entities", []),
                "outputs": outputs,
            }
            if start_time:
                enriched["startTime"] = start_time
            if end_time:
                enriched["endTime"] = end_time

            result.append(enriched)

    # Sort by start time (activities without times go last)
    result.sort(key=lambda a: a.get("startTime", "z"))
    return jsonify(result)


@activities_bp.route("/api/activity_output/<path:filename>")
def activities_activity_output(filename: str) -> Any:
    """Serve an activity output file.

    Only serves files under ``facets/`` in the journal directory.
    Returns JSON with content, format, and filename.
    """
    from pathlib import Path

    from convey import state

    if not filename.startswith("facets/"):
        return jsonify(error="Invalid path"), 400

    journal_root = Path(state.journal_root).resolve()
    file_path = (journal_root / filename).resolve()

    try:
        file_path.relative_to(journal_root)
    except ValueError:
        return jsonify(error="Invalid path"), 403

    if not file_path.is_file():
        return jsonify(error="File not found"), 404

    ext = file_path.suffix.lower()
    fmt = "json" if ext == ".json" else "md"

    try:
        content = file_path.read_text(encoding="utf-8")
    except IOError:
        return jsonify(error="Could not read file"), 500

    return jsonify(content=content, format=fmt, filename=file_path.name)


# ============================================================================
# DEVELOPER DEBUG VIEWS - Screen JSONL Viewer
# These routes are for debugging screen.jsonl files and can be removed when
# no longer needed. Look for _dev_ prefix to identify all related code.
# ============================================================================

# In-memory cache for decoded frames: {(day, timestamp): {frame_id: jpeg_bytes}}
_frame_cache: dict = {}


@activities_bp.route("/<day>/screens")
def _dev_activities_screens_list(day: str) -> str:
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


@activities_bp.route("/<day>/screens/<stream>/<timestamp>")
@activities_bp.route("/<day>/screens/<stream>/<timestamp>/<filename>")
def _dev_activities_screens_detail(
    day: str, stream: str, timestamp: str, filename: str = "screen.jsonl"
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
    segment_dir = str(get_segment_path(day, timestamp, stream))
    jsonl_path = os.path.join(segment_dir, filename)
    if not os.path.isfile(jsonl_path):
        return "", 404

    title = format_date(day)

    return render_template(
        "app.html",
        view="_dev_screens_detail",
        title=title,
        stream=stream,
        timestamp=timestamp,
        filename=filename,
    )


@activities_bp.route("/api/screen_files/<day>")
def _dev_screen_files(day: str) -> Any:
    """Return list of *screen.jsonl files for a day."""
    if not DATE_RE.fullmatch(day):
        return "", 404

    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return jsonify({"files": []})

    from glob import glob

    files = []
    # Look for segments across all streams
    for s_stream, s_key, s_path in iter_segments(day):
        # Found segment, check for *screen.jsonl files
        screen_files = glob(os.path.join(str(s_path), "*screen.jsonl"))
        for jsonl_path in sorted(screen_files):
            filename = os.path.basename(jsonl_path)
            timestamp = s_key

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
            try:
                time_obj = datetime.strptime(timestamp[:6], "%H%M%S")
                human_time = time_obj.strftime("%I:%M:%S %p").lstrip("0")
            except Exception:
                human_time = timestamp

            files.append(
                {
                    "stream": s_stream,
                    "timestamp": timestamp,
                    "filename": filename,
                    "human_time": human_time,
                    "frame_count": frame_count,
                    "file_size": file_size,
                }
            )

    return jsonify({"files": files})


@activities_bp.route("/api/screen_frames/<day>/<stream>/<timestamp>")
@activities_bp.route("/api/screen_frames/<day>/<stream>/<timestamp>/<filename>")
def _dev_screen_frames(
    day: str, stream: str, timestamp: str, filename: str = "screen.jsonl"
) -> Any:
    """Return all frame records and pre-cache decoded frames from video."""
    if not DATE_RE.fullmatch(day):
        return "", 404
    from think.utils import segment_key

    if not segment_key(timestamp):
        return "", 404

    # Validate filename matches *screen.jsonl pattern
    if not filename.endswith("screen.jsonl"):
        return "", 404

    segment_dir = str(get_segment_path(day, timestamp, stream))
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
        cache_key = (day, stream, timestamp, filename)
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


@activities_bp.route(
    "/api/screen_frame_image/<day>/<stream>/<timestamp>/<int:frame_id>"
)
def _dev_screen_frame_image(
    day: str, stream: str, timestamp: str, frame_id: int
) -> Any:
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
        cache_key = (day, stream, timestamp)
        if cache_key in _frame_cache and frame_id in _frame_cache[cache_key]:
            jpeg_bytes = _frame_cache[cache_key][frame_id]
            buffer = io.BytesIO(jpeg_bytes)
            buffer.seek(0)
            return send_file(buffer, mimetype="image/jpeg")

        # Frame not in cache
        return jsonify({"error": "Frame not cached. Load the frames list first."}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500
