# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Transcript viewer app - browse and playback daily transcripts."""

from __future__ import annotations

import os
import re
import shutil
from datetime import date
from glob import glob
from pathlib import Path
from typing import Any

from flask import (
    Blueprint,
    jsonify,
    redirect,
    render_template,
    send_file,
    url_for,
)

from apps.utils import log_app_action
from convey import emit, state
from convey.utils import DATE_RE, error_response, format_date, success_response
from observe.hear import format_audio
from observe.screen import format_screen
from observe.utils import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from think.cluster import cluster_scan, cluster_segments
from think.models import get_usage_cost
from think.utils import day_dirs, day_path
from think.utils import segment_key as validate_segment_key

# Regex for HHMMSS time format validation
TIME_RE = re.compile(r"\d{6}")

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
    if not DATE_RE.fullmatch(day):
        return "", 404

    title = format_date(day)

    return render_template("app.html", title=title)


@transcripts_bp.route("/api/ranges/<day>")
def transcript_ranges(day: str) -> Any:
    """Return available transcript ranges for a day."""
    if not DATE_RE.fullmatch(day):
        return "", 404

    audio_ranges, screen_ranges = cluster_scan(day)
    return jsonify({"audio": audio_ranges, "screen": screen_ranges})


@transcripts_bp.route("/api/segments/<day>")
def transcript_segments(day: str) -> Any:
    """Return individual recording segments for a day.

    Returns list of segments with their content types for the segment selector UI.
    """
    if not DATE_RE.fullmatch(day):
        return "", 404

    segments = cluster_segments(day)
    return jsonify({"segments": segments})


@transcripts_bp.route("/api/serve_file/<day>/<path:encoded_path>")
def serve_file(day: str, encoded_path: str) -> Any:
    """Serve actual media files for embedding."""
    if not DATE_RE.fullmatch(day):
        return "", 404

    try:
        rel_path = encoded_path.replace("__", "/")
        full_path = os.path.join(state.journal_root, day, rel_path)

        day_dir = str(day_path(day))
        if not os.path.commonpath([full_path, day_dir]) == day_dir:
            return "", 403

        if not os.path.isfile(full_path):
            return "", 404

        return send_file(full_path)

    except Exception:
        return "", 404


@transcripts_bp.route("/api/stats/<month>")
def api_stats(month: str):
    """Return transcript range counts for each day in a specific month.

    Args:
        month: YYYYMM format month string

    Returns:
        JSON dict mapping day (YYYYMMDD) to transcript range count.
        Transcripts app is not facet-aware, so returns simple {day: count} mapping.
    """
    if not TIME_RE.fullmatch(month):
        return jsonify({"error": "Invalid month format, expected YYYYMM"}), 400

    stats: dict[str, int] = {}

    for day_name in day_dirs().keys():
        if not day_name.startswith(month):
            continue

        audio_ranges, screen_ranges = cluster_scan(day_name)
        total_ranges = len(audio_ranges) + len(screen_ranges)
        if total_ranges > 0:
            stats[day_name] = total_ranges

    return jsonify(stats)


def _load_jsonl(path: str) -> list[dict]:
    """Load JSONL file and return list of entries."""
    import json

    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _format_time_from_offset(segment_key: str, offset_sec: float) -> str:
    """Convert segment start + offset to HH:MM:SS format."""
    from think.utils import segment_parse

    start_time, _ = segment_parse(segment_key)
    if not start_time:
        return ""

    total_sec = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
    total_sec += int(offset_sec)

    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


@transcripts_bp.route("/api/segment/<day>/<segment_key>")
def segment_content(day: str, segment_key: str) -> Any:
    """Return unified timeline of audio and screen entries for a segment.

    Uses format_audio() and format_screen() to get chunks with source data,
    then merges chronologically for unified display.

    Returns JSON with:
        - chunks: List of entries sorted by timestamp, each with:
            - type: "audio" or "screen"
            - time: formatted wall-clock time (HH:MM:SS)
            - timestamp: unix ms for ordering
            - markdown: formatted content
            - source_ref: key fields from source for media lookup
        - audio_file: URL to segment audio file (if exists)
        - video_files: dict mapping jsonl filename to video URL for client-side decoding
        - segment_key: segment directory name
        - cost: processing cost in USD (float, 0.0 if no data)
        - media_sizes: dict with audio/screen byte counts for raw media files
    """
    if not DATE_RE.fullmatch(day):
        return "", 404

    if not validate_segment_key(segment_key):
        return "", 404

    day_dir = str(day_path(day))
    segment_dir = os.path.join(day_dir, segment_key)
    if not os.path.isdir(segment_dir):
        return "", 404

    chunks: list[dict] = []
    audio_file_url = None
    video_files: dict[str, str] = {}  # jsonl filename -> video URL
    media_sizes: dict[str, int] = {"audio": 0, "screen": 0}

    # Process audio files
    audio_files = glob(os.path.join(segment_dir, "*audio.jsonl"))
    for audio_path in sorted(audio_files):
        try:
            entries = _load_jsonl(audio_path)
            formatted_chunks, meta = format_audio(entries, {"file_path": audio_path})

            # Find the raw audio file from metadata (first entry without "start")
            raw_audio = None
            for entry in entries:
                if "start" not in entry and "raw" in entry:
                    raw_audio = entry["raw"]
                    break

            # Validate raw points to an audio file (skip if not)
            if raw_audio and raw_audio.endswith(AUDIO_EXTENSIONS):
                rel_path = f"{segment_key}/{raw_audio}"
                audio_file_url = f"/app/transcripts/api/serve_file/{day}/{rel_path.replace('/', '__')}"
                audio_full = os.path.join(segment_dir, raw_audio)
                if os.path.isfile(audio_full):
                    media_sizes["audio"] += os.path.getsize(audio_full)

            for chunk in formatted_chunks:
                source = chunk.get("source", {})
                # Audio has start time in HH:MM:SS format
                time_str = source.get("start", "")
                chunks.append(
                    {
                        "type": "audio",
                        "time": time_str,
                        "timestamp": chunk.get("timestamp", 0),
                        "markdown": chunk.get("markdown", ""),
                        "source_ref": {
                            "start": time_str,
                            "source": source.get("source"),
                            "speaker": source.get("speaker"),
                        },
                    }
                )
        except Exception:
            continue

    # Process screen files and collect video URLs for client-side decoding
    screen_files = glob(os.path.join(segment_dir, "*screen.jsonl"))
    for screen_path in sorted(screen_files):
        try:
            entries = _load_jsonl(screen_path)
            formatted_chunks, meta = format_screen(entries, {"file_path": screen_path})

            filename = os.path.basename(screen_path)
            monitor = (
                filename.replace("_screen.jsonl", "")
                if filename != "screen.jsonl"
                else ""
            )

            # Extract video URL from header (first entry without frame_id)
            raw_video = None
            for entry in entries:
                if "frame_id" not in entry and "raw" in entry:
                    raw_video = entry["raw"]
                    break

            # Validate raw points to a video file (skip if not, e.g. tmux)
            if raw_video and raw_video.endswith(VIDEO_EXTENSIONS):
                video_full = os.path.join(segment_dir, raw_video)
                if os.path.isfile(video_full):
                    rel_path = f"{segment_key}/{raw_video}"
                    video_files[filename] = (
                        f"/app/transcripts/api/serve_file/{day}/{rel_path.replace('/', '__')}"
                    )
                    media_sizes["screen"] += os.path.getsize(video_full)

            for chunk in formatted_chunks:
                source = chunk.get("source", {})
                frame_id = source.get("frame_id")
                offset = source.get("timestamp", 0)

                # Calculate wall-clock time from segment start + offset
                time_str = _format_time_from_offset(segment_key, offset)

                # Basic frames have no enriched content
                frame_content = source.get("content", {})
                is_basic = not frame_content

                # Extract participant boxes for meeting frames
                participants = []
                meeting_data = frame_content.get("meeting")
                if meeting_data:
                    for p in meeting_data.get("participants", []):
                        box = p.get("box_2d")
                        # Only include participants with video and valid box_2d
                        if p.get("video") and box and len(box) == 4:
                            y_min, x_min, y_max, x_max = box
                            participants.append(
                                {
                                    "name": p.get("name", "Unknown"),
                                    "status": p.get("status", "unknown"),
                                    "top": y_min / 10,
                                    "left": x_min / 10,
                                    "height": (y_max - y_min) / 10,
                                    "width": (x_max - x_min) / 10,
                                }
                            )

                # Include box_2d for client-side bounding box drawing
                box_2d = source.get("box_2d")

                chunks.append(
                    {
                        "type": "screen",
                        "time": time_str,
                        "timestamp": chunk.get("timestamp", 0),
                        "markdown": chunk.get("markdown", ""),
                        "source_ref": {
                            "frame_id": frame_id,
                            "filename": filename,
                            "monitor": monitor,
                            "offset": offset,
                            "box_2d": box_2d,
                            "analysis": source.get("analysis"),
                            "participants": participants if participants else None,
                            "aruco": source.get("aruco"),
                        },
                        "basic": is_basic,
                    }
                )
        except Exception:
            continue

    # Sort all chunks by timestamp
    chunks.sort(key=lambda c: c["timestamp"])

    # Get cost data for this segment
    cost_data = get_usage_cost(day, segment=segment_key)

    # Collect agent .md files
    md_files = {}
    for md_path in sorted(Path(segment_dir).glob("*.md")):
        try:
            md_files[md_path.stem] = md_path.read_text()
        except Exception:
            continue

    return jsonify(
        {
            "chunks": chunks,
            "audio_file": audio_file_url,
            "video_files": video_files,
            "md_files": md_files,
            "segment_key": segment_key,
            "cost": cost_data["cost"],
            "media_sizes": media_sizes,
        }
    )


@transcripts_bp.route("/api/segment/<day>/<segment_key>", methods=["DELETE"])
def delete_segment(day: str, segment_key: str) -> Any:
    """Delete a segment directory and all its contents.

    This permanently removes all audio files, screen recordings, transcripts,
    and insights for the specified segment. This action cannot be undone.

    Args:
        day: Day in YYYYMMDD format
        segment_key: Segment directory name (HHMMSS_LEN format)

    Returns:
        JSON success response or error response
    """
    if not DATE_RE.fullmatch(day):
        return error_response("Invalid day format", 400)

    if not validate_segment_key(segment_key):
        return error_response("Invalid segment key format", 400)

    day_dir = str(day_path(day))
    segment_dir = os.path.join(day_dir, segment_key)

    # Verify segment exists
    if not os.path.isdir(segment_dir):
        return error_response("Segment not found", 404)

    # Security check: ensure segment_dir is within day_dir
    if not os.path.commonpath([segment_dir, day_dir]) == day_dir:
        return error_response("Invalid segment path", 403)

    try:
        # Remove the entire segment directory
        shutil.rmtree(segment_dir)

        # Log the deletion for audit trail
        log_app_action(
            app="transcripts",
            facet=None,  # Transcripts are not facet-scoped
            action="segment_delete",
            params={"day": day, "segment_key": segment_key},
            day=day,
        )

        # Trigger indexer rescan to remove deleted segment from search index
        # Supervisor queues by command name, serializing concurrent indexer requests
        emit(
            "supervisor",
            "request",
            cmd=["sol", "indexer", "--rescan-full"],
        )

        return success_response({"deleted": segment_key})

    except OSError as e:
        return error_response(f"Failed to delete segment: {e}", 500)
