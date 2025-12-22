"""Transcript viewer app - browse and playback daily transcripts."""

from __future__ import annotations

import io
import os
import re
from datetime import date
from glob import glob
from typing import Any

from flask import (
    Blueprint,
    jsonify,
    redirect,
    render_template,
    send_file,
    url_for,
)

from convey import state
from convey.utils import DATE_RE, format_date
from observe.hear import format_audio
from observe.screen import format_screen
from observe.see import decode_frames, image_to_jpeg_bytes
from observe.utils import load_analysis_frames
from think.cluster import cluster_scan, cluster_segments
from think.utils import day_dirs, day_path
from think.utils import segment_key as validate_segment_key

# Regex for HHMMSS time format validation
TIME_RE = re.compile(r"\d{6}")

# Single-segment screenshot cache (flushed when different segment requested)
_screen_cache: dict = {
    "segment": None,  # (day, segment_key) tuple or None
    "frames": {},  # {(filename, frame_id): jpeg_bytes}
    "metadata": [],  # List of frame records with monitor info
}

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

    return render_template("app.html", title=title)


@transcripts_bp.route("/api/ranges/<day>")
def transcript_ranges(day: str) -> Any:
    """Return available transcript ranges for a day."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    audio_ranges, screen_ranges = cluster_scan(day)
    return jsonify({"audio": audio_ranges, "screen": screen_ranges})


@transcripts_bp.route("/api/segments/<day>")
def transcript_segments(day: str) -> Any:
    """Return individual recording segments for a day.

    Returns list of segments with their content types for the segment selector UI.
    """
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    segments = cluster_segments(day)
    return jsonify({"segments": segments})


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


def _populate_screen_cache(day: str, segment_key: str) -> list[dict]:
    """Load and cache all screen frames for a segment.

    Populates the global _screen_cache with decoded JPEG frames and returns
    frame metadata. If already cached for this segment, returns cached metadata.

    Args:
        day: Day in YYYYMMDD format
        segment_key: Segment directory name (HHMMSS_LEN format)

    Returns:
        List of frame metadata dicts with keys: frame_id, filename, monitor,
        timestamp, box_2d, requests, analysis
    """
    day_dir = str(day_path(day))
    segment_dir = os.path.join(day_dir, segment_key)

    # Check if we already have this segment cached
    cache_key = (day, segment_key)
    if _screen_cache["segment"] == cache_key:
        return _screen_cache["metadata"]

    # Flush cache and load new segment
    _screen_cache["segment"] = cache_key
    _screen_cache["frames"] = {}
    _screen_cache["metadata"] = []

    # Find all *screen.jsonl files in segment
    screen_files = glob(os.path.join(segment_dir, "*screen.jsonl"))
    if not screen_files:
        return []

    all_metadata = []

    for jsonl_path in sorted(screen_files):
        filename = os.path.basename(jsonl_path)
        # Extract monitor name from filename (e.g., "center_DP-3_screen.jsonl" -> "center_DP-3")
        monitor = (
            filename.replace("_screen.jsonl", "") if filename != "screen.jsonl" else ""
        )

        try:
            all_frames = load_analysis_frames(jsonl_path)

            # First line is header with {"raw": "path"}, frames have frame_id
            frames = [f for f in all_frames if "frame_id" in f]
            if not frames:
                continue

            # Get video path from header
            raw_video_path = None
            if (
                all_frames
                and "raw" in all_frames[0]
                and "frame_id" not in all_frames[0]
            ):
                raw_video_path = all_frames[0].get("raw")

            if not raw_video_path:
                continue

            video_path = os.path.join(segment_dir, raw_video_path)
            if not os.path.isfile(video_path):
                continue

            # Decode all frames from video
            images = decode_frames(video_path, frames, annotate_boxes=True)

            # Cache JPEG bytes and build metadata
            for frame, img in zip(frames, images):
                if img is None:
                    continue

                frame_id = frame["frame_id"]
                jpeg_bytes = image_to_jpeg_bytes(img)
                _screen_cache["frames"][(filename, frame_id)] = jpeg_bytes
                img.close()

                # Build metadata for frontend
                meta = {
                    "frame_id": frame_id,
                    "filename": filename,
                    "monitor": monitor,
                    "timestamp": frame.get("timestamp", 0),
                    "box_2d": frame.get("box_2d"),
                    "requests": frame.get("requests", []),
                    "analysis": frame.get("analysis"),
                }
                all_metadata.append(meta)

        except Exception:
            # Skip files that fail to load
            continue

    # Sort by timestamp
    all_metadata.sort(key=lambda f: f["timestamp"])
    _screen_cache["metadata"] = all_metadata

    return all_metadata


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
        - segment_key: segment directory name
    """
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    if not validate_segment_key(segment_key):
        return "", 404

    day_dir = str(day_path(day))
    segment_dir = os.path.join(day_dir, segment_key)
    if not os.path.isdir(segment_dir):
        return "", 404

    chunks: list[dict] = []
    audio_file_url = None

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

            if raw_audio:
                rel_path = f"{segment_key}/{raw_audio}"
                audio_file_url = f"/app/transcripts/api/serve_file/{day}/{rel_path.replace('/', '__')}"

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

    # Process screen files - also populate cache for thumbnails
    _populate_screen_cache(day, segment_key)

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

            for chunk in formatted_chunks:
                source = chunk.get("source", {})
                frame_id = source.get("frame_id")
                offset = source.get("timestamp", 0)

                # Calculate wall-clock time from segment start + offset
                time_str = _format_time_from_offset(segment_key, offset)

                # Build thumbnail URL if frame_id exists and is cached
                thumb_url = None
                if frame_id is not None and (filename, frame_id) in _screen_cache.get(
                    "frames", {}
                ):
                    thumb_url = f"/app/transcripts/api/screen_frame/{day}/{segment_key}/{filename}/{frame_id}"

                # Basic frames have <= 1 vision request (just DESCRIBE_JSON)
                # Enhanced frames have > 1 (added text extraction or meeting analysis)
                requests = source.get("requests", [])
                is_basic = len(requests) <= 1

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
                            "analysis": source.get("analysis"),
                        },
                        "thumb_url": thumb_url,
                        "basic": is_basic,
                    }
                )
        except Exception:
            continue

    # Sort all chunks by timestamp
    chunks.sort(key=lambda c: c["timestamp"])

    return jsonify(
        {
            "chunks": chunks,
            "audio_file": audio_file_url,
            "segment_key": segment_key,
        }
    )


@transcripts_bp.route("/api/screen_frame/<day>/<segment_key>/<filename>/<int:frame_id>")
def screen_frame(day: str, segment_key: str, filename: str, frame_id: int) -> Any:
    """Serve a cached frame image as JPEG."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    if not validate_segment_key(segment_key):
        return "", 404

    if not filename.endswith("screen.jsonl"):
        return "", 404

    # Check cache matches requested segment
    cache_key = (day, segment_key)
    if _screen_cache["segment"] != cache_key:
        return jsonify({"error": "Segment not cached. Load frames list first."}), 404

    frame_key = (filename, frame_id)
    if frame_key not in _screen_cache["frames"]:
        return jsonify({"error": "Frame not found in cache."}), 404

    jpeg_bytes = _screen_cache["frames"][frame_key]
    buffer = io.BytesIO(jpeg_bytes)
    buffer.seek(0)
    return send_file(buffer, mimetype="image/jpeg")
