# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Transcript viewer app - browse and playback daily transcripts."""

from __future__ import annotations

import functools
import json
import logging
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
from think.cluster import cluster_scan, cluster_segments, scan_day
from think.entities.journal import get_journal_principal, load_journal_entity
from think.models import get_usage_cost
from think.supervisor import is_supervisor_up
from think.utils import STREAM_RE, day_dirs, day_path, segment_path
from think.utils import segment_key as validate_segment_key

logger = logging.getLogger(__name__)

# Regex for YYYYMM month format validation
MONTH_RE = re.compile(r"\d{6}")

transcripts_bp = Blueprint(
    "app:transcripts",
    __name__,
    url_prefix="/app/transcripts",
)


def _day_max_mtime(path: str) -> float:
    """Return the latest mtime under a day directory, skipping delete races."""
    day_dir = Path(path)
    try:
        max_mtime = day_dir.stat().st_mtime
    except FileNotFoundError:
        return 0.0

    try:
        for child in day_dir.rglob("*"):
            try:
                child_mtime = child.stat().st_mtime
            except FileNotFoundError:
                continue
            if child_mtime > max_mtime:
                max_mtime = child_mtime
    except FileNotFoundError:
        return max_mtime
    return max_mtime


@functools.lru_cache(maxsize=64)
def _stats_for_month(month: str, mtime_key: float) -> dict[str, int]:
    """Return cached transcript range counts for a month."""
    del mtime_key

    stats: dict[str, int] = {}
    for day_name in day_dirs().keys():
        if not day_name.startswith(month):
            continue

        audio_ranges, screen_ranges = cluster_scan(day_name)
        total_ranges = len(audio_ranges) + len(screen_ranges)
        if total_ranges > 0:
            stats[day_name] = total_ranges

    return stats


@transcripts_bp.route("/")
def index() -> Any:
    """Redirect to the most recent day with segments, falling back to today."""
    today = date.today().strftime("%Y%m%d")
    for day in sorted(day_dirs().keys(), reverse=True):
        if cluster_segments(day):
            return redirect(url_for("app:transcripts.transcripts_day", day=day))
    return redirect(url_for("app:transcripts.transcripts_day", day=today))


@transcripts_bp.route("/<day>")
def transcripts_day(day: str) -> str:
    """Render transcript viewer for a specific day."""
    if not DATE_RE.fullmatch(day):
        return error_response("Day not found", 404)

    title = format_date(day)

    return render_template("app.html", title=title)


@transcripts_bp.route("/api/ranges/<day>")
def transcript_ranges(day: str) -> Any:
    """Return available transcript ranges for a day."""
    if not DATE_RE.fullmatch(day):
        return error_response("Day not found", 404)

    audio_ranges, screen_ranges = cluster_scan(day)
    return jsonify({"audio": audio_ranges, "screen": screen_ranges})


@transcripts_bp.route("/api/segments/<day>")
def transcript_segments(day: str) -> Any:
    """Return individual recording segments for a day.

    Returns list of segments with their content types for the segment selector UI.
    """
    if not DATE_RE.fullmatch(day):
        return error_response("Day not found", 404)

    segments = cluster_segments(day)
    return jsonify({"segments": segments})


@transcripts_bp.route("/api/day/<day>")
def transcript_day_data(day: str) -> Any:
    """Return combined ranges and segments for a day in a single response."""
    if not DATE_RE.fullmatch(day):
        return error_response("Day not found", 404)

    audio_ranges, screen_ranges, segments = scan_day(day)
    return jsonify(
        {"audio": audio_ranges, "screen": screen_ranges, "segments": segments}
    )


@transcripts_bp.route("/api/serve_file/<day>/<path:rel_path>")
def serve_file(day: str, rel_path: str) -> Any:
    """Serve actual media files for embedding."""
    if not DATE_RE.fullmatch(day):
        return error_response("Day not found", 404)

    try:
        full_path = os.path.join(state.journal_root, day, rel_path)
        day_dir = str(day_path(day, create=False))
        if not os.path.commonpath([full_path, day_dir]) == day_dir:
            return error_response("Invalid file path", 403)
        if not os.path.isfile(full_path):
            return error_response("File not found", 404)
    except (OSError, ValueError):
        logger.warning(
            "serve_file path validation failed for %s/%s",
            day,
            rel_path,
            exc_info=True,
        )
        return error_response("Failed to serve file", 404)

    return send_file(full_path, conditional=True)


@transcripts_bp.route("/api/stats/<month>")
def api_stats(month: str):
    """Return transcript range counts for each day in a specific month.

    Args:
        month: YYYYMM format month string

    Returns:
        JSON dict mapping day (YYYYMMDD) to transcript range count.
        Transcripts app is not facet-aware, so returns simple {day: count} mapping.
    """
    if not MONTH_RE.fullmatch(month):
        return error_response("Invalid month format", 400)

    matching = [
        (day_name, path)
        for day_name, path in day_dirs().items()
        if day_name.startswith(month)
    ]
    if not matching:
        return jsonify({})

    mtime_key = max(_day_max_mtime(path) for _, path in matching)
    return jsonify(_stats_for_month(month, mtime_key))


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


@transcripts_bp.route("/api/segment/<day>/<stream>/<segment_key>")
def segment_content(day: str, stream: str, segment_key: str) -> Any:
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
        return error_response("Invalid day format", 404)

    if not STREAM_RE.fullmatch(stream):
        return error_response("Invalid stream format", 404)

    if not validate_segment_key(segment_key):
        return error_response("Invalid segment key format", 404)

    segment_dir = str(segment_path(day, segment_key, stream, create=False))
    if not os.path.isdir(segment_dir):
        return error_response("Segment directory not found", 404)

    chunks: list[dict] = []
    audio_file_url = None
    video_files: dict[str, str] = {}  # jsonl filename -> video URL
    media_sizes: dict[str, int] = {"audio": 0, "screen": 0}
    has_raw_reference = False
    has_raw_file = False
    warnings = 0

    # Load speaker labels if available.
    speaker_labels_path = Path(segment_dir) / "talents" / "speaker_labels.json"
    speaker_map: dict[int, dict] = {}
    if speaker_labels_path.is_file():
        try:
            with open(speaker_labels_path) as f:
                labels_data = json.load(f)
            principal = get_journal_principal()
            principal_id = principal["id"] if principal else None
            entity_cache: dict[str, dict | None] = {}
            for label in labels_data.get("labels", []):
                sid = label.get("sentence_id")
                entity_id = label.get("speaker")
                confidence = label.get("confidence")
                if sid is None or not entity_id or not confidence:
                    continue
                if entity_id not in entity_cache:
                    entity_cache[entity_id] = load_journal_entity(entity_id)
                entity = entity_cache[entity_id]
                name = entity["name"] if entity else entity_id
                is_owner = entity_id == principal_id
                speaker_map[sid] = {
                    "name": name,
                    "entity_id": entity_id,
                    "confidence": confidence,
                    "is_owner": is_owner,
                }
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    # Process audio files
    audio_files = glob(os.path.join(segment_dir, "*audio.jsonl"))
    for audio_path in sorted(audio_files):
        try:
            entries = _load_jsonl(audio_path)
            formatted_chunks, meta = format_audio(entries, {"file_path": audio_path})

            # Build sentence_id mapping (1-based over transcript entries only).
            entry_to_sid: dict[int, int] = {}
            sid = 0
            for entry in entries:
                if "start" in entry:
                    sid += 1
                    entry_to_sid[id(entry)] = sid

            # Find the raw audio file from metadata (first entry without "start")
            raw_audio = None
            for entry in entries:
                if "start" not in entry and "raw" in entry:
                    raw_audio = entry["raw"]
                    break

            # Validate raw points to an audio file (skip if not)
            if raw_audio and raw_audio.endswith(AUDIO_EXTENSIONS):
                has_raw_reference = True
                audio_full = os.path.join(segment_dir, raw_audio)
                if os.path.isfile(audio_full):
                    has_raw_file = True
                    rel_path = f"{stream}/{segment_key}/{raw_audio}"
                    audio_file_url = f"/app/transcripts/api/serve_file/{day}/{rel_path}"
                    media_sizes["audio"] += os.path.getsize(audio_full)

            for chunk in formatted_chunks:
                source = chunk.get("source", {})
                # Audio has start time in HH:MM:SS format
                time_str = source.get("start", "")
                markdown = chunk.get("markdown", "")

                chunk_sid = entry_to_sid.get(id(source))
                speaker_label = speaker_map.get(chunk_sid) if chunk_sid else None
                if speaker_label:
                    markdown = re.sub(r"Speaker \d+:\s*", "", markdown)

                chunk_data: dict[str, Any] = {
                    "type": "audio",
                    "time": time_str,
                    "timestamp": chunk.get("timestamp", 0),
                    "markdown": markdown,
                    "source_ref": {
                        "start": time_str,
                        "source": source.get("source"),
                        "speaker": source.get("speaker"),
                    },
                }
                if speaker_label:
                    chunk_data["speaker_label"] = speaker_label
                chunks.append(chunk_data)
        except Exception:
            logger.warning(
                "Failed to parse audio segment %s", audio_path, exc_info=True
            )
            warnings += 1
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
                has_raw_reference = True
                video_full = os.path.join(segment_dir, raw_video)
                if os.path.isfile(video_full):
                    has_raw_file = True
                    rel_path = f"{stream}/{segment_key}/{raw_video}"
                    video_files[filename] = (
                        f"/app/transcripts/api/serve_file/{day}/{rel_path}"
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
            logger.warning(
                "Failed to parse screen segment %s", screen_path, exc_info=True
            )
            warnings += 1
            continue

    # Sort all chunks by timestamp
    chunks.sort(key=lambda c: c["timestamp"])
    media_purged = has_raw_reference and not has_raw_file

    # Get cost data for this segment
    cost_data = get_usage_cost(day, segment=segment_key)

    # Collect talent .md files
    md_files = {}
    talents_dir = Path(segment_dir) / "talents"
    if talents_dir.is_dir():
        for md_path in sorted(talents_dir.rglob("*.md")):
            try:
                key = md_path.relative_to(talents_dir).with_suffix("").as_posix()
                md_files[key] = md_path.read_text()
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
            "media_purged": media_purged,
            "warnings": warnings,
        }
    )


@transcripts_bp.route("/api/segment/<day>/<stream>/<segment_key>", methods=["DELETE"])
def delete_segment(day: str, stream: str, segment_key: str) -> Any:
    """Delete a segment directory and all its contents.

    This permanently removes all audio files, screen recordings, transcripts,
    and insights for the specified segment. This action cannot be undone.

    Args:
        day: Day in YYYYMMDD format
        stream: Stream name
        segment_key: Segment directory name (HHMMSS_LEN format)

    Returns:
        JSON success response or error response
    """
    if not DATE_RE.fullmatch(day):
        return error_response("Invalid day format", 400)

    if not validate_segment_key(segment_key):
        return error_response("Invalid segment key format", 400)

    if not STREAM_RE.fullmatch(stream):
        return error_response("Invalid stream format", 400)

    day_dir = str(day_path(day, create=False))
    segment_dir = str(segment_path(day, segment_key, stream, create=False))

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

        payload = {"deleted": segment_key}
        if not is_supervisor_up():
            payload["search_index_warning"] = True

        # Trigger indexer rescan to remove deleted segment from search index
        # Supervisor queues by command name, serializing concurrent indexer requests
        emit(
            "supervisor",
            "request",
            cmd=["sol", "indexer", "--rescan-full"],
        )

        return success_response(payload)

    except OSError as e:
        return error_response(f"Failed to delete segment: {e}", 500)
