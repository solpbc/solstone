"""Transcript viewer app - browse and playback daily transcripts."""

from __future__ import annotations

import os
import re
from datetime import date, datetime
from glob import glob
from typing import Any

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from convey import state
from convey.utils import DATE_RE, format_date
from think.cluster import cluster_range, cluster_scan, cluster_segments
from think.utils import day_dirs, day_path

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


@transcripts_bp.route("/api/content/<day>")
def transcript_content(day: str) -> Any:
    """Return transcript markdown HTML for the selected range."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not re.fullmatch(r"\d{6}", start) or not re.fullmatch(r"\d{6}", end):
        return "", 400

    audio_enabled = request.args.get("audio", "true").lower() == "true"
    screen_enabled = request.args.get("screen", "true").lower() == "true"

    if not audio_enabled and not screen_enabled:
        markdown_text = "*Please select at least one source (Audio or Screen)*"
    elif audio_enabled and screen_enabled:
        markdown_text = cluster_range(
            day, start, end, audio=True, screen=True, insights=False
        )
    elif audio_enabled:
        markdown_text = cluster_range(
            day, start, end, audio=True, screen=False, insights=False
        )
    else:
        # Screen only - raw screencast transcripts
        markdown_text = cluster_range(
            day, start, end, audio=False, screen=True, insights=False
        )

    try:
        import markdown

        html_output = markdown.markdown(markdown_text, extensions=["extra", "nl2br"])
    except Exception:
        import html as html_mod

        html_output = f"<pre>{html_mod.escape(markdown_text)}</pre>"

    return jsonify({"html": html_output})


@transcripts_bp.route("/api/media_files/<day>")
def media_files(day: str) -> Any:
    """Return actual media files for embedding in the selected range."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    from think.cluster import get_entries_for_range
    from think.utils import get_raw_file

    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not re.fullmatch(r"\d{6}", start) or not re.fullmatch(r"\d{6}", end):
        return "", 400

    file_type = request.args.get("type", None)

    if file_type == "audio":
        entries = get_entries_for_range(day, start, end, audio=True, screen=False)
    elif file_type == "screen":
        entries = get_entries_for_range(day, start, end, audio=False, screen=True)
    else:
        entries = get_entries_for_range(day, start, end, audio=True, screen=True)

    media = []
    for e in entries:
        try:
            rel_path, mime_type, metadata = get_raw_file(day, e["name"])
            file_url = (
                f"/app/transcripts/api/serve_file/{day}/{rel_path.replace('/', '__')}"
            )
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

    start = request.args.get("start", "")
    end = request.args.get("end", "")
    if not re.fullmatch(r"\d{6}", start) or not re.fullmatch(r"\d{6}", end):
        return "", 400

    import subprocess
    import tempfile
    from pathlib import Path

    from flask import send_file

    from think.cluster import get_entries_for_range
    from think.utils import get_raw_file

    day_dir = str(day_path(day))
    if not os.path.isdir(day_dir):
        return jsonify({"error": "Day directory not found"}), 404

    entries = get_entries_for_range(day, start, end, audio=True, screen=False)

    audio_files = []
    for e in entries:
        if e.get("prefix") == "audio":
            try:
                rel_path, mime_type, metadata = get_raw_file(day, e["name"])
                flac_path = os.path.join(day_dir, rel_path)
                if os.path.isfile(flac_path):
                    audio_files.append(flac_path)
            except (ValueError, Exception):
                continue

    if not audio_files:
        return jsonify({"error": "No audio files found in the selected range"}), 404

    # Format times for filename and metadata
    start_dt = datetime.strptime(start, "%H%M%S")
    end_dt = datetime.strptime(end, "%H%M%S")
    filename = f"sunstone_{day}_{start[:4]}-{end[:4]}.mp3"

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


@transcripts_bp.route("/api/stats/<month>")
def api_stats(month: str):
    """Return transcript range counts for each day in a specific month.

    Args:
        month: YYYYMM format month string

    Returns:
        JSON dict mapping day (YYYYMMDD) to transcript range count.
        Transcripts app is not facet-aware, so returns simple {day: count} mapping.
    """
    if not re.fullmatch(r"\d{6}", month):
        return jsonify({"error": "Invalid month format, expected YYYYMM"}), 400

    from think.cluster import cluster_scan

    stats: dict[str, int] = {}

    for day_name in day_dirs().keys():
        # Filter to only days in requested month
        if not day_name.startswith(month):
            continue

        audio_ranges, screen_ranges = cluster_scan(day_name)
        # Count total ranges (audio + screen, but unique time slots)
        total_ranges = len(audio_ranges) + len(screen_ranges)
        if total_ranges > 0:
            stats[day_name] = total_ranges

    return jsonify(stats)


@transcripts_bp.route("/api/screen_frames/<day>/<segment_key>")
def screen_frames(day: str, segment_key: str) -> Any:
    """Load and cache all screen frames for a segment, return metadata.

    Flushes cache if a different segment is requested. Returns frame metadata
    for all monitors - frontend filters which frames to display.
    """
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    from think.utils import segment_key as validate_segment_key

    if not validate_segment_key(segment_key):
        return "", 404

    day_dir = str(day_path(day))
    segment_dir = os.path.join(day_dir, segment_key)
    if not os.path.isdir(segment_dir):
        return "", 404

    # Check if we already have this segment cached
    cache_key = (day, segment_key)
    if _screen_cache["segment"] == cache_key:
        return jsonify({"frames": _screen_cache["metadata"]})

    # Flush cache and load new segment
    _screen_cache["segment"] = cache_key
    _screen_cache["frames"] = {}
    _screen_cache["metadata"] = []

    # Find all *screen.jsonl files in segment
    screen_files = glob(os.path.join(segment_dir, "*screen.jsonl"))
    if not screen_files:
        return jsonify({"frames": []})

    from observe.see import decode_frames, image_to_jpeg_bytes
    from observe.utils import load_analysis_frames

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

    return jsonify({"frames": all_metadata})


@transcripts_bp.route("/api/screen_frame/<day>/<segment_key>/<filename>/<int:frame_id>")
def screen_frame(day: str, segment_key: str, filename: str, frame_id: int) -> Any:
    """Serve a cached frame image as JPEG."""
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    from think.utils import segment_key as validate_segment_key

    if not validate_segment_key(segment_key):
        return "", 404

    # Validate filename pattern
    if not filename.endswith("screen.jsonl"):
        return "", 404

    # Check cache matches requested segment
    cache_key = (day, segment_key)
    if _screen_cache["segment"] != cache_key:
        return jsonify({"error": "Segment not cached. Load frames list first."}), 404

    # Look up frame in cache
    frame_key = (filename, frame_id)
    if frame_key not in _screen_cache["frames"]:
        return jsonify({"error": "Frame not found in cache."}), 404

    import io

    from flask import send_file

    jpeg_bytes = _screen_cache["frames"][frame_key]
    buffer = io.BytesIO(jpeg_bytes)
    buffer.seek(0)
    return send_file(buffer, mimetype="image/jpeg")
