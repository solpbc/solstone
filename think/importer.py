# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import argparse
import datetime as dt
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from datetime import timedelta
from pathlib import Path

from observe.utils import find_available_segment
from think.callosum import CallosumConnection
from think.detect_created import detect_created
from think.detect_transcript import detect_transcript_json, detect_transcript_segment
from think.importer_utils import (
    save_import_file,
    save_import_segments,
    write_import_metadata,
)
from think.streams import stream_name, update_stream, write_segment_stream
from think.utils import (
    day_path,
    get_journal,
    now_ms,
    segment_key,
    setup_cli,
)

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

logger = logging.getLogger(__name__)

TIME_RE = re.compile(r"\d{8}_\d{6}")

# Importer tract state
_callosum: CallosumConnection | None = None
_message_queue: queue.Queue | None = None
_import_id: str | None = None
_current_stage: str = "initialization"
_start_time: float = 0.0
_stage_start_time: float = 0.0
_stages_run: list[str] = []
_status_thread: threading.Thread | None = None
_status_running: bool = False


def _get_relative_path(path: str) -> str:
    """Get path relative to journal, or return as-is if not under journal."""
    journal_path = get_journal()
    try:
        return os.path.relpath(path, journal_path)
    except ValueError:
        return path


def _set_stage(stage: str) -> None:
    """Update current stage and track timing."""
    global _current_stage, _stage_start_time
    _current_stage = stage
    _stage_start_time = time.monotonic()
    if stage not in _stages_run:
        _stages_run.append(stage)
    logger.debug(f"Stage changed to: {stage}")


def _status_emitter() -> None:
    """Background thread that emits status events every 5 seconds."""
    while _status_running:
        if _callosum and _import_id:
            elapsed_ms = int((time.monotonic() - _start_time) * 1000)
            stage_elapsed_ms = int((time.monotonic() - _stage_start_time) * 1000)
            _callosum.emit(
                "importer",
                "status",
                import_id=_import_id,
                stage=_current_stage,
                elapsed_ms=elapsed_ms,
                stage_elapsed_ms=stage_elapsed_ms,
            )
        time.sleep(5)


def _write_import_jsonl(
    file_path: str,
    entries: list[dict],
    *,
    import_id: str,
    raw_filename: str | None = None,
    facet: str | None = None,
    setting: str | None = None,
) -> None:
    """Write imported transcript entries in JSONL format.

    First line contains imported metadata, subsequent lines contain entries.
    Each entry gets source="import" added to match the imported_audio.jsonl convention.

    Args:
        file_path: Path to write JSONL file
        entries: List of transcript entries
        import_id: Import identifier
        raw_filename: Source file name (relative path from segment to imports/)
        facet: Optional facet name
        setting: Optional setting description
    """
    imported_meta: dict[str, str] = {"id": import_id}
    if facet:
        imported_meta["facet"] = facet
    if setting:
        imported_meta["setting"] = setting

    # Build top-level metadata with imported info
    metadata: dict[str, object] = {"imported": imported_meta}

    # Add raw file reference (path relative from segment to imports directory)
    if raw_filename:
        metadata["raw"] = f"../../imports/{import_id}/{raw_filename}"

    # Write JSONL: metadata first, then entries with source field
    jsonl_lines = [json.dumps(metadata)]
    for entry in entries:
        # Add source field if not already present (skip metadata entries like topics/setting)
        if "text" in entry and "source" not in entry:
            entry = {**entry, "source": "import"}
        jsonl_lines.append(json.dumps(entry))

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(jsonl_lines) + "\n")


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    val = value.lower()
    if val in {"y", "yes", "true", "t", "1"}:
        return True
    if val in {"n", "no", "false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def slice_audio_segment(
    source_path: str,
    output_path: str,
    start_seconds: float,
    duration_seconds: float,
) -> str:
    """Extract an audio segment from source file, preserving original format.

    Uses stream copy for lossless extraction when possible.

    Args:
        source_path: Path to source audio file
        output_path: Path for output segment file
        start_seconds: Start offset in seconds
        duration_seconds: Duration to extract in seconds

    Returns:
        Output path on success

    Raises:
        subprocess.CalledProcessError: If ffmpeg fails
    """
    cmd = [
        "ffmpeg",
        "-ss",
        str(start_seconds),
        "-i",
        source_path,
        "-t",
        str(duration_seconds),
        "-vn",  # No video
        "-c:a",
        "copy",  # Stream copy for lossless extraction
        "-y",  # Overwrite output
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        # Fallback: re-encode if stream copy fails (some formats don't support it)
        logger.debug(f"Stream copy failed, re-encoding: {output_path}")
        cmd_reencode = [
            "ffmpeg",
            "-ss",
            str(start_seconds),
            "-i",
            source_path,
            "-t",
            str(duration_seconds),
            "-vn",
            "-y",
            output_path,
        ]
        subprocess.run(cmd_reencode, check=True, capture_output=True, text=True)

    logger.info(f"Created audio segment: {output_path}")
    return output_path


def _get_audio_duration(audio_path: str) -> float | None:
    """Get audio duration in seconds using ffprobe.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds, or None if unable to determine
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"Could not determine audio duration: {e}")
        return None


def prepare_audio_segments(
    media_path: str,
    day_dir: str,
    base_dt: dt.datetime,
    import_id: str,
) -> list[tuple[str, Path, list[str]]]:
    """Slice audio into 5-minute segments for observe pipeline.

    Creates segment directories with audio slices, ready for transcription
    via observe.observing events.

    Args:
        media_path: Path to source audio file
        day_dir: Day directory path (YYYYMMDD)
        base_dt: Base datetime for timestamp calculation
        import_id: Import identifier

    Returns:
        List of (segment_key, segment_dir, files_list) tuples
        where files_list contains the audio filename(s) created
    """
    media = Path(media_path)
    source_ext = media.suffix.lower()
    day_path_obj = Path(day_dir)

    # Get audio duration to calculate number of segments
    duration = _get_audio_duration(media_path)
    if duration is None:
        raise RuntimeError(f"Could not determine duration of {media_path}")

    # Calculate number of 5-minute segments (ceiling division)
    segment_duration = 300  # 5 minutes
    num_segments = int((duration + segment_duration - 1) // segment_duration)
    if num_segments == 0:
        num_segments = 1  # At least one segment for very short audio

    segments: list[tuple[str, Path, list[str]]] = []

    for chunk_index in range(num_segments):
        # Calculate timestamp for this segment
        ts = base_dt + timedelta(minutes=chunk_index * 5)
        time_part = ts.strftime("%H%M%S")

        # Create segment key with 5-minute duration
        segment_key_candidate = f"{time_part}_{segment_duration}"

        # Check for collision and deconflict if needed
        available_key = find_available_segment(day_path_obj, segment_key_candidate)
        if available_key is None:
            logger.warning(
                f"Could not find available segment key near {segment_key_candidate}"
            )
            continue

        if available_key != segment_key_candidate:
            logger.info(
                f"Segment collision: {segment_key_candidate} -> {available_key}"
            )

        # Create segment directory
        segment_dir = day_path_obj / available_key
        segment_dir.mkdir(parents=True, exist_ok=True)

        # Slice audio for this segment
        audio_filename = f"imported_audio{source_ext}"
        audio_path = segment_dir / audio_filename
        start_seconds = chunk_index * segment_duration

        # For the last segment, use remaining duration
        if chunk_index == num_segments - 1:
            chunk_duration = duration - start_seconds
        else:
            chunk_duration = segment_duration

        try:
            slice_audio_segment(
                media_path,
                str(audio_path),
                start_seconds,
                chunk_duration,
            )
            segments.append((available_key, segment_dir, [audio_filename]))
            logger.info(f"Created segment: {available_key} with {audio_filename}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to slice segment {available_key}: {e}")
            # Clean up empty directory
            if segment_dir.exists() and not any(segment_dir.iterdir()):
                segment_dir.rmdir()

    return segments


def _read_transcript(path: str) -> str:
    """Return transcript text from a .txt/.md/.pdf file."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".md"}:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if ext == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf required for PDF support")
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)
        return "\n".join(parts)
    raise ValueError("unsupported transcript format")


def _time_to_seconds(time_str: str) -> int:
    """Convert HH:MM:SS time string to seconds from midnight."""
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s


def process_transcript(
    path: str,
    day_dir: str,
    base_dt: dt.datetime,
    *,
    import_id: str,
    facet: str | None = None,
    setting: str | None = None,
    audio_duration: int | None = None,
) -> list[str]:
    """Process a transcript file and write imported JSONL segments.

    Args:
        path: Path to transcript file
        day_dir: Journal day directory
        base_dt: Base datetime for the import
        import_id: Import identifier
        facet: Optional facet name
        setting: Optional setting description
        audio_duration: Optional total audio duration in seconds (for last segment)

    Returns:
        List of created file paths.
    """
    created_files = []
    text = _read_transcript(path)

    # Get start time from base_dt for segmentation
    start_time = base_dt.strftime("%H:%M:%S")

    # Get segments with their absolute start times
    segments = detect_transcript_segment(text, start_time)

    for idx, (start_at, seg_text) in enumerate(segments):
        # Convert segment text to structured JSON with absolute timestamps
        json_data = detect_transcript_json(seg_text, start_at)
        if not json_data:
            continue

        # Parse absolute time for segment directory name
        time_part = start_at.replace(":", "")  # "12:05:30" -> "120530"

        # Compute segment duration from absolute times
        start_seconds = _time_to_seconds(start_at)
        if idx + 1 < len(segments):
            next_start_at, _ = segments[idx + 1]
            next_seconds = _time_to_seconds(next_start_at)
            duration = next_seconds - start_seconds
        else:
            # Last segment: use remaining audio duration or default +5s
            if audio_duration:
                # audio_duration is total length, start_seconds is time-of-day
                # Need to calculate offset from recording start
                recording_start_seconds = _time_to_seconds(start_time)
                segment_offset = start_seconds - recording_start_seconds
                duration = audio_duration - segment_offset
            else:
                duration = 5

        # Negative duration indicates corrupted/invalid timestamp data
        if duration < 0:
            raise ValueError(
                f"Invalid segment duration: {duration}s for segment at {time_part}. "
                "Timestamps may be out of order or audio_duration is incorrect."
            )

        # Ensure minimum duration of 1 second
        duration = max(1, duration)

        segment_name = f"{time_part}_{duration}"
        ts_dir = os.path.join(day_dir, segment_name)
        os.makedirs(ts_dir, exist_ok=True)
        json_path = os.path.join(ts_dir, "imported_audio.jsonl")

        _write_import_jsonl(
            json_path,
            json_data,
            import_id=import_id,
            raw_filename=os.path.basename(path),
            facet=facet,
            setting=setting,
        )
        logger.info(f"Added transcript segment to journal: {json_path}")
        created_files.append(json_path)

    return created_files


def _run_import_summary(
    import_dir: Path,
    day: str,
    segments: list[str],
) -> bool:
    """Create a summary for imported segments using cortex generator.

    Args:
        import_dir: Directory where the summary will be saved
        day: Day string (YYYYMMDD format)
        segments: List of segment keys to summarize

    Returns:
        True if summary was created successfully, False otherwise
    """
    from think.cortex_client import cortex_request, wait_for_agents

    if not segments:
        logger.info("No segments to summarize")
        return False

    summary_path = import_dir / "summary.md"

    try:
        logger.info(f"Creating summary for {len(segments)} segments via cortex")

        # Spawn generator via cortex
        agent_id = cortex_request(
            prompt="",  # Generators don't use prompt
            name="importer",
            config={
                "day": day,
                "span": segments,
                "output": "md",
                "output_path": str(summary_path),
            },
        )

        # Wait for completion
        completed, timed_out = wait_for_agents([agent_id], timeout=300)

        if timed_out:
            logger.error(f"Import summary timed out (ID: {agent_id})")
            return False

        if completed:
            end_state = completed.get(agent_id, "unknown")
            if end_state == "finish" and summary_path.exists():
                logger.info(f"Created import summary: {summary_path}")
                return True
            else:
                logger.warning(
                    f"Generator ended with state {end_state}, "
                    f"summary exists: {summary_path.exists()}"
                )
                return False

        logger.error("Generator did not complete")
        return False

    except Exception as e:
        logger.error(f"Failed to create summary: {e}")
        return False


# MIME type mapping for import metadata
_MIME_TYPES = {
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
}


def _is_in_imports(media_path: str) -> bool:
    """Check if file path is already under journal/imports/."""
    imports_dir = os.path.join(get_journal(), "imports")
    abs_media = os.path.abspath(media_path)
    abs_imports = os.path.abspath(imports_dir)
    return abs_media.startswith(abs_imports + os.sep)


def _setup_import(
    media_path: str,
    timestamp: str,
    facet: str | None,
    setting: str | None,
    detection_result: dict | None,
    force: bool = False,
) -> str:
    """Copy file to imports/ and write metadata. Returns new file path."""
    journal_root = Path(get_journal())
    import_dir = journal_root / "imports" / timestamp

    # Check for conflict
    if import_dir.exists():
        if force:
            logger.info(f"Removing existing import directory: {import_dir}")
            shutil.rmtree(import_dir)
        else:
            raise SystemExit(
                f"Error: Import already exists for timestamp {timestamp}\n"
                f"To re-import, use --force to delete existing data and start over"
            )

    # Copy file to imports/
    filename = os.path.basename(media_path)
    new_path = save_import_file(
        journal_root=journal_root,
        timestamp=timestamp,
        source_path=Path(media_path),
        filename=filename,
    )

    # Build metadata matching app structure
    upload_ts = now_ms()
    ext = os.path.splitext(filename)[1].lower()
    metadata = {
        "original_filename": filename,
        "upload_timestamp": upload_ts,
        "upload_datetime": dt.datetime.fromtimestamp(upload_ts / 1000).isoformat(),
        "detection_result": detection_result,
        "detected_timestamp": timestamp,
        "user_timestamp": timestamp,
        "file_size": new_path.stat().st_size if new_path.exists() else 0,
        "mime_type": _MIME_TYPES.get(ext, "application/octet-stream"),
        "facet": facet,
        "setting": setting,
        "file_path": str(new_path),
    }

    write_import_metadata(
        journal_root=journal_root,
        timestamp=timestamp,
        metadata=metadata,
    )

    logger.info(f"Copied to journal: {new_path}")
    return str(new_path)


def _format_timestamp_display(timestamp: str) -> str:
    """Format timestamp for human-readable display."""
    try:
        dt_obj = dt.datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        return dt_obj.strftime("%a %b %d %Y, %-I:%M %p")
    except ValueError:
        return timestamp


def main() -> None:
    global _callosum, _message_queue, _import_id, _current_stage, _start_time
    global _stage_start_time, _stages_run, _status_thread, _status_running

    parser = argparse.ArgumentParser(description="Chunk a media file into the journal")
    parser.add_argument("media", help="Path to video or audio file")
    parser.add_argument(
        "--timestamp", help="Timestamp YYYYMMDD_HHMMSS for journal entry"
    )
    parser.add_argument(
        "--summarize",
        type=str2bool,
        default=True,
        help="Create summary.md after transcription completes",
    )
    parser.add_argument(
        "--facet",
        type=str,
        default=None,
        help="Facet name for this import",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default=None,
        help="Contextual setting description to store with import metadata",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Import source type (apple, plaud, audio, text). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip waiting for transcription and summary generation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-import by deleting existing import directory",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-accept detected timestamp and proceed with import",
    )
    args, extra = setup_cli(parser, parse_known=True)
    if extra and not args.timestamp:
        args.timestamp = extra[0]

    # Track detection result for metadata
    detection_result = None

    # If no timestamp provided, detect it
    if not args.timestamp:
        # Pass the original filename for better detection
        detection_result = detect_created(
            args.media, original_filename=os.path.basename(args.media)
        )
        if (
            detection_result
            and detection_result.get("day")
            and detection_result.get("time")
        ):
            detected_timestamp = f"{detection_result['day']}_{detection_result['time']}"
            display = _format_timestamp_display(detected_timestamp)
            if args.auto:
                print(
                    f"Detected timestamp: {detected_timestamp} ({display}) — auto-importing"
                )
                args.timestamp = detected_timestamp
            else:
                print(f"Detected timestamp: {detected_timestamp} ({display})")
                print("\nRun:")
                print(f"  sol import {args.media} --timestamp {detected_timestamp}")
                return
        else:
            raise SystemExit(
                "Could not detect timestamp. Please provide --timestamp YYYYMMDD_HHMMSS"
            )

    if not TIME_RE.fullmatch(args.timestamp):
        raise SystemExit("timestamp must be in YYYYMMDD_HHMMSS format")

    # Check if file needs setup (not already in imports/)
    needs_setup = not _is_in_imports(args.media)

    # Copy to imports/ if file is not already there
    if needs_setup:
        args.media = _setup_import(
            args.media,
            args.timestamp,
            args.facet,
            args.setting,
            detection_result,
            force=args.force,
        )
        print("Starting import...")

    base_dt = dt.datetime.strptime(args.timestamp, "%Y%m%d_%H%M%S")
    day = base_dt.strftime("%Y%m%d")
    logger.info(f"Using provided timestamp: {args.timestamp}")
    day_dir = str(day_path(day))

    # Derive stream identity for this import
    if args.source:
        import_source = args.source
    else:
        # Auto-detect from file extension
        _ext = os.path.splitext(args.media)[1].lower()
        if _ext == ".m4a":
            import_source = "apple"
        elif _ext in {".txt", ".md", ".pdf"}:
            import_source = "text"
        else:
            import_source = "audio"
    stream = stream_name(import_source=import_source)

    # Initialize importer tract state
    _import_id = args.timestamp
    _start_time = time.monotonic()
    _stage_start_time = _start_time
    _current_stage = "initialization"
    _stages_run = ["initialization"]

    # Start Callosum connection with message queue for receiving events
    _message_queue = queue.Queue()
    _callosum = CallosumConnection()
    _callosum.start(callback=lambda msg: _message_queue.put(msg))

    # Start status emitter thread
    _status_running = True
    _status_thread = threading.Thread(target=_status_emitter, daemon=True)
    _status_thread.start()

    # Emit started event
    ext = os.path.splitext(args.media)[1].lower()
    _callosum.emit(
        "importer",
        "started",
        import_id=_import_id,
        input_file=os.path.basename(args.media),
        file_type=ext.lstrip("."),
        day=day,
        facet=args.facet,
        setting=args.setting,
        options={
            "summarize": args.summarize,
            "skip_summary": args.skip_summary,
        },
        stage=_current_stage,
        stream=stream,
    )

    # Track all created files and processing metadata
    all_created_files: list[str] = []
    created_segments: list[str] = []
    journal_root = Path(get_journal())
    processing_results = {
        "processed_timestamp": args.timestamp,
        "target_day": base_dt.strftime("%Y%m%d"),
        "target_day_path": day_dir,
        "input_file": args.media,
        "processing_started": dt.datetime.now().isoformat(),
        "facet": args.facet,
        "setting": args.setting,
        "outputs": [],
    }

    # Get parent directory for saving metadata
    media_path = Path(args.media)
    import_dir = media_path.parent
    failed_segments: list[str] = []

    try:
        if ext in {".txt", ".md", ".pdf"}:
            # Text transcript processing (unchanged - no observe pipeline)
            _set_stage("segmenting")

            created_files = process_transcript(
                args.media,
                day_dir,
                base_dt,
                import_id=args.timestamp,
                facet=args.facet,
                setting=args.setting,
            )
            all_created_files.extend(created_files)
            processing_results["outputs"].append(
                {
                    "type": "transcript",
                    "format": "imported_audio.jsonl",
                    "description": "Transcript segments",
                    "files": created_files,
                    "count": len(created_files),
                }
            )

            # Extract segment keys for text imports
            for file_path in created_files:
                seg = segment_key(file_path)
                if seg and seg not in created_segments:
                    created_segments.append(seg)

            # Write stream markers for text import segments
            for seg in created_segments:
                try:
                    seg_dir = day_path(day) / seg
                    result = update_stream(stream, day, seg, type="import", host=None)
                    write_segment_stream(
                        seg_dir,
                        stream,
                        result["prev_day"],
                        result["prev_segment"],
                        result["seq"],
                    )
                except Exception as e:
                    logger.warning(f"Failed to write stream identity: {e}")

            # Emit observe.observed for text imports (already processed)
            for seg in created_segments:
                _callosum.emit(
                    "observe", "observed", segment=seg, day=day, stream=stream
                )
                logger.info(f"Emitted observe.observed for segment: {day}/{seg}")

        else:
            # Audio processing via observe pipeline
            _set_stage("segmenting")

            # Prepare audio segments (slice into 5-minute chunks)
            segments = prepare_audio_segments(
                args.media,
                day_dir,
                base_dt,
                args.timestamp,
            )

            if not segments:
                raise RuntimeError("No segments created from audio file")

            # Track created files and segment keys, write stream markers
            for seg_key, seg_dir, files in segments:
                created_segments.append(seg_key)
                for f in files:
                    all_created_files.append(str(seg_dir / f))
                try:
                    result = update_stream(
                        stream, day, seg_key, type="import", host=None
                    )
                    write_segment_stream(
                        seg_dir,
                        stream,
                        result["prev_day"],
                        result["prev_segment"],
                        result["seq"],
                    )
                except Exception as e:
                    logger.warning(f"Failed to write stream identity: {e}")

            # Save segment list for tracking
            save_import_segments(journal_root, args.timestamp, created_segments, day)

            processing_results["outputs"].append(
                {
                    "type": "audio_segments",
                    "description": "Audio segments queued for transcription",
                    "segments": created_segments,
                    "count": len(created_segments),
                }
            )

            # Build meta dict for observe.observing events
            meta: dict[str, str] = {"import_id": args.timestamp, "stream": stream}
            if args.facet:
                meta["facet"] = args.facet
            if args.setting:
                meta["setting"] = args.setting

            # Emit observe.observing per segment to trigger sense.py transcription
            for seg_key, seg_dir, files in segments:
                _callosum.emit(
                    "observe",
                    "observing",
                    segment=seg_key,
                    day=day,
                    files=files,
                    meta=meta,
                    stream=stream,
                )
                logger.info(f"Emitted observe.observing for segment: {day}/{seg_key}")

            # Wait for transcription to complete (unless --no-wait)
            if not args.skip_summary:
                _set_stage("transcribing")
                pending = set(created_segments)
                segment_timeout = 600  # 10 minutes since last progress
                last_progress = time.monotonic()

                logger.info(f"Waiting for {len(pending)} segments to complete")

                while pending:
                    # Poll for observe.observed events from message queue
                    try:
                        msg = _message_queue.get(timeout=5.0)
                    except queue.Empty:
                        # Check for timeout since last progress
                        if time.monotonic() - last_progress > segment_timeout:
                            timed_out = sorted(pending)
                            logger.error(f"Timed out waiting for segments: {timed_out}")
                            failed_segments.extend(timed_out)
                            pending.clear()
                        continue

                    tract = msg.get("tract")
                    event = msg.get("event")
                    seg = msg.get("segment")

                    if tract == "observe" and event == "observed" and seg in pending:
                        pending.discard(seg)
                        last_progress = time.monotonic()
                        if msg.get("error"):
                            errors = msg.get("errors", [])
                            logger.warning(
                                f"Segment {seg} failed: {errors} "
                                f"({len(pending)} remaining)"
                            )
                            failed_segments.append(seg)
                        else:
                            logger.info(
                                f"Segment {seg} transcribed "
                                f"({len(pending)} remaining)"
                            )

                if failed_segments:
                    logger.warning(
                        f"{len(failed_segments)} of {len(created_segments)} "
                        f"segments failed: {failed_segments}"
                    )
                else:
                    logger.info("All segments transcribed successfully")

        # Complete processing metadata
        processing_results["processing_completed"] = dt.datetime.now().isoformat()
        processing_results["total_files_created"] = len(all_created_files)
        processing_results["all_created_files"] = all_created_files
        processing_results["segments"] = created_segments
        if failed_segments:
            processing_results["failed_segments"] = failed_segments

        # Write imported.json with all processing metadata
        imported_path = import_dir / "imported.json"
        try:
            with open(imported_path, "w", encoding="utf-8") as f:
                json.dump(processing_results, f, indent=2)
            logger.info(f"Saved import processing metadata: {imported_path}")
        except Exception as e:
            logger.warning(f"Failed to save imported.json: {e}")

        # Update import.json with processing summary if it exists
        import_metadata_path = import_dir / "import.json"
        if import_metadata_path.exists():
            try:
                with open(import_metadata_path, "r", encoding="utf-8") as f:
                    import_meta = json.load(f)
                import_meta["processing_completed"] = processing_results[
                    "processing_completed"
                ]
                import_meta["total_files_created"] = processing_results[
                    "total_files_created"
                ]
                import_meta["imported_json_path"] = str(imported_path)
                import_meta["segments"] = created_segments
                with open(import_metadata_path, "w", encoding="utf-8") as f:
                    json.dump(import_meta, f, indent=2)
                logger.info(f"Updated import metadata: {import_metadata_path}")
            except Exception as e:
                logger.warning(f"Failed to update import metadata: {e}")

        # Create summary if requested and we have segments
        if args.summarize and created_segments and not args.skip_summary:
            _set_stage("summarizing")
            _run_import_summary(import_dir, day, created_segments)

        # Emit completed event
        duration_ms = int((time.monotonic() - _start_time) * 1000)
        output_files_relative = [_get_relative_path(f) for f in all_created_files]
        metadata_file_relative = _get_relative_path(str(imported_path))

        _callosum.emit(
            "importer",
            "completed",
            import_id=_import_id,
            stage=_current_stage,
            duration_ms=duration_ms,
            total_files_created=len(all_created_files),
            output_files=output_files_relative,
            metadata_file=metadata_file_relative,
            stages_run=_stages_run,
            segments=created_segments,
            stream=stream,
        )

    except Exception as e:
        # Write error state to imported.json for persistent failure tracking
        duration_ms = int((time.monotonic() - _start_time) * 1000)
        partial_outputs = [_get_relative_path(f) for f in all_created_files]
        imported_path = import_dir / "imported.json"

        error_results = {
            **processing_results,  # Include all the metadata we have
            "processing_failed": dt.datetime.now().isoformat(),
            "error": str(e),
            "error_stage": _current_stage,
            "duration_ms": duration_ms,
            "total_files_created": len(all_created_files),
            "all_created_files": all_created_files,
            "stages_run": _stages_run,
        }

        try:
            with open(imported_path, "w", encoding="utf-8") as f:
                json.dump(error_results, f, indent=2)
            logger.info(f"Saved error state: {imported_path}")
        except Exception as write_err:
            logger.warning(f"Failed to write error state: {write_err}")

        # Emit error event
        if _callosum:
            _callosum.emit(
                "importer",
                "error",
                import_id=_import_id,
                stage=_current_stage,
                error=str(e),
                duration_ms=duration_ms,
                partial_outputs=partial_outputs,
            )

        logger.error(f"Import failed: {e}")
        raise

    finally:
        # Stop status thread
        _status_running = False
        if _status_thread:
            _status_thread.join(timeout=6)


if __name__ == "__main__":
    main()
