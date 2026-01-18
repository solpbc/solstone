# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import argparse
import datetime as dt
import json
import logging
import os
import re
import string
import subprocess
import threading
import time
import unicodedata
from datetime import timedelta
from pathlib import Path

from muse.models import generate
from observe.hear import load_transcript
from observe.transcribe.revai import convert_to_statements, transcribe_file
from think.callosum import CallosumConnection
from think.detect_created import detect_created
from think.detect_transcript import detect_transcript_json, detect_transcript_segment
from think.facets import get_facets
from think.importer_utils import save_import_file, write_import_metadata
from think.utils import (
    PromptNotFoundError,
    day_path,
    get_journal,
    load_prompt,
    segment_key,
    setup_cli,
)

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

logger = logging.getLogger(__name__)

TIME_RE = re.compile(r"\d{8}_\d{6}")
_ALLOWED_ASCII = set(string.ascii_letters + string.punctuation + " ")

# Importer tract state
_callosum: CallosumConnection | None = None
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
    raw_is_local: bool = False,
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
        raw_filename: Audio file name (local filename or imports/ relative path)
        raw_is_local: If True, raw_filename is in the segment dir (no imports/ prefix)
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

    # Add raw audio file reference if provided
    if raw_filename:
        if raw_is_local:
            # Local file in segment directory
            metadata["raw"] = raw_filename
        else:
            # Path is relative from segment directory to imports directory
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


def _sanitize_entities(entities: list[str]) -> list[str]:
    """Return Rev AI safe custom vocabulary phrases."""

    sanitized: list[str] = []
    seen: set[str] = set()

    for original in entities:
        normalized = unicodedata.normalize("NFKD", original)
        ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
        filtered = "".join(ch for ch in ascii_only if ch in _ALLOWED_ASCII)
        filtered = re.sub(r"\s+", " ", filtered).strip()

        if not filtered or not any(ch.isalpha() for ch in filtered):
            logger.debug(
                "Dropping entity without alpha characters after sanitizing: %s",
                original,
            )
            continue

        if filtered != original:
            logger.debug("Sanitized entity '%s' -> '%s'", original, filtered)

        if filtered in seen:
            continue

        seen.add(filtered)
        sanitized.append(filtered)

    return sanitized


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


def audio_transcribe(
    path: str,
    day_dir: str,
    base_dt: dt.datetime,
    *,
    import_id: str,
    facet: str | None = None,
    setting: str | None = None,
) -> tuple[list[str], dict]:
    """Transcribe audio using Rev AI and save 5-minute chunks as imported JSONL.

    Args:
        path: Path to audio file
        day_dir: Directory to save chunks to
        base_dt: Base datetime for timestamps
        facet: Optional facet name to extract entities from
        setting: Optional description of the setting to store with metadata

    Returns:
        Tuple of (list of created file paths, raw RevAI JSON result)
    """
    logger.info(f"Transcribing audio file: {path}")
    media_path = Path(path)
    created_files = []

    # Get facet entities if facet is specified
    entities = None
    if facet:
        try:
            from think.entities import load_entity_names

            # Load entity names from facet-specific entities.jsonl (spoken mode for short forms)
            entity_names = load_entity_names(facet=facet, spoken=True)
            if entity_names:
                # entity_names is already a list in spoken mode
                entities = _sanitize_entities(entity_names)
                if entities:
                    logger.info(
                        f"Using {len(entities)} entities from facet '{facet}' for transcription"
                    )
                else:
                    logger.info(f"Facet '{facet}' entities removed after sanitization")
            else:
                logger.info(f"No entities found for facet '{facet}'")
        except FileNotFoundError:
            logger.info(f"Facet '{facet}' has no entities.jsonl file")
        except Exception as e:
            logger.warning(f"Failed to load facet entities: {e}")

    # Build Rev.ai config
    revai_config: dict = {}
    if entities:
        revai_config["entities"] = entities

    # Transcribe using Rev AI
    try:
        revai_json = transcribe_file(media_path, revai_config)
    except Exception as e:
        logger.error(f"Failed to transcribe audio: {e}")
        raise

    # Convert to statements (per-speaker, with float timestamps)
    statements = convert_to_statements(revai_json)

    if not statements:
        logger.warning("No transcript entries found")
        return created_files, revai_json

    # Group statements into 5-minute chunks based on float start times
    chunks = []
    current_chunk = []
    chunk_start_time = None

    for stmt in statements:
        # Use float seconds directly (no string parsing needed)
        start_seconds = int(stmt.get("start", 0.0))

        # Determine which 5-minute chunk this belongs to
        chunk_index = start_seconds // 300  # 300 seconds = 5 minutes

        # If this is a new chunk, save the previous one
        if chunk_start_time is not None and chunk_index != chunk_start_time:
            if current_chunk:
                chunks.append((chunk_start_time, current_chunk))
            current_chunk = []
            chunk_start_time = chunk_index
        elif chunk_start_time is None:
            chunk_start_time = chunk_index

        current_chunk.append(stmt)

    # Don't forget the last chunk
    if current_chunk:
        chunks.append((chunk_start_time, current_chunk))

    # Get source file extension for audio slices
    source_ext = media_path.suffix.lower()

    # Save each chunk as a separate JSONL file with audio slice
    for chunk_index, chunk_entries in chunks:
        # Calculate timestamp for this chunk
        ts = base_dt + timedelta(minutes=chunk_index * 5)
        time_part = ts.strftime("%H%M%S")

        # Create segment directory with 5-minute (300 second) duration suffix
        segment_name = f"{time_part}_300"
        ts_dir = os.path.join(day_dir, segment_name)
        os.makedirs(ts_dir, exist_ok=True)
        json_path = os.path.join(ts_dir, "imported_audio.jsonl")

        # Extract audio slice for this segment
        audio_filename = f"imported_audio{source_ext}"
        audio_path = os.path.join(ts_dir, audio_filename)
        start_seconds = chunk_index * 300
        # Use 300s duration - ffmpeg handles EOF gracefully for last chunk
        duration = 300

        try:
            slice_audio_segment(path, audio_path, start_seconds, duration)
            created_files.append(audio_path)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to slice audio segment: {e}")
            audio_filename = None

        # Convert statements to entries with absolute timestamps
        absolute_entries = []
        for stmt in chunk_entries:
            # Convert float seconds to absolute HH:MM:SS
            relative_seconds = stmt.get("start", 0.0)
            absolute_dt = base_dt + timedelta(seconds=relative_seconds)

            entry = {
                "start": absolute_dt.strftime("%H:%M:%S"),
                "source": "import",
                "speaker": stmt.get("speaker", 1),
                "text": stmt.get("text", ""),
            }

            # Add description based on confidence
            confidence = stmt.get("confidence")
            if confidence is not None:
                if confidence < 0.7:
                    entry["description"] = "low confidence"
                elif confidence > 0.95:
                    entry["description"] = "clear"

            absolute_entries.append(entry)

        # Save the chunk with absolute timestamps
        # raw points to local audio slice (or None if slicing failed)
        _write_import_jsonl(
            json_path,
            absolute_entries,
            import_id=import_id,
            raw_filename=audio_filename,
            raw_is_local=True,
            facet=facet,
            setting=setting,
        )
        logger.info(f"Added transcript chunk to journal: {json_path}")
        created_files.append(json_path)

    return created_files, revai_json


def create_transcript_summary(
    import_dir: Path,
    audio_json_files: list[str],
    input_filename: str,
    timestamp: str,
    setting: str | None = None,
    facet: str | None = None,
) -> None:
    """Create a summary of all imported audio transcript files using LLM analysis.

    Args:
        import_dir: Directory where the summary will be saved
        audio_json_files: List of paths to imported_audio.jsonl files
        input_filename: Original media filename for context
        timestamp: Processing timestamp for context
        setting: Optional description of the setting to include in metadata
        facet: Optional facet name to include context about entities and description
    """
    if not audio_json_files:
        logger.info("No audio transcript files to summarize")
        return

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not set, skipping summarization")
        return

    # Read all transcript chunks
    all_transcripts = []
    for json_path in audio_json_files:
        try:
            # Load transcript with formatted text
            metadata, entries, formatted_text = load_transcript(json_path)
            if entries is None:
                error_msg = metadata.get("error", "Unknown error")
                logger.warning(f"Failed to read {json_path}: {error_msg}")
                continue

            all_transcripts.append(
                {
                    "file": os.path.basename(json_path),
                    "text": formatted_text,
                    "entry_count": len(entries),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to read {json_path}: {e}")

    if not all_transcripts:
        logger.warning("No transcripts could be read for summarization")
        return

    # Load the prompt from importer.txt (with journal preamble for identity context)
    try:
        importer_prompt = load_prompt(
            "importer", base_dir=Path(__file__).parent, include_journal=True
        )
    except PromptNotFoundError as exc:
        logger.error(f"Failed to load importer prompt: {exc}")
        return
    importer_prompt_template = importer_prompt.text

    # Add facet context if a facet is specified
    facet_context = ""
    if facet:
        try:
            from think.facets import facet_summary

            facet_context = facet_summary(facet)
            logger.info(f"Including facet context for '{facet}'")
        except FileNotFoundError:
            logger.warning(f"Facet '{facet}' not found, skipping facet context")
        except Exception as e:
            logger.warning(f"Failed to load facet context: {e}")

    # Add the context metadata to the prompt
    metadata_lines = [
        "\n\n## Metadata for this summary:",
        f"- Original file: {input_filename}",
        f"- Recording timestamp: {timestamp}",
        f"- Number of transcript segments: {len(all_transcripts)}",
        f"- Total transcript entries: {sum(t['entry_count'] for t in all_transcripts)}",
    ]
    if setting:
        metadata_lines.append(f"- Setting: {setting}")
    if facet:
        metadata_lines.append(f"- Facet: {facet}")

    # Combine: template + facet context + metadata
    prompt_parts = [importer_prompt_template]
    if facet_context:
        prompt_parts.append(f"\n\n## Facet Context\n{facet_context}")
    prompt_parts.append("\n".join(metadata_lines))
    importer_prompt = "".join(prompt_parts)

    # Format the transcript content for the user message
    user_message_parts = []

    for transcript_info in all_transcripts:
        user_message_parts.append(
            f"\n## Transcript Segment: {transcript_info['file']}\n"
        )
        user_message_parts.append(transcript_info["text"])

    user_message = "\n".join(user_message_parts)

    try:
        logger.info(f"Creating summary for {len(all_transcripts)} transcript segments")

        # Generate summary using configured provider
        response_text = generate(
            contents=user_message,
            context="observe.summarize",
            temperature=0.3,
            max_output_tokens=8192 * 4,
            system_instruction=importer_prompt,
        )

        # Save the summary
        summary_path = import_dir / "summary.md"
        total_entries = sum(t["entry_count"] for t in all_transcripts)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# Audio Transcript Summary\n\n")
            f.write(f"**Source File:** {input_filename}\n")
            f.write(f"**Import Timestamp:** {timestamp}\n")
            f.write(f"**Segments Processed:** {len(all_transcripts)}\n")
            f.write(f"**Total Entries:** {total_entries}\n\n")
            if setting:
                f.write(f"**Setting:** {setting}\n\n")
            f.write("---\n\n")
            f.write(response_text)

        logger.info(f"Created transcript summary: {summary_path}")

    except Exception as e:
        logger.error(f"Failed to create summary: {e}")
        # Don't fail the entire import process if summarization fails
        pass


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
) -> str:
    """Copy file to imports/ and write metadata. Returns new file path."""
    journal_root = Path(get_journal())
    import_dir = journal_root / "imports" / timestamp

    # Check for conflict
    if import_dir.exists():
        raise SystemExit(
            f"Error: Import already exists for timestamp {timestamp}\n"
            f"Use a different timestamp or delete: {import_dir}"
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
    now_ms = int(time.time() * 1000)
    ext = os.path.splitext(filename)[1].lower()
    metadata = {
        "original_filename": filename,
        "upload_timestamp": now_ms,
        "upload_datetime": dt.datetime.fromtimestamp(now_ms / 1000).isoformat(),
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


def _print_facets_help(facets: dict, media_path: str, timestamp: str) -> None:
    """Print available facets and suggested command."""
    print("\nAvailable facets:")
    max_name = max(len(name) for name in facets.keys())
    for name, info in sorted(facets.items()):
        title = info.get("title", name)
        print(f"  {name:<{max_name}}  {title}")

    print("\nAdd --facet <name>:")
    print(f"  sol import {media_path} --timestamp {timestamp} --facet <name>")


def _print_setting_help(media_path: str, timestamp: str, facet: str) -> None:
    """Print setting requirement and suggested command."""
    print("\nSetting is required (describes the context of this recording).")
    print("Examples: 'Team standup meeting', 'Jer lunch with Joe', 'Conference talk'")
    print("\nAdd --setting <description>:")
    print(
        f"  sol import {media_path} --timestamp {timestamp} "
        f'--facet {facet} --setting "description"'
    )


def main() -> None:
    global _callosum, _import_id, _current_stage, _start_time, _stage_start_time
    global _stages_run, _status_thread, _status_running

    parser = argparse.ArgumentParser(description="Chunk a media file into the journal")
    parser.add_argument("media", help="Path to video or audio file")
    parser.add_argument(
        "--timestamp", help="Timestamp YYYYMMDD_HHMMSS for journal entry"
    )
    parser.add_argument(
        "--hear", type=str2bool, default=True, help="Transcribe audio using Rev AI"
    )
    parser.add_argument(
        "--summarize",
        type=str2bool,
        default=True,
        help="Create summary.md for audio transcripts",
    )
    parser.add_argument(
        "--facet",
        type=str,
        default=None,
        help="Facet name to use for entity extraction",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default=None,
        help="Contextual setting description to store with import metadata",
    )
    args, extra = setup_cli(parser, parse_known=True)
    if extra and not args.timestamp:
        args.timestamp = extra[0]

    # Track detection result for metadata
    detection_result = None

    # If no timestamp provided, detect it and show instruction
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

    # Check facet requirement (only for files not already in imports/)
    if needs_setup and not args.facet:
        facets = get_facets()
        if facets:
            _print_facets_help(facets, args.media, args.timestamp)
            return
        # No facets configured - proceed without

    # Check setting requirement (only for files not already in imports/)
    if needs_setup and not args.setting and args.facet:
        _print_setting_help(args.media, args.timestamp, args.facet)
        return

    # Copy to imports/ if file is not already there
    if needs_setup:
        print("Copying to journal...")
        args.media = _setup_import(
            args.media, args.timestamp, args.facet, args.setting, detection_result
        )
        print("Starting import...")

    base_dt = dt.datetime.strptime(args.timestamp, "%Y%m%d_%H%M%S")
    day = base_dt.strftime("%Y%m%d")
    logger.info(f"Using provided timestamp: {args.timestamp}")
    day_dir = str(day_path(day))

    # Initialize importer tract state
    _import_id = args.timestamp
    _start_time = time.monotonic()
    _stage_start_time = _start_time
    _current_stage = "initialization"
    _stages_run = ["initialization"]

    # Start Callosum connection
    _callosum = CallosumConnection()
    _callosum.start()

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
            "hear": args.hear,
            "summarize": args.summarize,
        },
        stage=_current_stage,
    )

    # Track all created files and processing metadata
    all_created_files = []
    audio_transcript_files = []  # Track audio transcript files for summarization
    revai_json_data = None
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

    try:
        if ext in {".txt", ".md", ".pdf"}:
            # Set stage for transcript segmentation
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
            audio_transcript_files.extend(created_files)  # Track for summarization
            processing_results["outputs"].append(
                {
                    "type": "transcript",
                    "format": "imported_audio.jsonl",
                    "description": "Transcript segments",
                    "files": created_files,
                    "count": len(created_files),
                }
            )
        else:
            if args.hear:
                # Set stage for audio transcription
                _set_stage("transcribing")

                created_files, revai_json_data = audio_transcribe(
                    args.media,
                    day_dir,
                    base_dt,
                    import_id=args.timestamp,
                    facet=args.facet,
                    setting=args.setting,
                )
                all_created_files.extend(created_files)
                audio_transcript_files.extend(created_files)  # Track for summarization
                processing_results["outputs"].append(
                    {
                        "type": "audio_transcript",
                        "format": "imported_audio.jsonl",
                        "description": "Rev AI transcription chunks",
                        "files": created_files,
                        "count": len(created_files),
                        "transcription_service": "RevAI",
                    }
                )

        # Complete processing metadata
        processing_results["processing_completed"] = dt.datetime.now().isoformat()
        processing_results["total_files_created"] = len(all_created_files)
        processing_results["all_created_files"] = all_created_files

        # Get parent directory for saving metadata
        media_path = Path(args.media)
        import_dir = media_path.parent

        # Save RevAI JSON if we have it
        if revai_json_data:
            revai_path = import_dir / "revai.json"
            try:
                with open(revai_path, "w", encoding="utf-8") as f:
                    json.dump(revai_json_data, f, indent=2)
                logger.info(f"Saved raw RevAI transcription: {revai_path}")
                processing_results["revai_json_path"] = str(revai_path)
            except Exception as e:
                logger.warning(f"Failed to save RevAI JSON: {e}")

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
                    metadata = json.load(f)
                metadata["processing_completed"] = processing_results[
                    "processing_completed"
                ]
                metadata["total_files_created"] = processing_results[
                    "total_files_created"
                ]
                metadata["imported_json_path"] = str(imported_path)
                if revai_json_data:
                    metadata["revai_json_path"] = str(revai_path)
                with open(import_metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Updated import metadata: {import_metadata_path}")
            except Exception as e:
                logger.warning(f"Failed to update import metadata: {e}")

        # Extract segment names from created transcript files for event emission
        # Path format: YYYYMMDD/HHMMSS_300/imported_audio.jsonl
        created_segments = []
        for file_path in audio_transcript_files:
            seg = segment_key(file_path)
            if seg and seg not in created_segments:
                created_segments.append(seg)

        # Emit observe.observed events for each segment to trigger segment processing
        # This allows segment insights (sol dream --segment) to run in parallel with summary
        for seg in created_segments:
            _callosum.emit(
                "observe",
                "observed",
                segment=seg,
                day=day,
            )
            logger.info(f"Emitted observe.observed for segment: {day}/{seg}")

        # Create summary if requested and audio transcripts were created
        if args.summarize and audio_transcript_files:
            # Set stage for summarization
            _set_stage("summarizing")

            # Filter to only JSONL files (exclude audio slices)
            jsonl_files = [f for f in audio_transcript_files if f.endswith(".jsonl")]
            create_transcript_summary(
                import_dir=import_dir,
                audio_json_files=jsonl_files,
                input_filename=os.path.basename(args.media),
                timestamp=args.timestamp,
                setting=args.setting,
                facet=args.facet,
            )

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
        )

    except Exception as e:
        # Write error state to imported.json for persistent failure tracking
        duration_ms = int((time.monotonic() - _start_time) * 1000)
        partial_outputs = [_get_relative_path(f) for f in all_created_files]

        media_path = Path(args.media)
        import_dir = media_path.parent
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
