import argparse
import datetime as dt
import json
import logging
import os
import re
import string
import subprocess
import tempfile
import threading
import time
import unicodedata
from datetime import timedelta
from pathlib import Path

import cv2
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from observe.hear import load_transcript
from observe.revai import convert_revai_to_sunstone, transcribe_file
from think.callosum import CallosumConnection
from think.detect_created import detect_created
from think.detect_transcript import detect_transcript_json, detect_transcript_segment
from think.facets import get_facets
from think.importer_utils import save_import_file, write_import_metadata
from think.models import GEMINI_PRO, gemini_generate
from think.utils import PromptNotFoundError, day_path, load_prompt, setup_cli

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

logger = logging.getLogger(__name__)

MIN_THRESHOLD = 250
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
    """Get path relative to JOURNAL_PATH, or return as-is if not under JOURNAL_PATH."""
    journal_path = os.getenv("JOURNAL_PATH", "")
    if not journal_path:
        return path
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
    """
    imported_meta: dict[str, str] = {"id": import_id}
    if facet:
        imported_meta["facet"] = facet
    if setting:
        imported_meta["setting"] = setting

    # Build top-level metadata with imported info
    metadata: dict[str, object] = {"imported": imported_meta}

    # Add raw audio file reference if provided
    # Path is relative from timestamp directory (YYYYMMDD/HHMMSS/) to imports directory
    if raw_filename:
        metadata["raw"] = f"../../imports/{import_id}/{raw_filename}"

    # Write JSONL: metadata first, then entries
    jsonl_lines = [json.dumps(metadata)]
    jsonl_lines.extend(json.dumps(entry) for entry in entries)

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


def split_audio(path: str, out_dir: str, start: dt.datetime) -> list[str]:
    """Split audio from ``path`` into 5-minute FLAC segments in ``out_dir``.

    Returns:
        List of created file paths.
    """
    created_files = []
    # First, get the duration of the input file
    probe_cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        path,
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    duration = float(result.stdout.strip())

    # Calculate number of 5-minute (300-second) segments
    num_segments = int(duration // 300) + (1 if duration % 300 > 0 else 0)

    # Create each segment individually to ensure proper FLAC headers
    for idx in range(num_segments):
        segment_start = idx * 300
        ts = start + timedelta(seconds=segment_start)
        time_part = ts.strftime("%H%M%S")
        dest = os.path.join(out_dir, f"{time_part}_import_raw.flac")

        cmd = [
            "ffmpeg",
            "-i",
            path,
            "-ss",
            str(segment_start),
            "-t",
            "300",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "flac",
            "-y",  # Overwrite output files
            dest,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed for segment {idx}: {e.stderr}")
            raise
        logger.info(f"Added audio segment to journal: {dest}")
        created_files.append(dest)

    return created_files


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


def has_video_stream(path: str) -> bool:
    """Check if a media file contains video streams.

    Note: This is a placeholder for future video import support.
    Currently always returns False, so video processing is skipped.
    """
    # TODO: Implement video stream detection using ffprobe
    return False


def process_video(
    path: str, out_dir: str, start: dt.datetime, sample_s: float
) -> list[str]:
    """Process video frames and extract changes.

    Note: This is a placeholder for future video import support.
    Would extract keyframes at sample_s intervals and detect visual changes.

    Returns:
        List of created file paths.
    """
    # TODO: Implement video frame processing and change detection
    return []


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


def process_transcript(
    path: str,
    day_dir: str,
    base_dt: dt.datetime,
    *,
    import_id: str,
    facet: str | None = None,
    setting: str | None = None,
) -> list[str]:
    """Process a transcript file and write imported JSONL segments.

    Returns:
        List of created file paths.
    """
    created_files = []
    text = _read_transcript(path)
    segments = detect_transcript_segment(text)
    for idx, seg in enumerate(segments):
        json_data = detect_transcript_json(seg)
        if not json_data:
            continue
        ts = base_dt + timedelta(minutes=idx * 5)
        time_part = ts.strftime("%H%M%S")

        # Create segment directory with 5-minute (300 second) duration suffix
        segment_name = f"{time_part}_300"
        ts_dir = os.path.join(day_dir, segment_name)
        os.makedirs(ts_dir, exist_ok=True)
        json_path = os.path.join(ts_dir, "imported_audio.jsonl")

        # Ensure timestamps are absolute
        # Text transcripts might have relative timestamps (00:00:00, 00:01:23)
        # Convert them to absolute times based on the segment's base time
        absolute_entries = []
        for entry in json_data:
            entry_copy = entry.copy()
            if "start" in entry_copy:
                # Parse timestamp
                start_str = entry_copy["start"]
                h, m, s = map(int, start_str.split(":"))
                relative_seconds = h * 3600 + m * 60 + s

                # Convert to absolute timestamp based on this segment's time
                absolute_dt = ts + timedelta(seconds=relative_seconds)
                entry_copy["start"] = absolute_dt.strftime("%H:%M:%S")

            absolute_entries.append(entry_copy)

        _write_import_jsonl(
            json_path,
            absolute_entries,
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

    # Transcribe using Rev AI
    try:
        if entities:
            revai_json = transcribe_file(media_path, entities=entities)
        else:
            revai_json = transcribe_file(media_path)
    except Exception as e:
        logger.error(f"Failed to transcribe audio: {e}")
        raise

    # Convert to Sunstone format
    sunstone_transcript = convert_revai_to_sunstone(revai_json)

    if not sunstone_transcript:
        logger.warning("No transcript entries found")
        return created_files, revai_json

    # Group entries into 5-minute chunks
    chunks = []
    current_chunk = []
    chunk_start_time = None

    for entry in sunstone_transcript:
        # Parse the timestamp from the entry
        start_str = entry.get("start", "00:00:00")
        h, m, s = map(int, start_str.split(":"))
        entry_seconds = h * 3600 + m * 60 + s

        # Determine which 5-minute chunk this belongs to
        chunk_index = entry_seconds // 300  # 300 seconds = 5 minutes

        # If this is a new chunk, save the previous one
        if chunk_start_time is not None and chunk_index != chunk_start_time:
            if current_chunk:
                chunks.append((chunk_start_time, current_chunk))
            current_chunk = []
            chunk_start_time = chunk_index
        elif chunk_start_time is None:
            chunk_start_time = chunk_index

        current_chunk.append(entry)

    # Don't forget the last chunk
    if current_chunk:
        chunks.append((chunk_start_time, current_chunk))

    # Save each chunk as a separate JSONL file
    for chunk_index, chunk_entries in chunks:
        # Calculate timestamp for this chunk
        ts = base_dt + timedelta(minutes=chunk_index * 5)
        time_part = ts.strftime("%H%M%S")

        # Create segment directory with 5-minute (300 second) duration suffix
        segment_name = f"{time_part}_300"
        ts_dir = os.path.join(day_dir, segment_name)
        os.makedirs(ts_dir, exist_ok=True)
        json_path = os.path.join(ts_dir, "imported_audio.jsonl")

        # Convert relative timestamps to absolute timestamps
        # Rev AI returns timestamps relative to start of file (00:00:00, 00:00:06, etc.)
        # We need to convert them to absolute times based on base_dt
        absolute_entries = []
        for entry in chunk_entries:
            entry_copy = entry.copy()
            if "start" in entry_copy:
                # Parse relative timestamp from Rev AI
                h, m, s = map(int, entry_copy["start"].split(":"))
                relative_seconds = h * 3600 + m * 60 + s

                # Convert to absolute timestamp
                absolute_dt = base_dt + timedelta(seconds=relative_seconds)
                entry_copy["start"] = absolute_dt.strftime("%H:%M:%S")

            absolute_entries.append(entry_copy)

        # Save the chunk with absolute timestamps
        _write_import_jsonl(
            json_path,
            absolute_entries,
            import_id=import_id,
            raw_filename=os.path.basename(path),
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
    """Create a summary of all imported audio transcript files using Gemini Pro.

    Args:
        import_dir: Directory where the summary will be saved
        audio_json_files: List of paths to _imported_audio.json files
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
        logger.info(
            f"Creating summary with Gemini Pro for {len(all_transcripts)} transcript segments"
        )

        # Generate summary using Gemini
        response_text = gemini_generate(
            contents=user_message,
            model=GEMINI_PRO,
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
        logger.error(f"Failed to create summary with Gemini Pro: {e}")
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
    """Check if file path is already under {JOURNAL_PATH}/imports/."""
    journal = os.getenv("JOURNAL_PATH", "")
    if not journal:
        return False
    imports_dir = os.path.join(journal, "imports")
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
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    journal_root = Path(journal)
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

    print(f"\nAdd --facet <name>:")
    print(f"  think-importer {media_path} --timestamp {timestamp} --facet <name>")


def _print_setting_help(media_path: str, timestamp: str, facet: str) -> None:
    """Print setting requirement and suggested command."""
    print("\nSetting is required (describes the context of this recording).")
    print("Examples: 'Team standup meeting', 'Jer lunch with Joe', 'Conference talk'")
    print(f"\nAdd --setting <description>:")
    print(
        f"  think-importer {media_path} --timestamp {timestamp} "
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
        "--see", type=str2bool, default=True, help="Process video stream"
    )
    parser.add_argument(
        "--split", type=str2bool, default=False, help="Split audio stream into segments"
    )
    parser.add_argument(
        "--hear", type=str2bool, default=True, help="Transcribe audio using Rev AI"
    )
    parser.add_argument(
        "--summarize",
        type=str2bool,
        default=True,
        help="Create summary.md using Gemini Pro for audio transcripts",
    )
    parser.add_argument(
        "--see-sample",
        type=float,
        default=5.0,
        help="Video sampling interval in seconds",
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
            print(f"\nRun:")
            print(f"  think-importer {args.media} --timestamp {detected_timestamp}")
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
    logger.info(f"Using provided timestamp: {args.timestamp}")
    day_dir = str(day_path(base_dt.strftime("%Y%m%d")))

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
        day=base_dt.strftime("%Y%m%d"),
        facet=args.facet,
        setting=args.setting,
        options={
            "hear": args.hear,
            "split": args.split,
            "see": args.see,
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
                    "format": "imported_audio.json",
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
                        "format": "imported_audio.json",
                        "description": "Rev AI transcription chunks",
                        "files": created_files,
                        "count": len(created_files),
                        "transcription_service": "RevAI",
                    }
                )
            if args.split:
                created_files = split_audio(args.media, day_dir, base_dt)
                all_created_files.extend(created_files)
                processing_results["outputs"].append(
                    {
                        "type": "audio_segments",
                        "format": "import_raw.flac",
                        "description": "5-minute FLAC segments",
                        "files": created_files,
                        "count": len(created_files),
                    }
                )
            if args.see:
                if has_video_stream(args.media):
                    created_files = process_video(
                        args.media, day_dir, base_dt, args.see_sample
                    )
                    all_created_files.extend(created_files)
                    processing_results["outputs"].append(
                        {
                            "type": "video_frames",
                            "format": "import_1_diff.png",
                            "description": "Extracted video frames with changes",
                            "files": created_files,
                            "count": len(created_files),
                            "sampling_interval": args.see_sample,
                        }
                    )
                else:
                    logger.info(
                        f"No video stream found in {args.media}, skipping video processing"
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

        # Create Gemini Pro summary if requested and audio transcripts were created
        if args.summarize and audio_transcript_files:
            # Set stage for summarization
            _set_stage("summarizing")

            create_transcript_summary(
                import_dir=import_dir,
                audio_json_files=audio_transcript_files,
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
