import argparse
import datetime as dt
import json
import logging
import os
import re
import string
import subprocess
import tempfile
import unicodedata
from datetime import timedelta
from pathlib import Path

import cv2
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from observe.revai import convert_revai_to_sunstone, transcribe_file
from think.detect_created import detect_created
from think.detect_transcript import detect_transcript_json, detect_transcript_segment
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


def _build_import_payload(
    entries: list[dict],
    *,
    import_id: str,
    domain: str | None = None,
    setting: str | None = None,
) -> dict:
    """Return structured payload for imported transcript chunks."""

    imported_meta: dict[str, str] = {"id": import_id}
    if domain:
        imported_meta["domain"] = domain
    if setting:
        imported_meta["setting"] = setting

    return {"imported": imported_meta, "entries": entries}


def _write_import_jsonl(
    file_path: str,
    entries: list[dict],
    *,
    import_id: str,
    raw_filename: str | None = None,
    domain: str | None = None,
    setting: str | None = None,
) -> None:
    """Write imported transcript entries in JSONL format.

    First line contains imported metadata, subsequent lines contain entries.
    """
    imported_meta: dict[str, str] = {"id": import_id}
    if domain:
        imported_meta["domain"] = domain
    if setting:
        imported_meta["setting"] = setting

    # Build top-level metadata with imported info
    metadata: dict[str, object] = {"imported": imported_meta}

    # Add raw audio file reference if provided
    # Path is relative from day directory to imports directory
    if raw_filename:
        metadata["raw"] = f"../imports/{import_id}/{raw_filename}"

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
    """Split audio from ``path`` into 60s FLAC segments in ``out_dir``.

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

    # Calculate number of 60-second segments
    num_segments = int(duration // 60) + (1 if duration % 60 > 0 else 0)

    # Create each segment individually to ensure proper FLAC headers
    for idx in range(num_segments):
        segment_start = idx * 60
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
            "60",
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
        subprocess.run(cmd, check=True)
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
    """Check if a media file contains video streams."""
    # TODO: Implement video stream detection
    return False


def process_video(
    path: str, out_dir: str, start: dt.datetime, sample_s: float
) -> list[str]:
    """Process video frames and extract changes.

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
    domain: str | None = None,
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
        json_path = os.path.join(
            day_dir, f"{ts.strftime('%H%M%S')}_imported_audio.jsonl"
        )
        _write_import_jsonl(
            json_path,
            json_data,
            import_id=import_id,
            raw_filename=os.path.basename(path),
            domain=domain,
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
    domain: str | None = None,
    setting: str | None = None,
) -> tuple[list[str], dict]:
    """Transcribe audio using Rev AI and save 5-minute chunks as imported JSONL.

    Args:
        path: Path to audio file
        day_dir: Directory to save chunks to
        base_dt: Base datetime for timestamps
        domain: Optional domain name to extract entities from
        setting: Optional description of the setting to store with metadata

    Returns:
        Tuple of (list of created file paths, raw RevAI JSON result)
    """
    logger.info(f"Transcribing audio file: {path}")
    media_path = Path(path)
    created_files = []

    # Get domain entities if domain is specified
    entities = None
    if domain:
        try:
            from think.utils import load_entity_names

            # Load entity names from domain-specific entities.md (spoken mode for short forms)
            entity_names = load_entity_names(domain=domain, spoken=True)
            if entity_names:
                # entity_names is already a list in spoken mode
                entities = _sanitize_entities(entity_names)
                if entities:
                    logger.info(
                        f"Using {len(entities)} entities from domain '{domain}' for transcription"
                    )
                else:
                    logger.info(
                        f"Domain '{domain}' entities removed after sanitization"
                    )
            else:
                logger.info(f"No entities found for domain '{domain}'")
        except FileNotFoundError:
            logger.info(f"Domain '{domain}' has no entities.md file")
        except Exception as e:
            logger.warning(f"Failed to load domain entities: {e}")

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
        json_path = os.path.join(
            day_dir, f"{ts.strftime('%H%M%S')}_imported_audio.jsonl"
        )

        # Save the chunk
        _write_import_jsonl(
            json_path,
            chunk_entries,
            import_id=import_id,
            raw_filename=os.path.basename(path),
            domain=domain,
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
) -> None:
    """Create a summary of all imported audio transcript files using Gemini Pro.

    Args:
        import_dir: Directory where the summary will be saved
        audio_json_files: List of paths to _imported_audio.json files
        input_filename: Original media filename for context
        timestamp: Processing timestamp for context
        setting: Optional description of the setting to include in metadata
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
            with open(json_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
                if isinstance(transcript_data, dict) and "entries" in transcript_data:
                    entries = transcript_data.get("entries") or []
                else:
                    entries = (
                        transcript_data if isinstance(transcript_data, list) else []
                    )

                all_transcripts.append(
                    {"file": os.path.basename(json_path), "content": entries}
                )
        except Exception as e:
            logger.warning(f"Failed to read {json_path}: {e}")

    if not all_transcripts:
        logger.warning("No transcripts could be read for summarization")
        return

    # Load the prompt from importer.txt
    try:
        importer_prompt = load_prompt("importer", base_dir=Path(__file__).parent)
    except PromptNotFoundError as exc:
        logger.error(f"Failed to load importer prompt: {exc}")
        return
    importer_prompt_template = importer_prompt.text

    # Add the context metadata to the prompt
    metadata_lines = [
        "\n\n## Metadata for this summary:",
        f"- Original file: {input_filename}",
        f"- Recording timestamp: {timestamp}",
        f"- Number of transcript segments: {len(all_transcripts)}",
        f"- Total transcript entries: {sum(len(t['content']) for t in all_transcripts)}",
    ]
    if setting:
        metadata_lines.append(f"- Setting: {setting}")

    importer_prompt = importer_prompt_template + "\n".join(metadata_lines)

    # Format the transcript content for the user message
    user_message_parts = []

    for transcript_info in all_transcripts:
        entries = transcript_info["content"]
        if isinstance(entries, list):
            user_message_parts.append(
                f"\n## Transcript Segment: {transcript_info['file']}\n"
            )
            user_message_parts.append(json.dumps(entries, indent=2))

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
        total_entries = sum(len(t["content"]) for t in all_transcripts)
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


def main() -> None:
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
        "--domain",
        type=str,
        default=None,
        help="Domain name to use for entity extraction",
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

    journal = os.getenv("JOURNAL_PATH")

    # If no timestamp provided, detect it and show instruction
    if not args.timestamp:
        # Pass the original filename for better detection
        result = detect_created(
            args.media, original_filename=os.path.basename(args.media)
        )
        if result and result.get("day") and result.get("time"):
            detected_timestamp = f"{result['day']}_{result['time']}"
            print(f"Detected: --timestamp {detected_timestamp}")
            return
        else:
            raise SystemExit(
                "Could not detect timestamp. Please provide --timestamp YYYYMMDD_HHMMSS"
            )

    if not TIME_RE.fullmatch(args.timestamp):
        raise SystemExit("timestamp must be in YYYYMMDD_HHMMSS format")

    base_dt = dt.datetime.strptime(args.timestamp, "%Y%m%d_%H%M%S")
    logger.info(f"Using provided timestamp: {args.timestamp}")
    day_dir = str(day_path(base_dt.strftime("%Y%m%d")))

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
        "domain": args.domain,
        "setting": args.setting,
        "outputs": [],
    }

    ext = os.path.splitext(args.media)[1].lower()
    if ext in {".txt", ".md", ".pdf"}:
        created_files = process_transcript(
            args.media,
            day_dir,
            base_dt,
            import_id=args.timestamp,
            domain=args.domain,
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
            created_files, revai_json_data = audio_transcribe(
                args.media,
                day_dir,
                base_dt,
                import_id=args.timestamp,
                domain=args.domain,
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
                    "description": "60-second FLAC segments",
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
            metadata["total_files_created"] = processing_results["total_files_created"]
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
        create_transcript_summary(
            import_dir=import_dir,
            audio_json_files=audio_transcript_files,
            input_filename=os.path.basename(args.media),
            timestamp=args.timestamp,
            setting=args.setting,
        )


if __name__ == "__main__":
    main()
