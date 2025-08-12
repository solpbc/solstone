import argparse
import datetime as dt
import json
import logging
import os
import re
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path

import cv2
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from hear.revai import convert_revai_to_sunstone, transcribe_file
from see.screen_compare import compare_images
from think.detect_created import detect_created
from think.detect_transcript import detect_transcript_json, detect_transcript_segment
from think.utils import setup_cli

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

logger = logging.getLogger(__name__)

MIN_THRESHOLD = 250
TIME_RE = re.compile(r"\d{8}_\d{6}")


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


def has_video_stream(path: str) -> bool:
    """Check if a media file contains video streams."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False

    # Check if we can actually read a frame
    has_video = False
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 0:
        ret, frame = cap.read()
        has_video = ret and frame is not None

    cap.release()
    return has_video


def process_video(path: str, out_dir: str, start: dt.datetime, sample_s: float) -> list[str]:
    """Process video frames and extract changes.

    Returns:
        List of created file paths.
    """
    created_files = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return created_files
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        fps = 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return created_files
    interval = max(1, int(fps * sample_s))
    prev_img = None
    frame_idx = 0
    while frame_idx < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if prev_img is not None:
            boxes = compare_images(prev_img, img)
            if boxes:
                # fmt: off
                largest = max(
                    boxes,
                    key=lambda b: (b["box_2d"][3] - b["box_2d"][1]) * (b["box_2d"][2] - b["box_2d"][0]),
                )
                # fmt: on
                width = largest["box_2d"][3] - largest["box_2d"][1]
                height = largest["box_2d"][2] - largest["box_2d"][0]
                if width > MIN_THRESHOLD and height > MIN_THRESHOLD:
                    ts = start + timedelta(seconds=frame_idx / fps)
                    time_part = ts.strftime("%H%M%S")
                    img_path = os.path.join(out_dir, f"{time_part}_import_1_diff.png")

                    # Add box_2d to PNG metadata
                    pnginfo = PngInfo()
                    pnginfo.add_text("box_2d", json.dumps(largest["box_2d"]))

                    # Atomically save the image
                    with tempfile.NamedTemporaryFile(
                        dir=os.path.dirname(img_path), suffix=".pngtmp", delete=False
                    ) as tf:
                        img.save(tf, format="PNG", pnginfo=pnginfo)
                    os.replace(tf.name, img_path)
                    logger.info(f"Added video frame to journal: {img_path}")
                    created_files.append(img_path)
        prev_img = img
        frame_idx += interval
    cap.release()
    return created_files


def _read_transcript(path: str) -> str:
    """Return transcript text from a .txt or .pdf file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
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


def process_transcript(path: str, day_dir: str, base_dt: dt.datetime) -> list[str]:
    """Process a transcript file and write imported JSON segments.

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
            day_dir, f"{ts.strftime('%H%M%S')}_imported_audio.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Added transcript segment to journal: {json_path}")
        created_files.append(json_path)

    return created_files


def audio_transcribe(path: str, day_dir: str, base_dt: dt.datetime) -> tuple[list[str], dict]:
    """Transcribe audio using Rev AI and save 5-minute chunks as imported JSON.

    Returns:
        Tuple of (list of created file paths, raw RevAI JSON result)
    """
    logger.info(f"Transcribing audio file: {path}")
    media_path = Path(path)
    created_files = []

    # Transcribe using Rev AI
    try:
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

    # Save each chunk as a separate JSON file
    for chunk_index, chunk_entries in chunks:
        # Calculate timestamp for this chunk
        ts = base_dt + timedelta(minutes=chunk_index * 5)
        json_path = os.path.join(
            day_dir, f"{ts.strftime('%H%M%S')}_imported_audio.json"
        )

        # Save the chunk
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunk_entries, f, indent=2)
        logger.info(f"Added transcript chunk to journal: {json_path}")
        created_files.append(json_path)

    return created_files, revai_json


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
        "--see-sample",
        type=float,
        default=5.0,
        help="Video sampling interval in seconds",
    )
    args, extra = setup_cli(parser, parse_known=True)
    if extra and not args.timestamp:
        args.timestamp = extra[0]

    journal = os.getenv("JOURNAL_PATH")

    # If no timestamp provided, detect it and show instruction
    if not args.timestamp:
        # Pass the original filename for better detection
        result = detect_created(args.media, original_filename=os.path.basename(args.media))
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
    day_dir = os.path.join(journal, base_dt.strftime("%Y%m%d"))
    os.makedirs(day_dir, exist_ok=True)

    # Track all created files and processing metadata
    all_created_files = []
    revai_json_data = None
    processing_results = {
        "processed_timestamp": args.timestamp,
        "target_day": base_dt.strftime("%Y%m%d"),
        "target_day_path": day_dir,
        "input_file": args.media,
        "processing_started": dt.datetime.now().isoformat(),
        "outputs": []
    }

    ext = os.path.splitext(args.media)[1].lower()
    if ext in {".txt", ".pdf"}:
        created_files = process_transcript(args.media, day_dir, base_dt)
        all_created_files.extend(created_files)
        processing_results["outputs"].append({
            "type": "transcript",
            "format": "imported_audio.json",
            "description": "Transcript segments",
            "files": created_files,
            "count": len(created_files)
        })
    else:
        if args.hear:
            created_files, revai_json_data = audio_transcribe(args.media, day_dir, base_dt)
            all_created_files.extend(created_files)
            processing_results["outputs"].append({
                "type": "audio_transcript",
                "format": "imported_audio.json",
                "description": "Rev AI transcription chunks",
                "files": created_files,
                "count": len(created_files),
                "transcription_service": "RevAI"
            })
        if args.split:
            created_files = split_audio(args.media, day_dir, base_dt)
            all_created_files.extend(created_files)
            processing_results["outputs"].append({
                "type": "audio_segments",
                "format": "import_raw.flac",
                "description": "60-second FLAC segments",
                "files": created_files,
                "count": len(created_files)
            })
        if args.see:
            if has_video_stream(args.media):
                created_files = process_video(args.media, day_dir, base_dt, args.see_sample)
                all_created_files.extend(created_files)
                processing_results["outputs"].append({
                    "type": "video_frames",
                    "format": "import_1_diff.png",
                    "description": "Extracted video frames with changes",
                    "files": created_files,
                    "count": len(created_files),
                    "sampling_interval": args.see_sample
                })
            else:
                logger.info(f"No video stream found in {args.media}, skipping video processing")

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
            metadata["processing_completed"] = processing_results["processing_completed"]
            metadata["total_files_created"] = processing_results["total_files_created"]
            metadata["imported_json_path"] = str(imported_path)
            if revai_json_data:
                metadata["revai_json_path"] = str(revai_path)
            with open(import_metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Updated import metadata: {import_metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to update import metadata: {e}")


if __name__ == "__main__":
    main()
