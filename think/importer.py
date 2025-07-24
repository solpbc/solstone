import argparse
import datetime as dt
import json
import os
import re
import subprocess
import tempfile
from datetime import timedelta

import cv2
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from see.screen_compare import compare_images
from think.detect_created import detect_created
from think.detect_transcript import detect_transcript_json, detect_transcript_segment
from think.utils import setup_cli

try:
    from PyPDF2 import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

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


def split_audio(path: str, out_dir: str, start: dt.datetime) -> None:
    """Split audio from ``path`` into 60s FLAC segments in ``out_dir``."""
    with tempfile.TemporaryDirectory() as tmpdir:
        segment_template = os.path.join(tmpdir, "segment_%03d.flac")
        cmd = [
            "ffmpeg",
            "-i",
            path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "flac",
            "-f",
            "segment",
            "-segment_time",
            "60",
            segment_template,
        ]
        subprocess.run(cmd, check=True)

        segments = sorted(os.listdir(tmpdir))
        for idx, name in enumerate(segments):
            if not name.endswith(".flac"):
                continue
            ts = start + timedelta(seconds=idx * 60)
            time_part = ts.strftime("%H%M%S")
            dest = os.path.join(out_dir, f"{time_part}_import_raw.flac")
            os.replace(os.path.join(tmpdir, name), dest)


def process_video(path: str, out_dir: str, start: dt.datetime, sample_s: float) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        fps = 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return
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
                    img_path = os.path.join(out_dir, f"{time_part}_import_diff.png")

                    # Add box_2d to PNG metadata
                    pnginfo = PngInfo()
                    pnginfo.add_text("box_2d", json.dumps(largest["box_2d"]))

                    # Atomically save the image
                    with tempfile.NamedTemporaryFile(
                        dir=os.path.dirname(img_path), suffix=".pngtmp", delete=False
                    ) as tf:
                        img.save(tf, format="PNG", pnginfo=pnginfo)
                    os.replace(tf.name, img_path)
        prev_img = img
        frame_idx += interval
    cap.release()


def _read_transcript(path: str) -> str:
    """Return transcript text from a .txt or .pdf file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if ext == ".pdf":
        if PdfReader is None:
            raise RuntimeError("PyPDF2 required for PDF support")
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)
        return "\n".join(parts)
    raise ValueError("unsupported transcript format")


def process_transcript(path: str, journal: str) -> None:
    """Process a transcript file and write imported JSON segments."""
    info = detect_created(path)
    if not info:
        raise RuntimeError("unable to detect creation time")
    base_dt = dt.datetime.strptime(f"{info['day']}_{info['time']}", "%Y%m%d_%H%M%S")
    day_dir = os.path.join(journal, info["day"])
    os.makedirs(day_dir, exist_ok=True)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk a media file into the journal")
    parser.add_argument("media", help="Path to video or audio file")
    parser.add_argument(
        "timestamp", nargs="?", help="Timestamp YYYYMMDD_HHMMSS for journal entry"
    )
    parser.add_argument(
        "--see", type=str2bool, default=True, help="Process video stream"
    )
    parser.add_argument(
        "--hear", type=str2bool, default=True, help="Process audio stream"
    )
    parser.add_argument(
        "--see-sample",
        type=float,
        default=5.0,
        help="Video sampling interval in seconds",
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Detect creation timestamp using Gemini and exit",
    )
    args = setup_cli(parser)

    if args.detect:
        result = detect_created(args.media)
        print(json.dumps(result, indent=2))
        return

    journal = os.getenv("JOURNAL_PATH")

    ext = os.path.splitext(args.media)[1].lower()
    if ext in {".txt", ".pdf"}:
        process_transcript(args.media, journal)
        return

    if not args.timestamp or not TIME_RE.fullmatch(args.timestamp):
        raise SystemExit(
            "timestamp must be in YYYYMMDD_HHMMSS format or specify --detect"
        )
    base_dt = dt.datetime.strptime(args.timestamp, "%Y%m%d_%H%M%S")
    day_dir = os.path.join(journal, base_dt.strftime("%Y%m%d"))
    os.makedirs(day_dir, exist_ok=True)

    if args.hear:
        split_audio(args.media, day_dir, base_dt)
    if args.see:
        process_video(args.media, day_dir, base_dt, args.see_sample)


if __name__ == "__main__":
    main()
