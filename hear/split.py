import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import soundfile as sf

from think.utils import day_path, parse_time_range, setup_cli

TOLERANCE_SEC = 1.0


def split_file(path: Path, cleanup: bool) -> None:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim != 2 or data.shape[1] != 2:
        raise RuntimeError(f"{path} is not stereo")
    base = path.stem.replace("_audio", "")
    mic_path = path.with_name(f"{base}_mic_audio.flac")
    sys_path = path.with_name(f"{base}_system_audio.flac")
    if mic_path.exists() or sys_path.exists():
        logging.info("Skipping %s, split files exist", path.name)
        if cleanup and mic_path.exists() and sys_path.exists():
            path.unlink(missing_ok=True)
        return
    sf.write(mic_path, data[:, 0], sr, format="FLAC")
    sf.write(sys_path, data[:, 1], sr, format="FLAC")
    logging.info("Split %s -> %s, %s", path.name, mic_path.name, sys_path.name)
    if cleanup and mic_path.exists() and sys_path.exists():
        path.unlink(missing_ok=True)


def find_files(
    day_dir: Path, length_min: float, start: str | None = None, end: str | None = None
) -> list[Path]:
    """Return audio files in ``day_dir`` matching ``length_min`` and optional range."""

    expected = length_min * 60
    files: list[Path] = []
    for audio in sorted(day_dir.glob("*_audio.flac")):
        try:
            info = sf.info(audio)
        except Exception as exc:  # pragma: no cover - invalid file
            logging.error("Failed to read %s: %s", audio, exc)
            continue

        duration = info.frames / info.samplerate if info.samplerate else 0
        if abs(duration - expected) > TOLERANCE_SEC:
            continue

        if start and end:
            ts = audio.stem.split("_")[0]
            if not (start <= ts < end):
                continue

        files.append(audio)

    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Split stereo audio files")
    parser.add_argument(
        "range",
        nargs="?",
        help="Time range like 'July 19 3p-4p'. Overrides --day/--start/--length",
    )
    parser.add_argument("--day", help="Day YYYYMMDD")
    parser.add_argument("--start", metavar="HHMMSS", help="Start time")
    parser.add_argument("--length", type=float, help="Length in minutes")
    parser.add_argument(
        "--cleanup", action="store_true", help="Remove source after split"
    )
    parser.add_argument("--dry", action="store_true", help="Dry run")
    args = setup_cli(parser)

    if args.range:
        if args.day or args.start or args.length:
            parser.error("range and --day/--start/--length are mutually exclusive")
        parsed = parse_time_range(args.range)
        if not parsed:
            parser.error(f"Could not parse range: {args.range}")
        day, start, end = parsed
        length = (
            datetime.strptime(end, "%H%M%S") - datetime.strptime(start, "%H%M%S")
        ).seconds / 60
    else:
        if not (args.day and args.start and args.length is not None):
            parser.error("--day, --start and --length are required without range")
        day = args.day
        start = args.start
        length = args.length
        end_dt = datetime.strptime(start, "%H%M%S") + timedelta(minutes=length)
        end = end_dt.strftime("%H%M%S")

    day_dir = Path(day_path(day))
    files = find_files(day_dir, length, start, end)

    if args.dry:
        print(f"{len(files)} files to split")
        return

    for path in files:
        split_file(path, args.cleanup)


if __name__ == "__main__":
    main()
