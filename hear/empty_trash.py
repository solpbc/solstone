import argparse
import logging
import os
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path

import soundfile as sf
from silero_vad import load_silero_vad

from hear.audio_utils import SAMPLE_RATE, detect_speech
from think.utils import DATE_RE, day_log, setup_cli


def _speech_in_file(path: Path, vad_model) -> bool:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE and sr > 0:
        from scipy.signal import resample

        target_len = int(len(data) * SAMPLE_RATE / sr)
        data = resample(data, target_len)
    segments, _ = detect_speech(vad_model, path.name, data, None, True)
    return bool(segments)


def _process_file_worker(file_path_str: str) -> tuple[str, bool, str | None]:
    """Worker function for processing files in parallel processes."""
    path = Path(file_path_str)
    try:
        # Each worker loads its own model to avoid thread safety issues
        model = load_silero_vad()
        has_speech = _speech_in_file(path, model)
        return (file_path_str, has_speech, None)
    except Exception as exc:
        return (file_path_str, False, str(exc))


def _process_day_parallel(day_dir: Path, num_workers: int, args) -> tuple[int, int]:
    """Process a single day's trash directory using parallel workers."""
    trash_dir = day_dir / "trash"
    if not trash_dir.is_dir():
        return 0, 0

    file_paths = [
        str(path) for path in sorted(trash_dir.iterdir()) if not path.is_dir()
    ]
    if not file_paths:
        return 0, 0

    removed = 0
    restored = 0
    gio = shutil.which("gio")

    with Pool(processes=num_workers) as pool:
        results = pool.map(_process_file_worker, file_paths)

    for file_path_str, has_speech, error in results:
        path = Path(file_path_str)
        if error:
            logging.error("Failed to inspect %s: %s", path, error)
            continue

        if has_speech:
            if args.verbose:
                print(f"restore {path}")
            if not args.dry_run:
                path.rename(day_dir / path.name)
            restored += 1
        else:
            if args.verbose:
                print(f"remove {path}")
            if not args.dry_run:
                if gio:
                    subprocess.run([gio, "trash", str(path)], check=False)
                else:
                    path.unlink(missing_ok=True)
            removed += 1

    return removed, restored


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Permanently delete trashed audio files"
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Do not modify the filesystem",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel workers for speech detection",
    )
    args = setup_cli(parser)

    journal = Path(os.getenv("JOURNAL_PATH", ""))

    # Only load model for single-threaded processing
    model = load_silero_vad() if args.jobs == 1 else None

    total_removed = 0
    total_restored = 0

    for day_dir in sorted(p for p in journal.iterdir() if DATE_RE.fullmatch(p.name)):
        if args.jobs == 1:
            # Single-threaded processing (original logic)
            trash_dir = day_dir / "trash"
            if not trash_dir.is_dir():
                continue
            removed = 0
            restored = 0
            gio = shutil.which("gio")
            for path in sorted(trash_dir.iterdir()):
                if path.is_dir():
                    continue
                try:
                    has_speech = _speech_in_file(path, model)
                except Exception as exc:
                    logging.error("Failed to inspect %s: %s", path, exc)
                    continue
                if has_speech:
                    if args.verbose:
                        print(f"restore {path}")
                    if not args.dry_run:
                        path.rename(day_dir / path.name)
                    restored += 1
                else:
                    if args.verbose:
                        print(f"remove {path}")
                    if not args.dry_run:
                        if gio:
                            subprocess.run([gio, "trash", str(path)], check=False)
                        else:
                            path.unlink(missing_ok=True)
                    removed += 1
        else:
            # Multi-threaded processing
            removed, restored = _process_day_parallel(day_dir, args.jobs, args)

        if removed or restored:
            total_removed += removed
            total_restored += restored
            if args.verbose or True:
                print(f"{day_dir.name}: removed {removed}, restored {restored}")
            day_log(day_dir.name, f"empty-trash {removed} removed {restored} restored")

    print(f"Total removed {total_removed}, restored {total_restored}")


if __name__ == "__main__":
    main()
