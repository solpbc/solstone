import argparse
import logging
import os
import shutil
import subprocess
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Permanently delete trashed audio files")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Do not modify the filesystem")
    args = setup_cli(parser)

    journal = Path(os.getenv("JOURNAL_PATH", ""))
    model = load_silero_vad()
    gio = shutil.which("gio")

    total_removed = 0
    total_restored = 0

    for day_dir in sorted(p for p in journal.iterdir() if DATE_RE.fullmatch(p.name)):
        trash_dir = day_dir / "trash"
        if not trash_dir.is_dir():
            continue
        removed = 0
        restored = 0
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
        if removed or restored:
            total_removed += removed
            total_restored += restored
            if args.verbose or True:
                print(f"{day_dir.name}: removed {removed}, restored {restored}")
            day_log(day_dir.name, f"empty-trash {removed} removed {restored} restored")

    print(f"Total removed {total_removed}, restored {total_restored}")


if __name__ == "__main__":
    main()
