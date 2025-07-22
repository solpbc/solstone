import argparse
import datetime
import faulthandler
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from google import genai
from silero_vad import load_silero_vad
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from hear.audio_utils import SAMPLE_RATE, detect_speech, merge_streams, resample_audio
from hear.gemini import transcribe_segments
from think.crumbs import CrumbBuilder
from think.models import GEMINI_FLASH
from think.utils import day_log, setup_cli

# Constants
MODEL = GEMINI_FLASH
MIN_SPEECH_SECONDS = 1.0


class Transcriber:
    def __init__(
        self,
        journal_dir: Path,
        api_key: str,
        prompt_path: Path,
    ):
        self.journal_dir = journal_dir
        self.watch_dir: Optional[Path] = None
        self.client = genai.Client(api_key=api_key)
        self.prompt_path = prompt_path
        self.prompt_text = prompt_path.read_text().strip()
        self.entities_path = journal_dir / "entities.md"
        self.model = load_silero_vad()
        self.merged_stash = np.array([], dtype=np.float32)
        self.processing: list[Path] = []
        self.observer: Optional[Observer] = None
        self.executor = ThreadPoolExecutor()
        self.attempts: dict[str, int] = {}
        from df.enhance import init_df

        self._df_model, self._df_state, *_ = init_df(post_filter=True)
        self._df_sr = self._df_state.sr()

    def _trash_file(self, raw_path: Path) -> None:
        """Move the given file to a trash directory inside its day folder."""
        trash_dir = raw_path.parent / "trash"
        try:
            trash_dir.mkdir(exist_ok=True)
            raw_path.rename(trash_dir / raw_path.name)
        except Exception as e:
            logging.error(f"Failed to move {raw_path} to trash: {e}")

    def _move_to_heard(self, audio_path: Path) -> None:
        """Move processed ``audio_path`` into the day "heard" directory."""
        heard_dir = audio_path.parent / "heard"
        try:
            heard_dir.mkdir(exist_ok=True)
            audio_path.rename(heard_dir / audio_path.name)
            logging.info("Moved %s to %s", audio_path, heard_dir)
        except Exception as exc:  # pragma: no cover - filesystem errors
            logging.error("Failed to move %s to heard: %s", audio_path, exc)

    def _processed(self, raw_path: Path) -> None:
        """Move the given raw_path and precursors to heard directory, marking precursors as buffering."""
        # Mark and move precursor files to heard
        for precursor_path in self.processing:
            if not precursor_path.exists() or precursor_path == raw_path:
                continue
            json_path = self._get_json_path(precursor_path)
            try:
                json_path.write_text(json.dumps({"buffering": True}, indent=2))
                self._move_to_heard(precursor_path)
            except Exception as exc:  # pragma: no cover - filesystem errors
                logging.error(
                    "Failed to mark/move precursor %s: %s", precursor_path, exc
                )

        # Move the current raw_path to heard
        self._move_to_heard(raw_path)

        # Clear processing list
        self.processing.clear()

    def _process_raw(
        self, raw_path: Path, no_stash: bool = False
    ) -> List[Dict[str, object]] | None:
        try:
            data, sr = sf.read(raw_path, dtype="float32")

            mic_ranges: List[Tuple[float, float]] = []
            if data.ndim == 1:
                merged = resample_audio(
                    data, sr, SAMPLE_RATE, self._df_model, self._df_state, self._df_sr
                )
                logging.info(
                    f"Single channel audio detected in {raw_path} {sr}, no mic data."
                )
            else:
                logging.info(
                    f"Dual channel audio detected in {raw_path} {sr}, denoising mic channel."
                )
                mic_new = resample_audio(
                    data[:, 0],
                    sr,
                    SAMPLE_RATE,
                    self._df_model,
                    self._df_state,
                    self._df_sr,
                    denoise=True,
                )
                sys_new = resample_audio(
                    data[:, 1],
                    sr,
                    SAMPLE_RATE,
                    self._df_model,
                    self._df_state,
                    self._df_sr,
                )

                # when testing, save denoised mic audio for validation
                # sf.write("./mic_denoise.flac", mic_new, SAMPLE_RATE)

                merged, mic_ranges = merge_streams(sys_new, mic_new, sr)

            offset_seconds = len(self.merged_stash) / SAMPLE_RATE
            adjusted_ranges = [
                (s + offset_seconds, e + offset_seconds) for s, e in mic_ranges
            ]

            merged = np.concatenate((self.merged_stash, merged))
            segments, self.merged_stash = detect_speech(
                self.model, "mix", merged, adjusted_ranges, no_stash
            )

            if not segments:
                # Don't trash if there's unfinished speech in the stash
                if len(self.merged_stash) > 0:
                    logging.info(
                        f"No complete segments but {len(self.merged_stash) / SAMPLE_RATE:.1f}s in stash, keeping {raw_path}"
                    )
                    return None
                logging.info(f"No speech segments detected, moving {raw_path} to trash")
                self._trash_file(raw_path)
                return None

            total_seconds = sum(len(seg["data"]) / SAMPLE_RATE for seg in segments)
            if total_seconds < MIN_SPEECH_SECONDS and not no_stash:
                # Don't trash if there's unfinished speech in the stash
                if len(self.merged_stash) > 0:
                    logging.info(
                        "Total speech duration %.2fs < %.2fs but %.1fs in stash, keeping %s",
                        total_seconds,
                        MIN_SPEECH_SECONDS,
                        len(self.merged_stash) / SAMPLE_RATE,
                        raw_path,
                    )
                    return None
                logging.info(
                    "Total speech duration %.2fs < %.2fs, moving %s to trash",
                    total_seconds,
                    MIN_SPEECH_SECONDS,
                    raw_path,
                )
                self._trash_file(raw_path)
                return None

            day = raw_path.parent.name
            # Handle both _raw and _audio suffixes
            time_part = raw_path.stem.replace("_raw", "").replace("_audio", "")
            end_dt = datetime.datetime.strptime(f"{day}_{time_part}", "%Y%m%d_%H%M%S")
            file_duration = len(data) / SAMPLE_RATE
            base_dt = end_dt - datetime.timedelta(
                seconds=file_duration + offset_seconds
            )

            processed: List[Dict[str, object]] = []
            for seg in segments:
                start_dt = base_dt + datetime.timedelta(seconds=seg["offset"])
                start_str = start_dt.strftime("%H:%M:%S")
                audio_int16 = (np.clip(seg["data"], -1.0, 1.0) * 32767).astype(np.int16)
                buf = io.BytesIO()
                sf.write(buf, audio_int16, SAMPLE_RATE, format="FLAC")
                processed.append(
                    {
                        "start": start_str,
                        "source": "mic" if seg.get("mic") else "sys",
                        "bytes": buf.getvalue(),
                    }
                )

            logging.info(f"Processed {raw_path}: {len(processed)} segments")
            return processed
        except Exception as e:
            logging.error(f"Error processing {raw_path}: {e}", exc_info=True)
            return None

    def _get_json_path(self, audio_path: Path) -> Path:
        """Generate the corresponding JSON path for an audio file."""
        if audio_path.name.endswith("_raw.flac"):
            json_name = audio_path.name.replace("_raw.flac", "_audio.json")
        elif audio_path.name.endswith("_audio.flac"):
            json_name = audio_path.name.replace("_audio.flac", "_audio.json")
        else:
            # Fallback for other extensions
            json_name = audio_path.stem + "_audio.json"

        return audio_path.with_name(json_name)

    def _mark_buffering_precursors(self) -> None:
        """Mark any raw files awaiting transcription as buffering."""
        for raw_path in self.processing:
            if not raw_path.exists():
                continue
            json_path = self._get_json_path(raw_path)
            if not json_path.exists():
                try:
                    json_path.write_text(json.dumps({"buffering": True}, indent=2))
                except Exception as exc:  # pragma: no cover - filesystem errors
                    logging.error(
                        "Failed to write buffering json for %s: %s", json_path, exc
                    )
        self.processing.clear()

    def _transcribe_segments(
        self, raw_path: Path, segments: List[Dict[str, object]]
    ) -> bool:
        json_path = self._get_json_path(raw_path)
        attempts = 0
        while attempts < 2:
            try:
                entities_text = self.entities_path.read_text().strip()
                result = transcribe_segments(
                    self.client, MODEL, self.prompt_text, entities_text, segments
                )
                json_path.write_text(json.dumps({"text": result}, indent=2))
                logging.info(f"Transcribed {raw_path} -> {json_path}")

                crumb_builder = (
                    CrumbBuilder()
                    .add_file(self.prompt_path)
                    .add_file(self.entities_path)
                )
                crumb_builder = crumb_builder.add_file(raw_path).add_model(MODEL)
                crumb_path = crumb_builder.commit(str(json_path))
                logging.info(f"Crumb saved to {crumb_path}")
                return True
            except Exception as e:
                attempts += 1
                if attempts < 2:
                    logging.warning(f"Retrying {raw_path} due to error: {e}")
                else:
                    logging.error(f"Failed to transcribe {raw_path}: {e}")
                    return False
        # ran out of retries
        return False

    def _handle_raw(self, raw_path: Path, no_stash: bool = False) -> None:
        # skip files inside trash folder to avoid infinite loop
        if "trash" in raw_path.parts:
            logging.debug(f"Skipping file in trash: {raw_path}")
            return
        segments = self._process_raw(raw_path, no_stash)
        self.processing.append(raw_path)

        json_path = self._get_json_path(raw_path)
        if json_path.exists():
            self._processed(raw_path)
            return

        if segments is None:
            return

        success = self._transcribe_segments(raw_path, segments)
        if success:
            self._processed(raw_path)
        else:
            self.processing.clear()

    @staticmethod
    def scan_day(day_dir: Path) -> dict[str, list[str]]:
        """Return lists of raw, processed and repairable files within ``day_dir``.

        The ``processed`` list includes paths relative to ``day_dir`` so callers
        can easily open them. Processed audio lives in the ``heard/`` subfolder.
        """

        raw = sorted(p.name for p in day_dir.glob("*.flac"))

        heard_dir = day_dir / "heard"
        processed = (
            [f"heard/{p.name}" for p in sorted(heard_dir.glob("*.flac"))]
            if heard_dir.is_dir()
            else []
        )

        return {"raw": raw, "processed": processed, "repairable": raw.copy()}

    def repair_day(self, date_str: str, files: list[str], dry_run: bool = False) -> int:
        """Process ``files`` belonging to ``date_str`` and return the count."""
        day_dir = self.journal_dir / date_str
        if not day_dir.exists():
            logging.error(f"Day directory {day_dir} does not exist")
            return 0

        logging.info(f"Repairing day {date_str} in {day_dir}")

        if dry_run:
            return len(files)

        success = 0

        # Sort by HHMMSS for processing order
        files_sorted = sorted(files, key=lambda n: n.split("_")[0])
        for i, name in enumerate(files_sorted):
            audio_path = day_dir / name
            if not audio_path.exists():
                logging.warning(f"Skipping missing audio file {audio_path}")
                continue
            logging.info(f"Processing audio file: {audio_path}")
            json_path = self._get_json_path(audio_path)
            before = json_path.exists()
            if before:
                self._move_to_heard(audio_path)
                logging.info(f"Skipping already processed file: {json_path}")
                success += 1
                continue
            # Don't stash on the last file
            is_last_file = i == len(files_sorted) - 1
            self._handle_raw(audio_path, no_stash=is_last_file)
            if json_path.exists() and not before:
                success += 1

        return success

    def start(self):
        handler = PatternMatchingEventHandler(
            patterns=["*.flac"],
            ignore_directories=True,
            ignore_patterns=["*/trash/*", "*/heard/*"],
        )

        def on_created(event):
            raw_path = Path(event.src_path)
            logging.info(f"New raw audio file detected: {raw_path}")
            self.executor.submit(self._handle_raw, raw_path)

        handler.on_created = on_created

        self.observer = None
        current_day: Optional[str] = None
        try:
            while True:
                today_str = datetime.datetime.now().strftime("%Y%m%d")
                day_dir = self.journal_dir / today_str
                if day_dir.exists() and (current_day != today_str):
                    if self.observer:
                        self.observer.stop()
                        self.observer.join()
                    self.observer = Observer()
                    self.observer.schedule(handler, str(day_dir), recursive=True)
                    self.observer.start()
                    self.watch_dir = day_dir
                    current_day = today_str
                    logging.info(f"Watching {day_dir}")
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            self.executor.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser(description="Transcribe FLAC files using Gemini")
    parser.add_argument(
        "-p",
        "--prompt",
        type=Path,
        default=Path(__file__).with_name("transcribe.txt"),
        help="Path to the system prompt text",
    )
    parser.add_argument(
        "--repair",
        type=str,
        help="Repair mode: process incomplete files for specified day (YYYYMMDD format)",
    )
    args = setup_cli(parser)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Error: GOOGLE_API_KEY not found in environment.")

    faulthandler.enable()

    journal = Path(os.getenv("JOURNAL_PATH", ""))
    if not journal.is_dir():
        parser.error("JOURNAL_PATH not set or invalid")
    ent_path = journal / "entities.md"
    if not ent_path.is_file():
        parser.error(f"entities file not found: {ent_path}")

    transcriber = Transcriber(journal, api_key, args.prompt)

    if args.repair:
        try:
            datetime.datetime.strptime(args.repair, "%Y%m%d")
        except ValueError:
            parser.error(f"Invalid date format: {args.repair}. Use YYYYMMDD format.")
        info = Transcriber.scan_day(journal / args.repair)
        repaired = transcriber.repair_day(args.repair, info["repairable"])
        failed = len(info["repairable"]) - repaired
        msg = f"hear-transcribe repaired {repaired}"
        if failed:
            msg += f" failed {failed}"
        day_log(args.repair, msg)
    else:
        transcriber.start()


if __name__ == "__main__":
    main()
