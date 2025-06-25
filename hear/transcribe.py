import argparse
import datetime
import faulthandler
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from think.crumbs import CrumbBuilder

# Constants
MODEL = "gemini-2.5-flash"


class Transcriber:
    def __init__(
        self,
        journal_dir: Path,
        api_key: str,
        prompt_path: Path,
        entities_path: Path,
        voice_sample: Optional[Path] = None,
    ):
        self.journal_dir = journal_dir
        self.watch_dir: Optional[Path] = None
        self.client = genai.Client(api_key=api_key)
        self.prompt_path = prompt_path
        self.prompt_text = prompt_path.read_text().strip()
        self.entities_path = entities_path
        self.voice_sample_path = voice_sample or journal_dir / "voice_sample.flac"
        self.voice_sample_bytes: Optional[bytes] = None
        if self.voice_sample_path.is_file():
            try:
                self.voice_sample_bytes = self.voice_sample_path.read_bytes()
                logging.info(f"Loaded voice sample from {self.voice_sample_path}")
            except Exception as e:  # pragma: no cover - best effort
                logging.warning(f"Failed to load voice sample: {e}")
        self.processed: set[str] = set()
        self.observer: Optional[Observer] = None
        self.executor = ThreadPoolExecutor()
        self.attempts: dict[str, int] = {}

    def _process_once(self, flac_path: Path, json_path: Path) -> None:
        result = self.transcribe_file(flac_path)
        json_path.write_text(json.dumps({"text": result}, indent=2))
        logging.info(f"Transcribed {flac_path} -> {json_path}")
        crumb_builder = CrumbBuilder().add_file(self.prompt_path).add_file(self.entities_path)
        if self.voice_sample_bytes:
            crumb_builder = crumb_builder.add_file(self.voice_sample_path)
        crumb_builder = crumb_builder.add_file(flac_path).add_model(MODEL)
        crumb_path = crumb_builder.commit(str(json_path))
        logging.info(f"Crumb saved to {crumb_path}")

    def _process(self, flac_path: Path, json_path: Path) -> None:
        if json_path.exists() or json_path.name in self.processed:
            self.processed.add(json_path.name)
            return
        attempts = 0
        while attempts < 2:
            try:
                self._process_once(flac_path, json_path)
                break
            except Exception as e:
                attempts += 1
                if attempts < 2:
                    logging.warning(f"Retrying {flac_path} due to error: {e}")
                else:
                    logging.error(f"Failed to transcribe {flac_path}: {e}")
        self.processed.add(json_path.name)

    def transcribe_file(self, flac_path: Path) -> dict:
        logging.info(f"Processing {flac_path}")
        flac_bytes = flac_path.read_bytes()
        user_prompt = "Process the provided audio now and output your professional accurate transcription in the specified JSON format."
        entities_text = self.entities_path.read_text().strip()
        contents = [entities_text]
        if self.voice_sample_bytes:
            contents.append(
                "Here's a voice sample for Jeremie, if you are confident you hear him speaking when you are transcribing then tag his section with his name as the speaker."
            )
            contents.append(
                types.Part.from_bytes(data=self.voice_sample_bytes, mime_type="audio/flac")
            )
        contents.append(user_prompt)
        contents.append(types.Part.from_bytes(data=flac_bytes, mime_type="audio/flac"))

        response = self.client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=8192 * 2,
                response_mime_type="application/json",
                system_instruction=self.prompt_text,
            ),
        )
        result = json.loads(response.text)
        logging.info(f"Transcription result: {json.dumps(result, indent=2)}")
        return result

    def start(self):
        handler = PatternMatchingEventHandler(patterns=["*_audio.flac"], ignore_directories=True)

        def on_created(event):
            flac_path = Path(event.src_path)
            json_path = flac_path.with_suffix(".json")

            logging.info(f"New audio file detected: {flac_path}")
            self.executor.submit(self._process, flac_path, json_path)

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
                    self.processed.clear()
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
    load_dotenv()
    parser = argparse.ArgumentParser(description="Transcribe FLAC files using Gemini")
    parser.add_argument(
        "journal", type=Path, help="Journal directory containing daily audio folders"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=Path,
        default=Path(__file__).with_name("transcribe.txt"),
        help="Path to the system prompt text",
    )
    parser.add_argument(
        "-e",
        "--entities",
        type=Path,
        help="Path to master entities file (defaults to <journal>/entities.md)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Error: GOOGLE_API_KEY not found in environment.")

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    faulthandler.enable()

    ent_path = args.entities or args.journal / "entities.md"
    if not ent_path.is_file():
        parser.error(f"entities file not found: {ent_path}")

    transcriber = Transcriber(args.journal, api_key, args.prompt, ent_path)
    transcriber.start()


if __name__ == "__main__":
    main()
