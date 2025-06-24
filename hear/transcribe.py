import argparse
import faulthandler
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from think.crumbs import CrumbBuilder

# Constants
MODEL = "gemini-2.5-flash"


class Transcriber:
    def __init__(
        self,
        journal: Path,
        api_key: str,
        prompt_path: Path,
        entities_path: Path,
        poll_interval: int = 5,
    ):
        self.watch_dir = journal
        self.client = genai.Client(api_key=api_key)
        self.prompt_path = prompt_path
        self.prompt_text = prompt_path.read_text().strip()
        self.entities_path = entities_path
        self.poll_interval = poll_interval
        self.processed: set[str] = set()

    def transcribe_file(self, flac_path: Path) -> dict:
        flac_bytes = flac_path.read_bytes()
        user_prompt = "Process the provided audio now and output your professional accurate transcription in the specified JSON format."
        entities_text = self.entities_path.read_text().strip()
        contents = [
            entities_text,
            user_prompt,
            types.Part.from_bytes(data=flac_bytes, mime_type="audio/flac"),
        ]

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
        return json.loads(response.text)

    def scan(self) -> list[Path]:
        today = datetime.now().strftime("%Y%m%d")
        today_dir = self.watch_dir / today
        if not today_dir.exists():
            logging.info(f"No directory for today: {today_dir}")
            return []
        return [
            p for p in today_dir.rglob("*_audio.flac") if p.stem + ".json" not in self.processed
        ]

    def start(self):
        while True:
            for flac_path in self.scan():
                json_path = flac_path.with_suffix(".json")
                if json_path.exists():
                    self.processed.add(json_path.name)
                    continue
                try:
                    result = self.transcribe_file(flac_path)
                    json_path.write_text(json.dumps({"text": result}))
                    logging.info(f"Transcribed {flac_path} -> {json_path}")
                    crumb_builder = (
                        CrumbBuilder()
                        .add_file(self.prompt_path)
                        .add_file(self.entities_path)
                        .add_file(flac_path)
                        .add_model(MODEL)
                    )
                    crumb_path = crumb_builder.commit(str(json_path))
                    logging.info(f"Crumb saved to {crumb_path}")
                    self.processed.add(json_path.name)
                except Exception as e:
                    logging.error(f"Failed to transcribe {flac_path}: {e}")
            time.sleep(self.poll_interval)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Transcribe FLAC files using Gemini")
    parser.add_argument("journal", type=Path, help="Journal directory containing FLAC files")
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
        help="Path to master entities file (defaults to <watch_dir>/entities.md)",
    )
    parser.add_argument("-i", "--interval", type=int, default=5, help="Polling interval in seconds")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Error: GOOGLE_API_KEY not found in environment.")

    logging.basicConfig(level=logging.INFO)
    faulthandler.enable()

    ent_path = args.entities or args.journal / "entities.md"
    if not ent_path.is_file():
        parser.error(f"entities file not found: {ent_path}")

    transcriber = Transcriber(args.journal, api_key, args.prompt, ent_path, args.interval)
    transcriber.start()


if __name__ == "__main__":
    main()
