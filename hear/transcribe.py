import argparse
import faulthandler
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


class Transcriber:
    def __init__(
        self,
        watch_dir: Path,
        api_key: str,
        prompt_path: Path,
        entities_path: Path | None = None,
        poll_interval: int = 5,
    ):
        self.watch_dir = watch_dir
        self.client = genai.Client(api_key=api_key)
        self.prompt_text = prompt_path.read_text().strip()
        self.entities_text = None
        if entities_path and entities_path.exists():
            self.entities_text = entities_path.read_text().strip()
        self.poll_interval = poll_interval
        self.processed: set[str] = set()

    def transcribe_file(self, flac_path: Path) -> dict:
        flac_bytes = flac_path.read_bytes()
        user_prompt = "Process the provided audio now and output your professional accurate transcription in the specified JSON format."
        if self.entities_text:
            user_prompt += (
                " Here's an incomplete list of entity names you might encounter, they can be useful to help disambiguate some terms: "
                + self.entities_text
            )
        response = self.client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=[
                user_prompt,
                types.Part.from_bytes(data=flac_bytes, mime_type="audio/flac"),
            ],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=8192,
                response_mime_type="application/json",
                system_instruction=self.prompt_text,
            ),
        )
        return json.loads(response.text)

    def scan(self) -> list[Path]:
        return [
            p
            for p in self.watch_dir.rglob("*_audio.flac")
            if p.stem + ".json" not in self.processed
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
                    self.processed.add(json_path.name)
                except Exception as e:
                    logging.error(f"Failed to transcribe {flac_path}: {e}")
            time.sleep(self.poll_interval)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Transcribe FLAC files using Gemini")
    parser.add_argument("watch_dir", type=Path, help="Directory containing FLAC files")
    parser.add_argument(
        "-p",
        "--prompt",
        type=Path,
        default=Path(__file__).with_name("gemini_mic.txt"),
        help="Path to the system prompt text",
    )
    parser.add_argument(
        "-e", "--entities", type=Path, default=None, help="Optional entity names file"
    )
    parser.add_argument("-i", "--interval", type=int, default=5, help="Polling interval in seconds")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Error: GOOGLE_API_KEY not found in environment.")

    logging.basicConfig(level=logging.INFO)
    faulthandler.enable()

    transcriber = Transcriber(args.watch_dir, api_key, args.prompt, args.entities, args.interval)
    transcriber.start()


if __name__ == "__main__":
    main()
