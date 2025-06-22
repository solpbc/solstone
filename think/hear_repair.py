import argparse
import json
import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from think.crumbs import CrumbBuilder

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "hear", "transcribe.txt")


def find_missing(day_dir):
    if not os.path.isdir(day_dir):
        raise FileNotFoundError(f"Day directory not found: {day_dir}")
    missing = []
    for name in sorted(os.listdir(day_dir)):
        if name.endswith(".ogg") or name.endswith(".flac"):
            if name.endswith(".ogg"):
                base = name[:-4]  # Remove ".ogg"
            else:  # name.endswith(".flac")
                base = name[:-5]  # Remove ".flac"
            json_path = os.path.join(day_dir, base + ".json")
            if not os.path.exists(json_path):
                audio_path = os.path.join(day_dir, name)
                missing.append((audio_path, json_path))
    return missing


def transcribe_file(client, prompt_text, audio_path, model="gemini-2.5-flash"):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # Determine MIME type based on file extension
    if audio_path.endswith(".ogg"):
        mime_type = "audio/ogg"
        format_name = "OGG"
    elif audio_path.endswith(".flac"):
        mime_type = "audio/flac"
        format_name = "FLAC"
    else:
        raise ValueError(f"Unsupported audio format: {audio_path}")

    size_mb = len(audio_bytes) / (1024 * 1024)
    logging.info(f"Transcribing {audio_path} ({format_name}, {size_mb:.2f}MB)")
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                "Process the provided audio now and output your professional accurate transcription in the specified JSON format.",
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=8192,
                response_mime_type="application/json",
                system_instruction=prompt_text,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return ""


def process_files(files, delay, client, prompt_text, model="gemini-2.5-flash"):
    for audio_path, json_path in files:
        result = transcribe_file(client, prompt_text, audio_path, model)
        if result:
            with open(json_path, "w") as f:
                json.dump(result, f)
            print(f"Saved {json_path}")
            crumb_builder = (
                CrumbBuilder().add_file(PROMPT_PATH).add_file(audio_path).add_model(model)
            )
            crumb_path = crumb_builder.commit(json_path)
            print(f"Crumb saved to: {crumb_path}")
        else:
            print(f"Gemini returned no result for {audio_path}")
        time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Repair missing Gemini JSON for audio files")
    parser.add_argument("day_dir", help="Day directory path containing audio files")
    parser.add_argument(
        "--wait", type=float, default=0, help="Seconds to wait between API calls (default: 0)"
    )
    parser.add_argument(
        "-p", "--pro", action="store_true", help="Use gemini-2.5-pro instead of flash model"
    )
    args = parser.parse_args()

    try:
        missing = find_missing(args.day_dir)
    except FileNotFoundError as e:
        print(str(e))
        return

    if not missing:
        print(f"No missing JSON files found in {args.day_dir}.")
        return

    print(f"Found {len(missing)} missing JSON files.")

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not set in environment")
        return

    client = genai.Client(api_key=api_key)
    with open(PROMPT_PATH, "r") as f:
        prompt_text = f.read().strip()

    model = "gemini-2.5-pro" if args.pro else "gemini-2.5-flash"

    logging.basicConfig(level=logging.INFO)
    process_files(missing, args.wait, client, prompt_text, model)


if __name__ == "__main__":
    main()
