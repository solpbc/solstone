import argparse
import os
import time
import json
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types


PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "hear", "gemini_mic.txt")


def find_missing(folder, day):
    day_dir = os.path.join(folder, day)
    if not os.path.isdir(day_dir):
        raise FileNotFoundError(f"Day directory not found: {day_dir}")
    missing = []
    for name in sorted(os.listdir(day_dir)):
        if name.endswith("_audio.ogg"):
            base = name[:-4]
            json_path = os.path.join(day_dir, base + "json")
            if not os.path.exists(json_path):
                ogg_path = os.path.join(day_dir, name)
                missing.append((ogg_path, json_path))
    return missing


def transcribe_file(client, prompt_text, ogg_path):
    with open(ogg_path, "rb") as f:
        ogg_bytes = f.read()
    size_mb = len(ogg_bytes) / (1024 * 1024)
    logging.info(f"Transcribing {ogg_path} ({size_mb:.2f}MB)")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=[
                "Process the provided audio now and output your professional accurate transcription in the specified JSON format.",
                types.Part.from_bytes(data=ogg_bytes, mime_type="audio/ogg"),
            ],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=8192,
                response_mime_type="application/json",
                system_instruction=prompt_text,
            ),
        )
        return response.text
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return ""


def process_files(files, delay, client, prompt_text):
    for ogg_path, json_path in files:
        result = transcribe_file(client, prompt_text, ogg_path)
        if result:
            with open(json_path, "w") as f:
                json.dump({"text": result}, f)
            print(f"Saved {json_path}")
        else:
            print(f"Gemini returned no result for {ogg_path}")
        time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Repair missing Gemini JSON for audio files")
    parser.add_argument("folder", help="Base directory containing day folders")
    parser.add_argument("day", help="Day folder (YYYYMMDD)")
    args = parser.parse_args()

    try:
        missing = find_missing(args.folder, args.day)
    except FileNotFoundError as e:
        print(str(e))
        return

    if not missing:
        print("No missing JSON files found.")
        return

    print(f"Found {len(missing)} missing JSON files.")
    try:
        delay = float(input("Seconds to wait between API calls (0 to cancel): "))
    except ValueError:
        print("Invalid input; aborting")
        return
    if delay <= 0:
        print("Aborted")
        return

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not set in environment")
        return

    client = genai.Client(api_key=api_key)
    with open(PROMPT_PATH, "r") as f:
        prompt_text = f.read().strip()

    logging.basicConfig(level=logging.INFO)
    process_files(missing, delay, client, prompt_text)


if __name__ == "__main__":
    main()
