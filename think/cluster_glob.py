import argparse
import glob
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

PRO_MODEL = "gemini-2.5-pro"
FLASH_MODEL = "gemini-2.5-flash"


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """Extract date from filename containing YYYYMMDD pattern."""
    # Look for YYYYMMDD pattern in the entire filepath, not just basename
    date_match = re.search(r"(\d{8})", filename)
    if not date_match:
        return None

    date_str = date_match.group(1)
    try:
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        return datetime(year, month, day)
    except ValueError:
        return None


def format_friendly_date(dt: datetime) -> str:
    """Convert datetime to friendly format like 'Monday May 1st, 2025'."""
    day_name = dt.strftime("%A")
    month_name = dt.strftime("%B")
    day_num = dt.day
    year = dt.year

    # Add ordinal suffix
    if 10 <= day_num % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day_num % 10, "th")

    return f"{day_name} {month_name} {day_num}{suffix}, {year}"


def cluster_glob(filepaths: List[str]) -> str:
    """Generate markdown from files with friendly date headers."""
    if not filepaths:
        return "No files provided"

    # Process files and extract dates
    file_data: List[Tuple[datetime, str, str]] = []

    for filepath in filepaths:
        if not os.path.isfile(filepath):
            print(f"Warning: File not found {filepath}. Skipping.", file=sys.stderr)
            continue

        date = extract_date_from_filename(filepath)
        if date is None:
            print(
                f"Warning: Could not extract date from filename {filepath}. Skipping.",
                file=sys.stderr,
            )
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            file_data.append((date, filepath, content))
        except Exception as e:
            print(f"Warning: Could not read file {filepath}: {e}", file=sys.stderr)
            continue

    if not file_data:
        return "No valid files with extractable dates found"

    # Sort by date
    file_data.sort(key=lambda x: x[0])

    # Generate markdown
    lines = []
    for date, filepath, content in file_data:
        friendly_date = format_friendly_date(date)
        lines.append(f"# {friendly_date}")
        lines.append("")
        lines.append(content.strip())
        lines.append("")

    return "\n".join(lines)


def send_to_gemini(
    markdown_content: str, prompt_text: str, api_key: str, model_name: str, is_json_mode: bool
) -> tuple[Optional[str], Optional[object]]:
    """Send markdown content and a prompt to Gemini API."""
    client = genai.Client(api_key=api_key)

    done = threading.Event()

    def progress():
        elapsed = 0
        while not done.is_set():
            time.sleep(5)
            elapsed += 5
            if not done.is_set():
                print(f"... {elapsed}s elapsed", file=sys.stderr)

    t = threading.Thread(target=progress, daemon=True)
    t.start()
    try:
        generation_config_args = {
            "temperature": 0.3,
            "max_output_tokens": 8192 * 2,
            "thinking_config": types.ThinkingConfig(
                thinking_budget=8192 * 2,
            ),
            "system_instruction": prompt_text,
        }
        if is_json_mode:
            generation_config_args["response_mime_type"] = "application/json"

        response = client.models.generate_content(
            model=model_name,
            contents=[markdown_content],
            config=types.GenerateContentConfig(**generation_config_args),
        )
        return response.text, response.usage_metadata

    except Exception as e:
        print(f"Error during Gemini API call: {e}", file=sys.stderr)
        return None, None
    finally:
        done.set()
        t.join()


def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown from files with friendly date headers, optionally sending to Gemini."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="File paths (use shell globbing: ~/dir/2025*/ponder*.md)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="Path to a prompt text file to use with Gemini.",
    )

    args = parser.parse_args()

    try:
        markdown_output = cluster_glob(args.files)

        if args.prompt:
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("Error: GOOGLE_API_KEY not found in environment variables.", file=sys.stderr)
                sys.exit(1)

            try:
                with open(args.prompt, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
            except FileNotFoundError:
                print(f"Error: Prompt file not found: {args.prompt}", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"Error reading prompt file {args.prompt}: {e}", file=sys.stderr)
                sys.exit(1)

            model_name = PRO_MODEL
            is_json_mode = "```json" in prompt_text.lower()  # Check for json in prompt

            print(f"Sending to Gemini with model: {model_name}", file=sys.stderr)
            if is_json_mode:
                print("JSON mode detected in prompt.", file=sys.stderr)

            gemini_response_text, usage_metadata = send_to_gemini(
                markdown_output, prompt_text, api_key, model_name, is_json_mode
            )

            if usage_metadata:
                prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
                thoughts_tokens = getattr(usage_metadata, "thoughts_token_count", 0)
                candidates_tokens = getattr(usage_metadata, "candidates_token_count", 0)
                print(
                    f"Usage: prompt={prompt_tokens} thoughts={thoughts_tokens} candidates={candidates_tokens}"
                )

            if gemini_response_text:
                print(gemini_response_text)
            else:
                print("Error: No response text received from Gemini.", file=sys.stderr)
                sys.exit(1)
        else:
            print(markdown_output)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
