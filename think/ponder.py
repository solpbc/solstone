import argparse
import glob
import json
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from think.cluster import cluster
from think.crumbs import CrumbBuilder
from think.models import GEMINI_FLASH, GEMINI_PRO
from think.utils import day_log, day_path

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "ponder")
DEFAULT_PROMPT_PATH = os.path.join(PROMPT_DIR, "day.txt")


def _prompt_basenames() -> list[str]:
    """Return available prompt basenames under :data:`PROMPT_DIR`."""
    return sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(PROMPT_DIR, "*.txt"))
    )


def _output_paths(day_dir: os.PathLike[str], basename: str) -> tuple[Path, Path]:
    """Return markdown and JSON output paths for ``basename`` in ``day_dir``."""
    day = Path(day_dir)
    return day / f"ponder_{basename}.md", day / f"ponder_{basename}.json"


def scan_day(day: str) -> dict[str, list[str]]:
    """Return lists of processed and pending ponder markdown files."""
    day_dir = Path(day_path(day))
    pondered: list[str] = []
    unpondered: list[str] = []
    for base in _prompt_basenames():
        md_path, _ = _output_paths(day_dir, base)
        if md_path.exists():
            pondered.append(md_path.name)
        else:
            unpondered.append(md_path.name)
    return {"processed": sorted(pondered), "repairable": sorted(unpondered)}


def count_tokens(markdown: str, prompt: str, api_key: str, model: str) -> None:
    client = genai.Client(api_key=api_key)

    total_tokens = client.models.count_tokens(
        model=model,
        contents=[markdown],
    )
    print(f"Token count: {total_tokens}")


def send_markdown(markdown: str, prompt: str, api_key: str, model: str) -> tuple[str, object]:
    client = genai.Client(api_key=api_key)

    done = threading.Event()

    def progress():
        elapsed = 0
        while not done.is_set():
            time.sleep(5)
            elapsed += 5
            if not done.is_set():
                print(f"... {elapsed}s elapsed")

    t = threading.Thread(target=progress, daemon=True)
    t.start()
    try:
        gen_config_args = {
            "temperature": 0.3,
            "max_output_tokens": 8192 * 6,
            "thinking_config": types.ThinkingConfig(
                thinking_budget=8192 * 3,
            ),
            "system_instruction": prompt,
        }

        response = client.models.generate_content(
            model=model,
            contents=[markdown],
            config=types.GenerateContentConfig(**gen_config_args),
        )
        return response.text, response.usage_metadata
    finally:
        done.set()
        t.join()


def send_occurrence(markdown: str, prompt: str, api_key: str, model: str) -> object:
    """Send markdown to generate occurrence data and return parsed JSON."""
    client = genai.Client(api_key=api_key)

    gen_config_args = {
        "temperature": 0.3,
        "max_output_tokens": 8192 * 6,
        "thinking_config": types.ThinkingConfig(
            thinking_budget=8192 * 3,
        ),
        "response_mime_type": "application/json",
        "system_instruction": prompt,
    }

    response = client.models.generate_content(
        model=model,
        contents=[markdown],
        config=types.GenerateContentConfig(**gen_config_args),
    )

    try:
        occurrences = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}: {response.text[:100]}")

    if not isinstance(occurrences, list):
        raise ValueError(f"Response is not an array: {response.text[:100]}")

    return occurrences


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a day's clustered Markdown to Gemini for analysis."
    )
    parser.add_argument(
        "day",
        help="Day in YYYYMMDD format",
    )
    parser.add_argument(
        "-f",
        "--prompt",
        default=DEFAULT_PROMPT_PATH,
        help="Prompt file to use",
    )
    parser.add_argument(
        "-p",
        "--pro",
        action="store_true",
        help="Use the gemini 2.5 pro model",
    )
    parser.add_argument(
        "-c",
        "--count",
        action="store_true",
        help="Count tokens only and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    markdown, file_count = cluster(args.day)
    day_dir = day_path(args.day)
    prompt_basename = Path(args.prompt).stem
    success = False

    try:

        load_dotenv()
        if args.verbose:
            print("Verbose mode enabled")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            parser.error("GOOGLE_API_KEY not found in environment")

        try:
            with open(args.prompt, "r") as f:
                prompt = f.read().strip()
        except FileNotFoundError:
            parser.error(f"Prompt file not found: {args.prompt}")

        model = GEMINI_PRO if args.pro else GEMINI_FLASH
        day = args.day
        size_kb = len(markdown.encode("utf-8")) / 1024

        print(
            f"Prompt: {args.prompt} | Model: {model} | Day: {day} | Files: {file_count} | Size: {size_kb:.1f}KB"
        )

        if args.count:
            count_tokens(markdown, prompt, api_key, model)
            return

        md_path, json_path = _output_paths(day_dir, prompt_basename)

        # Check if markdown file already exists
        md_exists = md_path.exists() and md_path.stat().st_size > 0

        if md_exists and not args.force:
            print(f"Markdown file already exists: {md_path}. Loading existing content.")
            with open(md_path, "r") as f:
                result = f.read()
            usage_metadata = None
        elif md_exists and args.force:
            print("Markdown file exists but --force specified. Regenerating.")
            result, usage_metadata = send_markdown(markdown, prompt, api_key, model)
        else:
            result, usage_metadata = send_markdown(markdown, prompt, api_key, model)

        # Extract and display only the essential token counts if we have usage data
        if usage_metadata is not None:
            prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
            thoughts_tokens = getattr(usage_metadata, "thoughts_token_count", 0)
            candidates_tokens = getattr(usage_metadata, "candidates_token_count", 0)
            print(
                f"Usage: prompt={prompt_tokens} thoughts={thoughts_tokens} candidates={candidates_tokens}"
            )

        # Check if we got a valid response
        if result is None:
            print("Error: No text content in response")
            return

        # Only write markdown if it was newly generated
        if not md_exists or args.force:
            os.makedirs(day_dir, exist_ok=True)
            with open(md_path, "w") as f:
                f.write(result)
            print(f"Results saved to: {md_path}")

            crumb_builder = (
                CrumbBuilder()
                .add_file(args.prompt)
                .add_glob(os.path.join(day_dir, "*_audio.json"))
                .add_glob(os.path.join(day_dir, "*_screen.md"))
                .add_model(model)
            )
            crumb_path = crumb_builder.commit(str(md_path))
            print(f"Crumb saved to: {crumb_path}")

        # Create a corresponding occurrence JSON from the markdown summary
        occ_prompt_path = os.path.join(os.path.dirname(__file__), "ponder.txt")
        try:
            with open(occ_prompt_path, "r") as f:
                occ_prompt = f.read().strip().replace("DAY", day)
        except FileNotFoundError:
            print(f"Occurrence prompt not found: {occ_prompt_path}")
            return

        occ_output_path = json_path
        json_exists = occ_output_path.exists() and occ_output_path.stat().st_size > 0

        if json_exists and not args.force:
            print(f"JSON file already exists: {occ_output_path}. Use --force to overwrite.")
            return
        elif json_exists and args.force:
            print("JSON file exists but --force specified. Regenerating.")

        try:
            occurrences = send_occurrence(result, occ_prompt, api_key, model)
        except ValueError as e:
            print(f"Error: {e}")
            return

        full_occurrence_obj = {"day": day, "occurrences": occurrences}

        occ_result = json.dumps(full_occurrence_obj, indent=2)

        with open(occ_output_path, "w") as f:
            f.write(occ_result)

        print(f"Results saved to: {occ_output_path}")

        occ_crumb_builder = (
            CrumbBuilder().add_file(occ_prompt_path).add_file(md_path).add_model(model)
        )
        occ_crumb_path = occ_crumb_builder.commit(str(occ_output_path))
        print(f"Crumb saved to: {occ_crumb_path}")
        success = True

    finally:
        msg = f"ponder {prompt_basename} {'ok' if success else 'failed'}"
        if args.force:
            msg += " --force"
        day_log(args.day, msg)


if __name__ == "__main__":
    main()
