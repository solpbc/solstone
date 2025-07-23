import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from think.crumbs import CrumbBuilder
from think.models import GEMINI_FLASH
from think.utils import day_log, day_path, setup_cli

DEFAULT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "reduce.txt")


class TokenTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_candidates_tokens = 0
        self._lock = threading.Lock()

    def add_usage(self, usage_metadata):
        prompt_tokens = getattr(usage_metadata, "prompt_token_count", None) or 0
        candidates_tokens = getattr(usage_metadata, "candidates_token_count", None) or 0
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_candidates_tokens += candidates_tokens

    def get_compression_percent(self):
        if self.total_prompt_tokens == 0:
            return 0.0
        return (1 - self.total_candidates_tokens / self.total_prompt_tokens) * 100

    def print_summary(self):
        logging.info("\n=== Summary ===")
        logging.info("Total tokens in: %s", self.total_prompt_tokens)
        logging.info("Total tokens out: %s", self.total_candidates_tokens)
        logging.info("Total compression: %.2f%%", self.get_compression_percent())


def parse_monitor_files(day_dir):
    pattern = re.compile(r"^(\d{6})_([a-z]+)_(\d+)_diff\.json$")
    entries = []
    for name in os.listdir(day_dir):
        m = pattern.match(name)
        if not m:
            continue
        time_part, source, ident = m.groups()
        try:
            ts = datetime.strptime(time_part, "%H%M%S")
        except ValueError:
            continue
        entries.append(
            {
                "timestamp": ts,
                "source": source,
                "id": int(ident),
                "monitor": int(ident) if source == "monitor" else None,
                "path": os.path.join(day_dir, name),
            }
        )
    entries.sort(key=lambda e: e["timestamp"])
    return entries


def group_entries(entries):
    groups = {}
    for e in entries:
        interval_minute = e["timestamp"].minute - (e["timestamp"].minute % 5)
        start = e["timestamp"].replace(minute=interval_minute, second=0, microsecond=0)
        key = start  # Group all sources together by timestamp
        groups.setdefault(key, []).append(e)
    return groups


def scan_day(day: str) -> dict[str, list[str]]:
    """Return lists of processed screen summaries and repairable diff JSON files."""
    day_dir = Path(day_path(day))
    reduced = sorted(p.name for p in day_dir.glob("*_screen.md"))

    unreduced: list[str] = []
    for start, files in _iter_groups(str(day_dir), None, None):
        md_path = day_dir / f"{start.strftime('%H%M%S')}_screen.md"
        if not md_path.exists():
            unreduced.extend(Path(e["path"]).name for e in files)
            continue
        crumb_path = md_path.with_suffix(md_path.suffix + ".crumb")
        try:
            with open(crumb_path, "r", encoding="utf-8") as f:
                crumb = json.load(f)
            deps = {
                Path(d.get("path", "")).name
                for d in crumb.get("dependencies", [])
                if d.get("type") == "file"
            }
        except Exception:
            unreduced.extend(Path(e["path"]).name for e in files)
            continue
        for e in files:
            if Path(e["path"]).name not in deps:
                unreduced.append(Path(e["path"]).name)

    return {"processed": reduced, "repairable": sorted(set(unreduced))}


def load_prompt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.error("Prompt file not found: %s", path)
        sys.exit(1)


def _get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment")
    return api_key


def call_gemini(markdown, prompt, api_key, debug=False):
    client = genai.Client(api_key=api_key)
    done = threading.Event()

    def progress():
        elapsed = 0
        while not done.is_set():
            time.sleep(5)
            elapsed += 5
            if not done.is_set():
                logging.info("... %ss elapsed", elapsed)

    if debug:
        logging.debug("\n=== DEBUG: Prompt to Gemini ===")
        logging.debug("System instruction: %s", prompt)
        logging.debug("\n=== DEBUG: Content to Gemini ===")
        logging.debug(markdown)
        logging.debug("\n=== DEBUG: End of input ===\n")

    t = threading.Thread(target=progress, daemon=True)
    t.start()
    try:
        response = client.models.generate_content(
            model=GEMINI_FLASH,
            contents=[markdown],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=8192 * 2,
                system_instruction=prompt,
            ),
        )

        if debug:
            logging.debug("\n=== DEBUG: Response from Gemini ===")
            logging.debug(response.text)
            logging.debug("\n=== DEBUG: End of response ===\n")

        return response.text, response.usage_metadata
    finally:
        done.set()
        t.join()


def process_group(
    start,
    files,
    prompt,
    prompt_path,
    api_key,
    force,
    day_dir,
    token_tracker,
    debug=False,
):
    out_name = f"{start.strftime('%H%M%S')}_screen.md"
    out_path = os.path.join(day_dir, out_name)
    if os.path.exists(out_path) and not force:
        logging.info("Skipping existing %s", out_name)
        return

    lines = []
    end = start + timedelta(minutes=5)
    lines.append(
        f"## Screen Activity {start.strftime('%H:%M')} - {end.strftime('%H:%M')}"
    )
    lines.append("")

    # Group files by monitor for organized display
    monitor_files = {}
    for e in files:
        monitor_files.setdefault(e["monitor"], []).append(e)

    # Process each monitor in order
    for monitor in sorted(monitor_files.keys()):
        monitor_entries = sorted(monitor_files[monitor], key=lambda x: x["timestamp"])
        lines.append(f"### Monitor {monitor}")
        lines.append("")

        for e in monitor_entries:
            with open(e["path"], "r", encoding="utf-8") as f:
                content = f.read().strip()
            lines.append(f"#### {e['timestamp'].strftime('%H:%M:%S')}")
            lines.append("```json")
            lines.append(content)
            lines.append("```")
            lines.append("")

    markdown = "\n".join(lines)
    monitor_count = len(monitor_files)
    file_count = len(files)
    logging.info(
        "Processing screen activity starting %s (%s monitors, %s files)",
        start.strftime("%H:%M"),
        monitor_count,
        file_count,
    )
    try:
        result, usage_metadata = call_gemini(markdown, prompt, api_key, debug)
        token_tracker.add_usage(usage_metadata)

        # Calculate compression for this file
        prompt_tokens = usage_metadata.prompt_token_count
        candidates_tokens = usage_metadata.candidates_token_count
        compression = (
            (1 - candidates_tokens / prompt_tokens) * 100 if prompt_tokens > 0 else 0
        )

        logging.info(
            "Tokens: (%s) -> (%s) compression: %.2f%%",
            prompt_tokens,
            candidates_tokens,
            compression,
        )
    except Exception as e:
        raise RuntimeError(f"Gemini call failed: {e}") from e

    logging.info("Writing %s (%s)", out_name, len(result))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    logging.info("Saved %s", out_path)

    crumb_builder = (
        CrumbBuilder()
        .add_file(prompt_path)
        .add_files([e["path"] for e in files])
        .add_model(GEMINI_FLASH)
    )
    crumb_path = crumb_builder.commit(out_path)
    logging.info("Crumb saved to: %s", crumb_path)


def _normalize_time(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return datetime.combine(datetime.min, value.time())


def _iter_groups(day_dir: str, start: datetime | None, end: datetime | None):
    entries = parse_monitor_files(day_dir)
    if not entries:
        return
    groups = group_entries(entries)
    for group_start, files in sorted(groups.items()):
        if start and group_start < start:
            continue
        if end and group_start >= end:
            continue
        yield group_start, files


def reduce_day(
    day: str,
    prompt_path: str = DEFAULT_PROMPT_PATH,
    *,
    force: bool = False,
    debug: bool = False,
    start: datetime | None = None,
    end: datetime | None = None,
    jobs: int = 1,
) -> tuple[int, int]:
    """Process monitor diffs under ``day`` between ``start`` and ``end``.

    When ``start``/``end`` are ``None`` all available 5 minute groups are processed.
    The function mirrors the behaviour of the ``see-reduce`` CLI and is exposed
    so that other modules can opportunistically reduce recent captures.

    Parameters
    ----------
    jobs:
        Number of parallel Gemini calls to allow. A ``ThreadPoolExecutor`` is
        always used so ``jobs=1`` still executes via the thread pool.
    """

    day_dir = day_path(day)
    if not os.path.isdir(day_dir):
        logging.error("Folder not found: %s", day_dir)
        return 0, 0

    start = _normalize_time(start)
    end = _normalize_time(end)

    logging.info("Processing folder: %s", day_dir)

    groups_iter = list(_iter_groups(day_dir, start, end))
    if not groups_iter:
        logging.info("No monitor diff JSON files found")
        return 0, 0

    try:
        api_key = _get_api_key()
    except RuntimeError as exc:
        logging.error(str(exc))
        return 0, 0

    prompt = load_prompt(prompt_path)
    token_tracker = TokenTracker()

    success = 0
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        future_map = {
            executor.submit(
                process_group,
                group_start,
                files,
                prompt,
                prompt_path,
                api_key,
                force,
                day_dir,
                token_tracker,
                debug,
            ): group_start
            for group_start, files in groups_iter
        }
        for future in concurrent.futures.as_completed(future_map):
            group_start = future_map[future]
            try:
                future.result()
                success += 1
            except Exception as e:
                logging.error("Error reducing %s: %s", group_start, e)
                failed += 1

    token_tracker.print_summary()

    return success, failed


def main():
    parser = argparse.ArgumentParser(
        description="Summarize all diff JSON files in 5 minute chunks using Gemini"
    )
    parser.add_argument(
        "day", help="Day in YYYYMMDD format containing *_<source>_<id>_diff.json files"
    )
    parser.add_argument(
        "-p", "--prompt", default=DEFAULT_PROMPT_PATH, help="Prompt file"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing markdown files"
    )
    parser.add_argument("--start", help="Start time HH:MM for partial processing")
    parser.add_argument("--end", help="End time HH:MM for partial processing")
    parser.add_argument(
        "-j", "--jobs", type=int, default=1, help="Number of parallel Gemini calls"
    )
    args = setup_cli(parser)

    day_dir = day_path(args.day)
    if not os.path.isdir(day_dir):
        parser.error(f"Folder not found: {day_dir}")

    start = datetime.strptime(args.start, "%H:%M") if args.start else None
    end = datetime.strptime(args.end, "%H:%M") if args.end else None

    success, failed = reduce_day(
        args.day,
        args.prompt,
        force=args.force,
        debug=args.debug,
        start=start,
        end=end,
        jobs=max(1, args.jobs),
    )

    msg = f"see-reduce {success} blocks"
    if failed:
        msg += f" failed {failed}"
    if args.force:
        msg += " --force"
    day_log(args.day, msg)


if __name__ == "__main__":
    main()
