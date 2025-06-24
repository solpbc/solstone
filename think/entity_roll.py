import argparse
import glob
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List

from dotenv import load_dotenv

from think.cluster_glob import PRO_MODEL, cluster_glob, send_to_gemini
from think.crumbs import CrumbBuilder

DATE_RE = re.compile(r"\d{8}")
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "entity_roll.txt")


def find_day_dirs(journal: str) -> Dict[str, str]:
    """Return mapping of YYYYMMDD string to full path."""
    days = {}
    for name in os.listdir(journal):
        if DATE_RE.fullmatch(name):
            path = os.path.join(journal, name)
            if os.path.isdir(path):
                days[name] = path
    return days


def gather_files(day: datetime, day_dirs: Dict[str, str]) -> List[str]:
    files = []
    for i in range(8):
        d = day - timedelta(days=i)
        key = d.strftime("%Y%m%d")
        dir_path = day_dirs.get(key)
        if not dir_path:
            continue
        kg_pattern = os.path.join(dir_path, "ponder_kg*.md")
        kg_files = glob.glob(kg_pattern)
        files.extend(kg_files)
    return files


def process_day(day_str: str, day_dirs: Dict[str, str], force: bool) -> None:
    out_path = os.path.join(day_dirs[day_str], "entities.md")
    if os.path.exists(out_path) and not force:
        print(f"Skipping {day_str}: entities.md exists")
        return

    day = datetime.strptime(day_str, "%Y%m%d")
    files = gather_files(day, day_dirs)
    if not files:
        print(f"No ponder_kg.md files for {day_str}")
        return

    print(f"Processing {day_str}:")
    print(f"  Found {len(files)} ponder_kg files from 8-day window")
    for file in files:
        print(f"    {os.path.basename(file)}")

    print("  Clustering and merging content...")
    markdown = cluster_glob(files)

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not found in environment")
        return

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    print("  Sending to Gemini for entity extraction...")
    result, _ = send_to_gemini(markdown, prompt, api_key, PRO_MODEL, False)
    if not result:
        print(f"Gemini returned no result for {day_str}")
        return

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Wrote {out_path}")

    crumb_builder = CrumbBuilder().add_file(PROMPT_PATH).add_files(files).add_model(PRO_MODEL)
    crumb_path = crumb_builder.commit(out_path)
    print(f"Crumb saved to: {crumb_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge ponder_kg files from a rolling 8-day window and generate entities.md"
    )
    parser.add_argument("journal", help="Journal directory containing YYYYMMDD folders")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    if not os.path.isdir(args.journal):
        parser.error(f"Journal directory not found: {args.journal}")

    day_dirs = find_day_dirs(args.journal)
    if not day_dirs:
        parser.error("No YYYYMMDD directories found")

    for day_str in sorted(day_dirs.keys()):
        process_day(day_str, day_dirs, args.force)


if __name__ == "__main__":
    main()
