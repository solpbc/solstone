import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime, timedelta


def run_command(cmd: list[str]) -> None:
    print(f"==> {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        sys.exit(result.returncode)


def build_commands(journal: str, day: str, force: bool, repair: bool) -> list[list[str]]:
    day_dir = os.path.join(journal, day)
    commands: list[list[str]] = []

    if repair:
        commands.append(["gemini-transcribe", journal, "--repair", day])
        commands.append(["screen-describe", journal, "--repair", day])

    reduce_cmd = ["reduce-screen", day_dir]
    if force:
        reduce_cmd.append("--force")
    commands.append(reduce_cmd)

    think_dir = os.path.dirname(__file__)
    prompt_paths = sorted(glob.glob(os.path.join(think_dir, "ponder", "*.txt")))
    for prompt in prompt_paths:
        cmd = ["ponder-day", day_dir, "-f", prompt]
        if force:
            cmd.append("--force")
        commands.append(cmd)

    entity_cmd = ["entity-roll", journal]
    if force:
        entity_cmd.append("--force")
    commands.append(entity_cmd)

    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily processing tasks on a journal day")
    parser.add_argument("--journal", required=True, help="Path to the journal directory")
    parser.add_argument(
        "--day",
        help="Day folder in YYYYMMDD format (defaults to yesterday)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Run hear and see repair routines before processing",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Remove existing outputs before running repairs (implies --repair)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    day = args.day
    if day is None:
        day = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    day_dir = os.path.join(args.journal, day)
    if not os.path.isdir(day_dir):
        print(f"Day folder not found: {day_dir}")
        sys.exit(1)

    repair = args.repair or args.rebuild
    if args.rebuild:
        for pattern in ("*_audio.json", "*_monitor_*_diff.json"):
            for path in glob.glob(os.path.join(day_dir, pattern)):
                try:
                    os.remove(path)
                except OSError:
                    pass
                crumb = path + ".crumb"
                if os.path.exists(crumb):
                    os.remove(crumb)

    commands = build_commands(args.journal, day, args.force, repair)
    for cmd in commands:
        run_command(cmd)


if __name__ == "__main__":
    main()
