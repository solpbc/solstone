import argparse
import glob
import logging
import os
import subprocess
from datetime import datetime, timedelta

from think.utils import day_log, setup_cli


def run_command(cmd: list[str]) -> bool:
    logging.info("==> %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logging.error("Command failed with exit code %s: %s", result.returncode, " ".join(cmd))
        return False
    return True


def build_commands(day: str, force: bool, repair: bool, verbose: bool = False) -> list[list[str]]:
    commands: list[list[str]] = []

    if repair:
        logging.info("Running repair routines for %s", day)
        cmd = ["gemini-transcribe", "--repair", day]
        if verbose:
            cmd.append("-v")
        commands.append(cmd)
        cmd = ["screen-describe", "--repair", day]
        if verbose:
            cmd.append("-v")
        commands.append(cmd)

    reduce_cmd = ["reduce-screen", day]
    if verbose:
        reduce_cmd.append("--verbose")
    if force:
        reduce_cmd.append("--force")
    commands.append(reduce_cmd)

    think_dir = os.path.dirname(__file__)
    prompt_paths = sorted(glob.glob(os.path.join(think_dir, "ponder", "*.txt")))
    for prompt in prompt_paths:
        cmd = ["ponder", day, "-f", prompt, "-p"]
        if verbose:
            cmd.append("--verbose")
        if force:
            cmd.append("--force")
        commands.append(cmd)

    entity_cmd = ["entity-roll", "--day", day]
    if verbose:
        entity_cmd.append("--verbose")
    if force:
        entity_cmd.append("--force")
    commands.append(entity_cmd)

    return commands


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run daily processing tasks on a journal day")
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
    return parser


def main() -> None:
    parser = parse_args()
    args = setup_cli(parser)
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        parser.error("JOURNAL_PATH not set")

    day = args.day
    if day is None:
        day = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    day_dir = os.path.join(journal, day)

    if not os.path.isdir(day_dir):
        parser.error(f"Day folder not found: {day_dir}")

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

    commands = build_commands(day, args.force, repair, verbose=args.verbose)
    success_count = 0
    fail_count = 0
    for cmd in commands:
        if run_command(cmd):
            success_count += 1
        else:
            fail_count += 1

    msg = f"process-day {success_count}"
    if fail_count:
        msg += f" failed {fail_count}"
    if args.force:
        msg += " --force"
    day_log(day, msg)


if __name__ == "__main__":
    main()
