import argparse
import glob
import logging
import os
import subprocess
from datetime import datetime, timedelta

from think.utils import day_log, day_path, get_topics, setup_cli


def run_command(cmd: list[str], day: str) -> bool:
    logging.info("==> %s", " ".join(cmd))
    # Extract command name for logging (e.g., "think-summarize" -> "summarize")
    cmd_name = cmd[0].replace("think-", "").replace("-", "_")
    try:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logging.error(
                "Command failed with exit code %s: %s", result.returncode, " ".join(cmd)
            )
            day_log(day, f"{cmd_name} error {result.returncode}")
            return False
        return True
    except Exception as e:
        logging.error("Command exception: %s: %s", e, " ".join(cmd))
        day_log(day, f"{cmd_name} exception")
        return False


def build_commands(day: str, force: bool, verbose: bool = False) -> list[list[str]]:
    commands: list[list[str]] = []

    logging.info("Running repair routines for %s", day)
    cmd = ["observe-sense", "--day", day]
    if verbose:
        cmd.append("-v")
    commands.append(cmd)

    reduce_cmd = ["see-reduce", day]
    if verbose:
        reduce_cmd.append("--verbose")
    if force:
        reduce_cmd.append("--force")
    commands.append(reduce_cmd)

    topics = get_topics()
    for topic_name, topic_data in topics.items():
        # Skip disabled topics
        if topic_data.get("disabled", False):
            logging.info("Skipping disabled topic: %s", topic_name)
            continue
        cmd = ["think-summarize", day, "-f", topic_data["path"], "-p"]
        if verbose:
            cmd.append("--verbose")
        if force:
            cmd.append("--force")
        commands.append(cmd)

    entity_cmd = ["think-entity-roll", "--day", day]
    if verbose:
        entity_cmd.append("--verbose")
    if force:
        entity_cmd.append("--force")
    commands.append(entity_cmd)

    # Run journal stats at the end to update overall statistics
    stats_cmd = ["think-journal-stats"]
    if verbose:
        stats_cmd.append("--verbose")
    commands.append(stats_cmd)

    # Rescan all indexes to pick up the new day's content
    indexer_cmd = ["think-indexer", "--rescan-all"]
    if verbose:
        indexer_cmd.append("--verbose")
    commands.append(indexer_cmd)

    return commands


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run daily processing tasks on a journal day"
    )
    parser.add_argument(
        "--day",
        help="Day folder in YYYYMMDD format (defaults to yesterday)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Remove existing outputs before running",
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
    day_dir = day_path(day)

    if not day_dir.is_dir():
        parser.error(f"Day folder not found: {day_dir}")

    if args.rebuild:
        for pattern in ("*_audio.json", "*_[a-z]*_*_diff.json"):
            for path in glob.glob(os.path.join(day_dir, pattern)):
                try:
                    os.remove(path)
                except OSError:
                    pass
                crumb = path + ".crumb"
                if os.path.exists(crumb):
                    os.remove(crumb)

    commands = build_commands(day, args.force, verbose=args.verbose)
    success_count = 0
    fail_count = 0
    for cmd in commands:
        # Log every command attempt
        day_log(day, f"starting: {' '.join(cmd)}")

        if run_command(cmd, day):
            success_count += 1
        else:
            fail_count += 1

    msg = f"think-dream {success_count}"
    if fail_count:
        msg += f" failed {fail_count}"
    if args.force:
        msg += " --force"
    day_log(day, msg)


if __name__ == "__main__":
    main()
