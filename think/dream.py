import argparse
import logging
import os
from datetime import datetime, timedelta

from think.runner import run_task
from think.utils import day_log, day_path, get_topics, setup_cli


def run_command(cmd: list[str], day: str) -> bool:
    logging.info("==> %s", " ".join(cmd))
    # Extract command name for logging (e.g., "think-summarize" -> "summarize")
    cmd_name = cmd[0].replace("think-", "").replace("-", "_")

    # Use unified runner with automatic logging
    try:
        success, exit_code = run_task(cmd)
        if not success:
            logging.error(
                "Command failed with exit code %s: %s", exit_code, " ".join(cmd)
            )
            day_log(day, f"{cmd_name} error {exit_code}")
            return False
        return True
    except Exception as e:
        logging.error("Command exception: %s: %s", e, " ".join(cmd))
        day_log(day, f"{cmd_name} exception")
        return False


def build_commands(
    day: str, force: bool, verbose: bool = False, period: str | None = None
) -> list[list[str]]:
    """Build processing commands for a day or specific period.

    Args:
        day: YYYYMMDD format
        period: Optional HHMMSS_LEN format (e.g., "163045_300")
        force: Overwrite existing files
        verbose: Verbose logging
    """
    commands: list[list[str]] = []

    # Determine target frequency and what to run
    if period:
        logging.info("Running period processing for %s/%s", day, period)
        target_frequency = "period"
        # No sense repair for periods (already processed during observation)
    else:
        logging.info("Running daily processing for %s", day)
        target_frequency = "daily"
        # Daily-only: repair routines
        cmd = ["observe-sense", "--day", day]
        if verbose:
            cmd.append("-v")
        commands.append(cmd)

    # Run topics filtered by frequency
    topics = get_topics()
    for topic_name, topic_data in topics.items():
        # Skip disabled topics
        if topic_data.get("disabled", False):
            logging.info("Skipping disabled topic: %s", topic_name)
            continue

        # Filter by frequency (defaults to "daily" if not specified)
        topic_frequency = topic_data.get("frequency", "daily")
        if topic_frequency != target_frequency:
            continue

        cmd = ["think-summarize", day, "-f", topic_data["path"], "-p"]
        if period:
            cmd.extend(["--period", period])
        if verbose:
            cmd.append("--verbose")
        if force:
            cmd.append("--force")
        commands.append(cmd)

    # Targeted indexing
    indexer_cmd = ["think-indexer", "--rescan-all", "--day", day]
    if period:
        indexer_cmd.extend(["--period", period])
    if verbose:
        indexer_cmd.append("--verbose")
    commands.append(indexer_cmd)

    # Daily-only: journal stats
    if not period:
        stats_cmd = ["think-journal-stats"]
        if verbose:
            stats_cmd.append("--verbose")
        commands.append(stats_cmd)

    return commands


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run processing tasks on a journal day or period"
    )
    parser.add_argument(
        "--day",
        help="Day folder in YYYYMMDD format (defaults to yesterday)",
    )
    parser.add_argument(
        "--period",
        help="Period key in HHMMSS_LEN format (processes period topics only)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
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

    commands = build_commands(day, args.force, verbose=args.verbose, period=args.period)
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
