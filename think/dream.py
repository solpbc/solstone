# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta

from think.callosum import CallosumConnection
from think.runner import run_task
from think.utils import day_log, day_path, get_insights, setup_cli

# Module-level callosum connection for event emission
_callosum: CallosumConnection | None = None


def run_command(cmd: list[str], day: str) -> bool:
    logging.info("==> %s", " ".join(cmd))
    # Extract command name for logging (e.g., "think-insight" -> "insight")
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
    day: str, force: bool, verbose: bool = False, segment: str | None = None
) -> list[list[str]]:
    """Build processing commands for a day or specific segment.

    Args:
        day: YYYYMMDD format
        segment: Optional HHMMSS_LEN format (e.g., "163045_300")
        force: Overwrite existing files
        verbose: Verbose logging
    """
    commands: list[list[str]] = []

    # Determine target frequency and what to run
    if segment:
        logging.info("Running segment processing for %s/%s", day, segment)
        target_frequency = "segment"
        # No sense repair for segments (already processed during observation)

    else:
        logging.info("Running daily processing for %s", day)
        target_frequency = "daily"
        # Daily-only: repair routines
        cmd = ["observe-sense", "--day", day]
        if verbose:
            cmd.append("-v")
        commands.append(cmd)

    # Run insights filtered by frequency
    insights = get_insights()
    for insight_name, insight_data in insights.items():
        # Skip disabled insights
        if insight_data.get("disabled", False):
            logging.info("Skipping disabled insight: %s", insight_name)
            continue

        # Filter by frequency (defaults to "daily" if not specified)
        insight_frequency = insight_data.get("frequency", "daily")
        if insight_frequency != target_frequency:
            continue

        cmd = ["think-insight", day, "-f", insight_data["path"]]
        if segment:
            cmd.extend(["--segment", segment])
        if verbose:
            cmd.append("--verbose")
        if force:
            cmd.append("--force")
        commands.append(cmd)

    # Re-index (light mode: excludes historical days, mtime-cached)
    indexer_cmd = ["think-indexer", "--rescan"]
    if verbose:
        indexer_cmd.append("--verbose")
    commands.append(indexer_cmd)

    # Daily-only: journal stats
    if not segment:
        stats_cmd = ["think-journal-stats"]
        if verbose:
            stats_cmd.append("--verbose")
        commands.append(stats_cmd)

    return commands


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run processing tasks on a journal day or segment"
    )
    parser.add_argument(
        "--day",
        help="Day folder in YYYYMMDD format (defaults to yesterday)",
    )
    parser.add_argument(
        "--segment",
        help="Segment key in HHMMSS_LEN format (processes segment topics only)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    return parser


def emit(event: str, **fields) -> None:
    """Emit a daily tract event if callosum is connected."""
    if _callosum:
        _callosum.emit("daily", event, **fields)


def main() -> None:
    global _callosum

    parser = parse_args()
    args = setup_cli(parser)

    day = args.day
    if day is None:
        day = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    day_dir = day_path(day)

    if not day_dir.is_dir():
        parser.error(f"Day folder not found: {day_dir}")

    # Start callosum connection for event emission
    _callosum = CallosumConnection()
    _callosum.start()

    try:
        commands = build_commands(
            day, args.force, verbose=args.verbose, segment=args.segment
        )

        # Build command names list for the started event
        command_names = [cmd[0] for cmd in commands]

        # Emit dream_started event
        start_time = time.time()
        emit(
            "dream_started",
            day=day,
            segment=args.segment,
            commands=command_names,
            total=len(commands),
        )

        success_count = 0
        fail_count = 0
        for index, cmd in enumerate(commands):
            # Log every command attempt
            day_log(day, f"starting: {' '.join(cmd)}")

            # Emit dream_command event
            emit(
                "dream_command",
                day=day,
                segment=args.segment,
                command=cmd[0],
                index=index,
                total=len(commands),
            )

            if run_command(cmd, day):
                success_count += 1
            else:
                fail_count += 1

        duration_ms = int((time.time() - start_time) * 1000)

        # Emit dream_completed event
        emit(
            "dream_completed",
            day=day,
            segment=args.segment,
            success=success_count,
            failed=fail_count,
            duration_ms=duration_ms,
        )

        msg = f"think-dream {success_count}"
        if fail_count:
            msg += f" failed {fail_count}"
        if args.force:
            msg += " --force"
        day_log(day, msg)

        if fail_count > 0:
            logging.error(f"{fail_count} insight(s) failed, exiting with error")
            sys.exit(1)
    finally:
        _callosum.stop()


if __name__ == "__main__":
    main()
