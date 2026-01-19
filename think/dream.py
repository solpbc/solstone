# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from muse.cortex_client import cortex_request, get_agent_status
from think.callosum import CallosumConnection
from think.facets import get_active_facets, get_facets
from think.runner import run_task
from think.utils import (
    day_input_summary,
    day_log,
    day_path,
    get_agents,
    get_insights,
    get_journal,
    setup_cli,
)

# Module-level callosum connection for event emission
_callosum: CallosumConnection | None = None


def run_command(cmd: list[str], day: str) -> bool:
    logging.info("==> %s", " ".join(cmd))
    # Extract command name for logging (e.g., "sol insight" -> "insight")
    cmd_name = cmd[1] if cmd[0] == "sol" else cmd[0]
    cmd_name = cmd_name.replace("-", "_")

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


def run_queued_command(cmd: list[str], day: str, timeout: int = 600) -> bool:
    """Run a command through supervisor's task queue and wait for completion.

    This ensures the command is serialized with other requests for the same
    command type (e.g., multiple indexer requests are queued, not concurrent).

    Args:
        cmd: Command to run (e.g., ["sol", "indexer", "--rescan"])
        day: Day for logging
        timeout: Maximum wait time in seconds (default 600 = 10 minutes)

    Returns:
        True if command succeeded, False otherwise
    """
    import threading
    import uuid

    cmd_name = cmd[1] if cmd[0] == "sol" else cmd[0]
    cmd_name_log = cmd_name.replace("-", "_")

    # Generate unique ref to track this specific request
    ref = f"dream-{uuid.uuid4().hex[:8]}"

    logging.info("==> %s (queued, ref=%s)", " ".join(cmd), ref)

    if not _callosum:
        logging.error("Callosum not connected, cannot queue command")
        day_log(day, f"{cmd_name_log} error no_callosum")
        return False

    # Track completion via supervisor.stopped event matching our ref
    result = {"completed": False, "exit_code": None}
    result_event = threading.Event()

    def on_message(msg: dict) -> None:
        if msg.get("tract") != "supervisor":
            return
        if msg.get("event") != "stopped":
            return
        # Match by ref to ensure we're waiting for OUR request, not another
        if msg.get("ref") != ref:
            return

        result["completed"] = True
        result["exit_code"] = msg.get("exit_code", -1)
        result_event.set()

    # Create a separate connection to listen for completion
    # (can't reuse _callosum as it may be busy with other events)
    listener = CallosumConnection()
    listener.start(callback=on_message)

    try:
        # Emit request to supervisor with our ref for tracking
        _callosum.emit("supervisor", "request", cmd=cmd, ref=ref)

        # Wait for completion
        if not result_event.wait(timeout=timeout):
            logging.error(f"Timeout waiting for {cmd_name} to complete (ref={ref})")
            day_log(day, f"{cmd_name_log} error timeout")
            return False

        if result["exit_code"] != 0:
            logging.error(
                "Command failed with exit code %s: %s",
                result["exit_code"],
                " ".join(cmd),
            )
            day_log(day, f"{cmd_name_log} error {result['exit_code']}")
            return False

        return True

    finally:
        listener.stop()


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
        cmd = ["sol", "sense", "--day", day]
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

        cmd = ["sol", "insight", day, "-f", insight_data["path"]]
        if segment:
            cmd.extend(["--segment", segment])
        if verbose:
            cmd.append("--verbose")
        if force:
            cmd.append("--force")
        commands.append(cmd)

    # Re-index (light mode: excludes historical days, mtime-cached)
    indexer_cmd = ["sol", "indexer", "--rescan"]
    if verbose:
        indexer_cmd.append("--verbose")
    commands.append(indexer_cmd)

    # Daily-only: journal stats
    if not segment:
        stats_cmd = ["sol", "journal-stats"]
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
    parser.add_argument(
        "--skip-insights",
        action="store_true",
        help="Skip insight processing, run agents only",
    )
    parser.add_argument(
        "--skip-agents",
        action="store_true",
        help="Skip agent processing, run insights only",
    )
    return parser


def check_callosum_available() -> bool:
    """Check if Callosum socket exists (supervisor running).

    Returns True if socket exists at JOURNAL_PATH/health/callosum.sock.
    """
    socket_path = Path(get_journal()) / "health" / "callosum.sock"
    return socket_path.exists()


def wait_for_agents(
    agent_ids: list[str], timeout: int = 600
) -> tuple[list[str], list[str]]:
    """Poll until all agents complete or timeout.

    Polls get_agent_status() every 1 second.

    Args:
        agent_ids: List of agent IDs to wait for
        timeout: Maximum wait time in seconds (default 600 = 10 minutes)

    Returns:
        Tuple of (completed_ids, timed_out_ids)
    """
    start = time.time()
    pending = set(agent_ids)
    completed = []

    while pending and (time.time() - start) < timeout:
        for agent_id in list(pending):
            status = get_agent_status(agent_id)
            if status == "completed":
                completed.append(agent_id)
                pending.discard(agent_id)
            elif status == "not_found":
                # Agent never started - consider failed
                logging.warning(f"Agent {agent_id} not found (never started)")
                pending.discard(agent_id)

        if pending:
            time.sleep(1)

    timed_out = list(pending)
    return completed, timed_out


def run_daily_agents(day: str) -> tuple[int, int]:
    """Run scheduled daily agents grouped by priority.

    Loads agents with schedule="daily", groups by priority field (default 50),
    expands multi_facet agents to one per active non-muted facet, spawns each
    group and waits for completion before proceeding to the next.

    Args:
        day: Day in YYYYMMDD format

    Returns:
        Tuple of (success_count, fail_count)
    """
    # Check callosum availability (warning only - cortex_request will fail if not)
    if not check_callosum_available():
        logging.warning("Callosum socket not found - agents may fail to spawn")

    agents = get_agents()

    # Group agents by priority
    priority_groups: dict[int, list[tuple[str, dict]]] = {}
    for persona_id, config in agents.items():
        if config.get("schedule") == "daily":
            priority = config.get("priority", 50)
            priority_groups.setdefault(priority, []).append((persona_id, config))

    if not priority_groups:
        logging.info("No scheduled daily agents found")
        return (0, 0)

    # Pre-compute shared data for multi-facet agents
    day_formatted = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
    input_summary = day_input_summary(day)
    facets = get_facets()
    enabled_facets = {k: v for k, v in facets.items() if not v.get("muted", False)}
    active_facets = get_active_facets(day)

    # Log muted facets once (applies to all multi_facet agents)
    muted_facets = [k for k, v in facets.items() if v.get("muted", False)]
    if muted_facets:
        logging.info(
            f"Excluding {len(muted_facets)} muted facet(s): {', '.join(muted_facets)}"
        )

    total_agents = sum(len(agents_list) for agents_list in priority_groups.values())
    num_groups = len(priority_groups)

    logging.info(
        f"Running {total_agents} scheduled agents for {day} in {num_groups} priority groups"
    )

    # Emit agents_started event
    emit(
        "agents_started",
        mode="daily",
        day=day,
        count=total_agents,
        groups=num_groups,
    )

    agents_start_time = time.time()
    total_completed = 0
    total_failed = 0

    # Process each priority group in order
    for priority in sorted(priority_groups.keys()):
        agents_list = priority_groups[priority]
        logging.info(f"Starting priority {priority} agents ({len(agents_list)} agents)")

        # Emit group_started event
        emit(
            "group_started",
            mode="daily",
            day=day,
            priority=priority,
            count=len(agents_list),
        )

        spawned_ids = []

        for persona_id, config in agents_list:
            try:
                # Check if this is a multi-facet agent
                if config.get("multi_facet"):
                    always_run = config.get("always", False)

                    for facet_name in enabled_facets.keys():
                        # Skip inactive facets unless agent has always=true
                        if not always_run and facet_name not in active_facets:
                            logging.info(
                                f"Skipping {persona_id} for {facet_name}: "
                                f"no activity on {day_formatted}"
                            )
                            continue

                        logging.info(f"Spawning {persona_id} for facet: {facet_name}")
                        agent_id = cortex_request(
                            prompt=f"Processing facet '{facet_name}' for {day_formatted}: {input_summary}. Use get_facet('{facet_name}') to load context.",
                            persona=persona_id,
                            config={"facet": facet_name},
                        )
                        spawned_ids.append(agent_id)
                        logging.info(
                            f"Started {persona_id} for {facet_name} (ID: {agent_id})"
                        )
                else:
                    # Regular single-instance agent
                    agent_id = cortex_request(
                        prompt=f"Running daily scheduled task for {day_formatted}: {input_summary}.",
                        persona=persona_id,
                    )
                    spawned_ids.append(agent_id)
                    logging.info(f"Started {persona_id} agent (ID: {agent_id})")
            except Exception as e:
                logging.error(f"Failed to spawn {persona_id}: {e}")
                total_failed += 1

        # Wait for this priority group to complete
        group_completed = 0
        group_timed_out = 0
        if spawned_ids:
            logging.info(
                f"Waiting for {len(spawned_ids)} agents in priority {priority}..."
            )
            completed, timed_out = wait_for_agents(spawned_ids, timeout=600)
            group_completed = len(completed)
            group_timed_out = len(timed_out)
            total_completed += group_completed
            total_failed += group_timed_out

            if timed_out:
                logging.warning(
                    f"Priority {priority}: {len(timed_out)} agents timed out: {timed_out}"
                )

        # Emit group_completed event
        emit(
            "group_completed",
            mode="daily",
            day=day,
            priority=priority,
            completed=group_completed,
            timed_out=group_timed_out,
        )

    # Emit agents_completed event
    agents_duration_ms = int((time.time() - agents_start_time) * 1000)
    emit(
        "agents_completed",
        mode="daily",
        day=day,
        success=total_completed,
        failed=total_failed,
        duration_ms=agents_duration_ms,
    )

    logging.info(
        f"Daily agents completed: {total_completed} succeeded, {total_failed} failed"
    )
    return (total_completed, total_failed)


def run_segment_agents(day: str, segment: str) -> int:
    """Spawn segment agents (fire-and-forget).

    Loads agents with schedule="segment" and spawns each with SEGMENT_KEY env var.
    Does NOT wait for completion.

    Args:
        day: Day in YYYYMMDD format
        segment: Segment key in HHMMSS_LEN format

    Returns:
        Number of agents spawned
    """
    agents = get_agents()
    spawned = 0

    for persona_id, config in agents.items():
        if config.get("schedule") == "segment":
            try:
                cortex_request(
                    prompt=f"Processing segment {segment} from {day}. Use available tools to analyze this specific recording window.",
                    persona=persona_id,
                    config={"env": {"SEGMENT_KEY": segment}},
                )
                spawned += 1
                logging.info(f"Spawned segment agent: {persona_id}")
            except Exception as e:
                logging.error(f"Failed to spawn {persona_id}: {e}")

    return spawned


def emit(event: str, **fields) -> None:
    """Emit a dream tract event if callosum is connected."""
    if _callosum:
        _callosum.emit("dream", event, **fields)


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
        start_time = time.time()
        insight_fail_count = 0
        agent_fail_count = 0

        # Determine mode based on segment presence
        mode = "segment" if args.segment else "daily"

        # Build base event fields (mode always, segment only for segment mode)
        def event_fields(**extra):
            fields = {"mode": mode, "day": day}
            if args.segment:
                fields["segment"] = args.segment
            fields.update(extra)
            return fields

        # Emit started event
        emit("started", **event_fields())

        # Phase 1: Insights
        if not args.skip_insights:
            commands = build_commands(
                day, args.force, verbose=args.verbose, segment=args.segment
            )

            # Build command names list for logging
            command_names = [cmd[1] for cmd in commands]
            logging.info(f"Running {len(commands)} insight commands: {command_names}")

            success_count = 0
            for index, cmd in enumerate(commands):
                # Log every command attempt
                day_log(day, f"starting: {' '.join(cmd)}")

                # Emit command event
                emit(
                    "command",
                    **event_fields(command=cmd[1], index=index, total=len(commands)),
                )

                # Route indexer commands through supervisor queue for serialization
                is_indexer = cmd[0] == "sol" and len(cmd) > 1 and cmd[1] == "indexer"
                if is_indexer:
                    success = run_queued_command(cmd, day)
                else:
                    success = run_command(cmd, day)

                if success:
                    success_count += 1
                else:
                    insight_fail_count += 1

            # Emit insights_completed event
            emit(
                "insights_completed",
                **event_fields(
                    success=success_count,
                    failed=insight_fail_count,
                    duration_ms=int((time.time() - start_time) * 1000),
                ),
            )

            logging.info(
                f"Insights completed: {success_count} succeeded, {insight_fail_count} failed"
            )

            # Exit early if insights failed and agents are requested
            if insight_fail_count > 0 and not args.skip_agents:
                logging.error("Insights failed, skipping agents")
                emit(
                    "completed",
                    **event_fields(
                        insight_failed=insight_fail_count,
                        agent_failed=0,
                        duration_ms=int((time.time() - start_time) * 1000),
                    ),
                )
                day_log(day, f"dream insights failed {insight_fail_count}")
                sys.exit(1)

        # Phase 2: Agents
        if not args.skip_agents:
            if args.segment:
                # Segment mode: fire-and-forget
                spawned = run_segment_agents(day, args.segment)
                logging.info(f"Spawned {spawned} segment agents")
            else:
                # Daily mode: priority groups with waiting
                agent_success, agent_fail_count = run_daily_agents(day)

                # Full rescan after agents (via supervisor queue for serialization)
                if agent_success > 0 or agent_fail_count > 0:
                    logging.info("Running full index rescan after agents...")
                    run_queued_command(["sol", "indexer", "--rescan-full"], day)

        # Emit completed event (all processing done)
        emit(
            "completed",
            **event_fields(
                insight_failed=insight_fail_count,
                agent_failed=agent_fail_count,
                duration_ms=int((time.time() - start_time) * 1000),
            ),
        )

        # Build log message
        msg = "dream"
        if args.skip_insights:
            msg += " --skip-insights"
        if args.skip_agents:
            msg += " --skip-agents"
        if args.force:
            msg += " --force"
        if insight_fail_count:
            msg += f" insights_failed={insight_fail_count}"
        if agent_fail_count:
            msg += f" agents_failed={agent_fail_count}"
        day_log(day, msg)

        # Exit with error if any failures
        if insight_fail_count > 0 or agent_fail_count > 0:
            total_failures = insight_fail_count + agent_fail_count
            logging.error(f"{total_failures} task(s) failed, exiting with error")
            sys.exit(1)
    finally:
        _callosum.stop()


if __name__ == "__main__":
    main()
