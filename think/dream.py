# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from think.callosum import CallosumConnection
from think.cortex_client import cortex_request, wait_for_agents
from think.facets import get_active_facets, get_facets
from think.runner import run_task
from think.utils import (
    day_input_summary,
    day_log,
    day_path,
    get_journal,
    get_muse_configs,
    iso_date,
    setup_cli,
)

# Module-level callosum connection for event emission
_callosum: CallosumConnection | None = None


def run_command(cmd: list[str], day: str) -> bool:
    logging.info("==> %s", " ".join(cmd))
    # Extract command name for logging (e.g., "sol generate" -> "generate")
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


def build_pre_generator_commands(
    day: str, verbose: bool = False, segment: str | None = None
) -> list[list[str]]:
    """Build pre-generator commands (sense repair for daily mode).

    Args:
        day: YYYYMMDD format
        segment: Optional HHMMSS_LEN format (if set, skip sense)
        verbose: Verbose logging
    """
    commands: list[list[str]] = []

    if not segment:
        # Daily-only: repair routines run before generators
        cmd = ["sol", "sense", "--day", day]
        if verbose:
            cmd.append("-v")
        commands.append(cmd)

    return commands


def build_post_generator_commands(
    day: str, verbose: bool = False, segment: str | None = None
) -> list[list[str]]:
    """Build post-generator commands (indexer, journal-stats).

    Args:
        day: YYYYMMDD format
        segment: Optional HHMMSS_LEN format
        verbose: Verbose logging
    """
    commands: list[list[str]] = []

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


def run_generators_via_cortex(
    day: str, force: bool, segment: str | None = None
) -> tuple[int, int]:
    """Run generators via cortex requests sequentially.

    Args:
        day: YYYYMMDD format
        segment: Optional HHMMSS_LEN format

    Returns:
        Tuple of (success_count, fail_count)
    """
    from think.cortex_client import get_agent_end_state

    target_schedule = "segment" if segment else "daily"
    generators = get_muse_configs(
        has_tools=False, has_output=True, schedule=target_schedule
    )

    if not generators:
        logging.info("No generators found for schedule: %s", target_schedule)
        return (0, 0)

    logging.info(
        "Running %d generators for %s via cortex: %s",
        len(generators),
        day,
        list(generators.keys()),
    )

    success_count = 0
    fail_count = 0

    # Run generators sequentially
    for generator_name, generator_data in generators.items():
        logging.info("Starting generator: %s", generator_name)

        # Build config for cortex request
        config = {
            "day": day,
            "output": generator_data.get("output", "md"),
        }
        if segment:
            config["segment"] = segment
        if force:
            config["force"] = True

        try:
            # Spawn via cortex
            agent_id = cortex_request(
                prompt="",  # Generators don't use prompt
                name=generator_name,
                config=config,
            )
            logging.info("Spawned generator %s (ID: %s)", generator_name, agent_id)

            # Wait for completion
            completed, timed_out = wait_for_agents([agent_id], timeout=600)

            if timed_out:
                logging.error(
                    "Generator %s timed out (ID: %s)", generator_name, agent_id
                )
                fail_count += 1
            elif completed:
                # Check if it finished successfully or with error
                end_state = get_agent_end_state(agent_id)
                if end_state == "finish":
                    logging.info("Generator %s completed successfully", generator_name)
                    success_count += 1
                else:
                    logging.error(
                        "Generator %s ended with state: %s", generator_name, end_state
                    )
                    fail_count += 1
            else:
                logging.error("Generator %s did not complete", generator_name)
                fail_count += 1

        except Exception as e:
            logging.error("Failed to run generator %s: %s", generator_name, e)
            fail_count += 1

    return (success_count, fail_count)


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
        "--skip-generators",
        action="store_true",
        help="Skip generator processing, run agents only",
    )
    parser.add_argument(
        "--skip-agents",
        action="store_true",
        help="Skip agent processing, run generators only",
    )
    parser.add_argument(
        "--run",
        metavar="NAME",
        help="Run a single prompt by name (e.g., 'activity', 'timeline')",
    )
    parser.add_argument(
        "--facet",
        metavar="NAME",
        help="Target a specific facet (only used with --run for multi-facet agents)",
    )
    return parser


def check_callosum_available() -> bool:
    """Check if Callosum socket exists (supervisor running).

    Returns True if socket exists at JOURNAL_PATH/health/callosum.sock.
    """
    socket_path = Path(get_journal()) / "health" / "callosum.sock"
    return socket_path.exists()


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

    agents = get_muse_configs(has_tools=True)

    # Group agents by priority
    priority_groups: dict[int, list[tuple[str, dict]]] = {}
    for agent_name, config in agents.items():
        if config.get("schedule") == "daily":
            priority = config.get("priority", 50)
            priority_groups.setdefault(priority, []).append((agent_name, config))

    if not priority_groups:
        logging.info("No scheduled daily agents found")
        return (0, 0)

    # Pre-compute shared data for multi-facet agents
    day_formatted = iso_date(day)
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

        for agent_name, config in agents_list:
            try:
                # Check if this is a multi-facet agent
                if config.get("multi_facet"):
                    always_run = config.get("always", False)

                    for facet_name in enabled_facets.keys():
                        # Skip inactive facets unless agent has always=true
                        if not always_run and facet_name not in active_facets:
                            logging.info(
                                f"Skipping {agent_name} for {facet_name}: "
                                f"no activity on {day_formatted}"
                            )
                            continue

                        logging.info(f"Spawning {agent_name} for facet: {facet_name}")
                        agent_id = cortex_request(
                            prompt=f"Processing facet '{facet_name}' for {day_formatted}: {input_summary}. Use get_facet('{facet_name}') to load context.",
                            name=agent_name,
                            config={"facet": facet_name},
                        )
                        spawned_ids.append(agent_id)
                        logging.info(
                            f"Started {agent_name} for {facet_name} (ID: {agent_id})"
                        )
                else:
                    # Regular single-instance agent
                    agent_id = cortex_request(
                        prompt=f"Running daily scheduled task for {day_formatted}: {input_summary}.",
                        name=agent_name,
                    )
                    spawned_ids.append(agent_id)
                    logging.info(f"Started {agent_name} agent (ID: {agent_id})")
            except Exception as e:
                logging.error(f"Failed to spawn {agent_name}: {e}")
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
    agents = get_muse_configs(has_tools=True)
    spawned = 0

    for agent_name, config in agents.items():
        if config.get("schedule") == "segment":
            try:
                cortex_request(
                    prompt=f"Processing segment {segment} from {day}. Use available tools to analyze this specific recording window.",
                    name=agent_name,
                    config={"segment": segment, "env": {"SEGMENT_KEY": segment}},
                )
                spawned += 1
                logging.info(f"Spawned segment agent: {agent_name}")
            except Exception as e:
                logging.error(f"Failed to spawn {agent_name}: {e}")

    return spawned


def run_single_prompt(
    day: str,
    name: str,
    segment: str | None = None,
    force: bool = False,
    facet: str | None = None,
) -> bool:
    """Run a single prompt (generator or agent) by name.

    Args:
        day: Day in YYYYMMDD format
        name: Prompt name from muse/*.md (e.g., 'activity', 'timeline')
        segment: Optional segment key in HHMMSS_LEN format
        force: Whether to regenerate existing output
        facet: Optional facet name for multi-facet agents

    Returns:
        True if successful, False if failed
    """
    from think.cortex_client import get_agent_end_state

    # Load all configs to find the prompt
    all_configs = get_muse_configs(include_disabled=True)

    if name not in all_configs:
        logging.error(f"Prompt not found: {name}")
        logging.info(f"Available prompts: {', '.join(sorted(all_configs.keys()))}")
        return False

    config = all_configs[name]

    # Check if disabled
    if config.get("disabled"):
        logging.warning(f"Prompt '{name}' is disabled")
        return False

    # Determine if this is a generator (has output, no tools) or agent (has tools)
    has_tools = bool(config.get("tools"))
    has_output = bool(config.get("output"))
    is_generator = has_output and not has_tools

    # Validate segment compatibility with schedule
    prompt_schedule = config.get("schedule")
    if segment and prompt_schedule == "daily":
        logging.error(
            f"'{name}' is a daily prompt (schedule='daily'), "
            "but --segment was specified. Remove --segment to run this prompt."
        )
        return False
    if not segment and prompt_schedule == "segment":
        logging.error(
            f"'{name}' is a segment prompt (schedule='segment'), "
            "but no --segment was specified. Add --segment HHMMSS_LEN to run this prompt."
        )
        return False

    # Validate facet usage
    if facet and not config.get("multi_facet"):
        logging.warning(f"'{name}' is not a multi-facet agent, --facet will be ignored")
        facet = None

    day_formatted = iso_date(day)

    if is_generator:
        # Run as generator
        logging.info(f"Running generator: {name}")

        request_config = {
            "day": day,
            "output": config.get("output", "md"),
        }
        if segment:
            request_config["segment"] = segment
        if force:
            request_config["force"] = True

        try:
            agent_id = cortex_request(
                prompt="",  # Generators don't use prompt
                name=name,
                config=request_config,
            )
            logging.info(f"Spawned generator {name} (ID: {agent_id})")

            # Wait for completion
            completed, timed_out = wait_for_agents([agent_id], timeout=600)

            if timed_out:
                logging.error(f"Generator {name} timed out (ID: {agent_id})")
                return False

            end_state = get_agent_end_state(agent_id)
            if end_state == "finish":
                logging.info(f"Generator {name} completed successfully")
                day_log(day, f"dream --run {name} ok")
                return True
            else:
                logging.error(f"Generator {name} ended with state: {end_state}")
                return False

        except Exception as e:
            logging.error(f"Failed to run generator {name}: {e}")
            return False

    else:
        # Run as agent
        logging.info(f"Running agent: {name}")

        input_summary = day_input_summary(day)
        spawned_ids = []

        if config.get("multi_facet"):
            # Multi-facet agent - run for specific facet or all active facets
            facets_data = get_facets()
            enabled_facets = {
                k: v for k, v in facets_data.items() if not v.get("muted", False)
            }
            active_facets = get_active_facets(day)
            always_run = config.get("always", False)

            if facet:
                # Run for specific facet
                if facet not in enabled_facets:
                    logging.error(f"Facet '{facet}' not found or is muted")
                    return False
                target_facets = [facet]
            else:
                # Run for all active facets (or all if always=true)
                target_facets = [
                    f for f in enabled_facets.keys() if always_run or f in active_facets
                ]

            if not target_facets:
                logging.warning(f"No active facets for {name} on {day_formatted}")
                return True  # Not a failure, just nothing to do

            for facet_name in target_facets:
                try:
                    logging.info(f"Spawning {name} for facet: {facet_name}")
                    agent_id = cortex_request(
                        prompt=f"Processing facet '{facet_name}' for {day_formatted}: {input_summary}. Use get_facet('{facet_name}') to load context.",
                        name=name,
                        config={"facet": facet_name},
                    )
                    spawned_ids.append(agent_id)
                    logging.info(f"Started {name} for {facet_name} (ID: {agent_id})")
                except Exception as e:
                    logging.error(f"Failed to spawn {name} for {facet_name}: {e}")

        else:
            # Regular single-instance agent
            try:
                request_config = {}
                if segment:
                    request_config["segment"] = segment
                    request_config["env"] = {"SEGMENT_KEY": segment}

                agent_id = cortex_request(
                    prompt=f"Running task for {day_formatted}: {input_summary}.",
                    name=name,
                    config=request_config if request_config else None,
                )
                spawned_ids.append(agent_id)
                logging.info(f"Started {name} agent (ID: {agent_id})")
            except Exception as e:
                logging.error(f"Failed to spawn {name}: {e}")
                return False

        if not spawned_ids:
            return False

        # Wait for all spawned agents
        logging.info(f"Waiting for {len(spawned_ids)} agent(s)...")
        completed, timed_out = wait_for_agents(spawned_ids, timeout=600)

        if timed_out:
            logging.warning(f"{len(timed_out)} agent(s) timed out: {timed_out}")

        # Check end states for completed agents
        error_count = 0
        for agent_id in completed:
            end_state = get_agent_end_state(agent_id)
            if end_state == "error":
                logging.error(f"Agent {agent_id} ended with error")
                error_count += 1

        success = len(completed) > 0 and len(timed_out) == 0 and error_count == 0
        if success:
            day_log(day, f"dream --run {name} ok")
        elif error_count > 0:
            logging.error(f"{error_count} agent(s) ended with errors")
        return success


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

    # Validate --run is mutually exclusive with --skip-generators/--skip-agents
    if args.run and (args.skip_generators or args.skip_agents):
        parser.error("--run cannot be used with --skip-generators or --skip-agents")

    # Validate --facet requires --run
    if args.facet and not args.run:
        parser.error("--facet requires --run")

    # Handle single prompt execution mode
    if args.run:
        # Start callosum for cortex communication
        _callosum = CallosumConnection()
        _callosum.start()
        try:
            success = run_single_prompt(
                day=day,
                name=args.run,
                segment=args.segment,
                force=args.force,
                facet=args.facet,
            )
            sys.exit(0 if success else 1)
        finally:
            _callosum.stop()

    # Start callosum connection for event emission
    _callosum = CallosumConnection()
    _callosum.start()

    try:
        start_time = time.time()
        generator_fail_count = 0
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

        # Phase 1: Generators (pre-commands, generators via cortex, post-commands)
        if not args.skip_generators:
            # Run pre-generator commands (e.g., sense repair)
            pre_commands = build_pre_generator_commands(
                day, verbose=args.verbose, segment=args.segment
            )
            for cmd in pre_commands:
                day_log(day, f"starting: {' '.join(cmd)}")
                if not run_command(cmd, day):
                    generator_fail_count += 1

            # Run generators via cortex
            gen_success, gen_fail = run_generators_via_cortex(
                day, args.force, segment=args.segment
            )
            generator_fail_count += gen_fail

            # Run post-generator commands (indexer, journal-stats)
            post_commands = build_post_generator_commands(
                day, verbose=args.verbose, segment=args.segment
            )
            for index, cmd in enumerate(post_commands):
                day_log(day, f"starting: {' '.join(cmd)}")

                # Emit command event
                emit(
                    "command",
                    **event_fields(
                        command=cmd[1], index=index, total=len(post_commands)
                    ),
                )

                # Route indexer commands through supervisor queue for serialization
                is_indexer = cmd[0] == "sol" and len(cmd) > 1 and cmd[1] == "indexer"
                if is_indexer:
                    success = run_queued_command(cmd, day)
                else:
                    success = run_command(cmd, day)

                if not success:
                    generator_fail_count += 1

            # Emit generators_completed event
            emit(
                "generators_completed",
                **event_fields(
                    success=gen_success,
                    failed=generator_fail_count,
                    duration_ms=int((time.time() - start_time) * 1000),
                ),
            )

            logging.info(
                f"Generators completed: {gen_success} succeeded, {generator_fail_count} failed"
            )

            # Exit early if generators failed and agents are requested
            if generator_fail_count > 0 and not args.skip_agents:
                logging.error("Generators failed, skipping agents")
                emit(
                    "completed",
                    **event_fields(
                        generator_failed=generator_fail_count,
                        agent_failed=0,
                        duration_ms=int((time.time() - start_time) * 1000),
                    ),
                )
                day_log(day, f"dream generators failed {generator_fail_count}")
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
                    full_rescan_cmd = ["sol", "indexer", "--rescan-full"]
                    if args.verbose:
                        full_rescan_cmd.append("--verbose")
                    run_queued_command(full_rescan_cmd, day, timeout=3600)

        # Emit completed event (all processing done)
        emit(
            "completed",
            **event_fields(
                generator_failed=generator_fail_count,
                agent_failed=agent_fail_count,
                duration_ms=int((time.time() - start_time) * 1000),
            ),
        )

        # Build log message
        msg = "dream"
        if args.skip_generators:
            msg += " --skip-generators"
        if args.skip_agents:
            msg += " --skip-agents"
        if args.force:
            msg += " --force"
        if generator_fail_count:
            msg += f" generators_failed={generator_fail_count}"
        if agent_fail_count:
            msg += f" agents_failed={agent_fail_count}"
        day_log(day, msg)

        # Exit with error if any failures
        if generator_fail_count > 0 or agent_fail_count > 0:
            total_failures = generator_fail_count + agent_fail_count
            logging.error(f"{total_failures} task(s) failed, exiting with error")
            sys.exit(1)
    finally:
        _callosum.stop()


if __name__ == "__main__":
    main()
