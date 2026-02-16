# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unified prompt execution pipeline for solstone.

Runs all scheduled prompts (generators and agents) in priority order.
Lower priority numbers run first. All prompts at the same priority
run in parallel, then dream waits for completion before the next group.
"""

import argparse
import logging
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from think.activities import get_activity_output_path, load_activity_records
from think.callosum import CallosumConnection
from think.cluster import cluster_segments
from think.cortex_client import cortex_request, wait_for_agents
from think.facets import (
    get_active_facets,
    get_enabled_facets,
    get_facets,
    load_segment_facets,
)
from think.muse import get_muse_configs, get_output_path
from think.runner import run_task
from think.utils import (
    day_input_summary,
    day_log,
    day_path,
    get_journal,
    get_rev,
    iso_date,
    setup_cli,
)

# Module-level callosum connection for event emission
_callosum: CallosumConnection | None = None
# Status tracking for periodic status emission
_status: dict = {}
_status_lock = threading.Lock()
_stop_status = threading.Event()


def _update_status(**fields) -> None:
    """Update shared status dict (thread-safe)."""
    with _status_lock:
        _status.update(fields)


def _clear_status() -> None:
    """Clear shared status dict (thread-safe)."""
    with _status_lock:
        _status.clear()


def _emit_periodic_status() -> None:
    """Emit dream.status every 5 seconds while active (runs in daemon thread)."""
    while not _stop_status.is_set():
        _stop_status.wait(5)
        if _stop_status.is_set():
            break
        try:
            with _status_lock:
                snapshot = dict(_status) if _status else None
            if snapshot:
                emit("status", **snapshot)
        except Exception:
            logging.debug("Status emission failed", exc_info=True)


def run_command(cmd: list[str], day: str) -> bool:
    """Run a shell command synchronously."""
    logging.info("==> %s", " ".join(cmd))
    cmd_name = cmd[1] if cmd[0] == "sol" else cmd[0]
    cmd_name = cmd_name.replace("-", "_")

    try:
        success, exit_code, _log_path = run_task(cmd, day=day)
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
    """Run a command through supervisor's task queue and wait for completion."""
    import uuid

    cmd_name = cmd[1] if cmd[0] == "sol" else cmd[0]
    cmd_name_log = cmd_name.replace("-", "_")
    ref = f"dream-{uuid.uuid4().hex[:8]}"

    logging.info("==> %s (queued, ref=%s)", " ".join(cmd), ref)

    if not _callosum:
        logging.error("Callosum not connected, cannot queue command")
        day_log(day, f"{cmd_name_log} error no_callosum")
        return False

    result = {"completed": False, "exit_code": None}
    result_event = threading.Event()

    def on_message(msg: dict) -> None:
        if msg.get("tract") != "supervisor":
            return
        if msg.get("event") != "stopped":
            return
        if msg.get("ref") != ref:
            return
        result["completed"] = True
        result["exit_code"] = msg.get("exit_code", -1)
        result_event.set()

    listener = CallosumConnection()
    listener.start(callback=on_message)

    try:
        _callosum.emit("supervisor", "request", cmd=cmd, ref=ref, day=day)

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


def emit(event: str, **fields) -> None:
    """Emit a dream tract event if callosum is connected."""
    if _callosum:
        _callosum.emit("dream", event, **fields)


def check_callosum_available() -> bool:
    """Check if Callosum socket exists (supervisor running)."""
    socket_path = Path(get_journal()) / "health" / "callosum.sock"
    return socket_path.exists()


_SEND_RETRY_DELAYS = (0.5, 1.0)  # seconds between retries (3 attempts total)


def _cortex_request_with_retry(**kwargs) -> str | None:
    """Call cortex_request with retries on Callosum send failure.

    Retries up to len(_SEND_RETRY_DELAYS) times with short sleeps in between.
    Returns the agent_id on success, or None if all attempts failed.
    """
    agent_id = cortex_request(**kwargs)
    if agent_id is not None:
        return agent_id

    name = kwargs.get("name", "unknown")
    for i, delay in enumerate(_SEND_RETRY_DELAYS, 1):
        logging.warning("Retrying cortex request for '%s' (attempt %d)", name, i + 1)
        time.sleep(delay)
        agent_id = cortex_request(**kwargs)
        if agent_id is not None:
            return agent_id

    logging.error("All cortex request attempts failed for '%s'", name)
    return None


def _drain_priority_batch(
    spawned: list[tuple[str, str, dict, str | None]],
    target_schedule: str,
    day: str,
    segment: str | None,
    stream: str | None = None,
) -> tuple[int, int, list[str]]:
    """Wait for a batch of spawned agents and process their results.

    Waits for all agents in the batch to complete, checks end states,
    emits completion events, and runs incremental indexing for generators.

    Args:
        spawned: List of (agent_id, prompt_name, config, facet) tuples
        target_schedule: "segment" or "daily"
        day: Day in YYYYMMDD format
        segment: Optional segment key
        stream: Optional stream name

    Returns:
        Tuple of (success_count, failed_count, failed_names) where
        failed_names contains descriptions like "digest (error)" or
        "recap/work (timeout)".
    """
    if not spawned:
        return (0, 0, [])

    agent_ids = [agent_id for agent_id, _, _, _ in spawned]
    logging.info(f"Waiting for {len(agent_ids)} agents...")

    completed, timed_out = wait_for_agents(agent_ids, timeout=600)

    success = 0
    failed = 0
    failed_names: list[str] = []

    if timed_out:
        logging.warning(f"{len(timed_out)} agents timed out: {timed_out}")
        failed += len(timed_out)
        for agent_id in timed_out:
            timed_name = next(
                (n for aid, n, _, _ in spawned if aid == agent_id), "unknown"
            )
            timed_facet = next((f for aid, _, _, f in spawned if aid == agent_id), None)
            label = f"{timed_name}/{timed_facet}" if timed_facet else timed_name
            failed_names.append(f"{label} (timeout)")
            emit(
                "agent_completed",
                mode=target_schedule,
                day=day,
                segment=segment,
                name=timed_name,
                agent_id=agent_id,
                state="timeout",
                **({"facet": timed_facet} if timed_facet else {}),
            )

    for agent_id, prompt_name, config, agent_facet in spawned:
        if agent_id in timed_out:
            continue

        end_state = completed.get(agent_id, "unknown")
        if end_state == "finish":
            logging.info(f"{prompt_name} completed successfully")
            success += 1
            emit(
                "agent_completed",
                mode=target_schedule,
                day=day,
                segment=segment,
                name=prompt_name,
                agent_id=agent_id,
                state="finish",
                **({"facet": agent_facet} if agent_facet else {}),
            )

            # Incremental indexing for generators (skip JSON —
            # structured metadata not suitable for full-text index)
            is_generate = config["type"] == "generate"
            output_format = config.get("output", "md")
            if is_generate and output_format != "json":
                output_path = get_output_path(
                    day_path(day),
                    prompt_name,
                    segment=segment,
                    output_format=output_format,
                    stream=stream,
                )

                if output_path.exists():
                    logging.debug(f"Indexing {output_path}")
                    run_queued_command(
                        ["sol", "indexer", "--rescan-file", str(output_path)],
                        day,
                        timeout=60,
                    )
        else:
            label = f"{prompt_name}/{agent_facet}" if agent_facet else prompt_name
            logging.error(f"{label} ended with state: {end_state}")
            failed += 1
            failed_names.append(f"{label} ({end_state})")
            emit(
                "agent_completed",
                mode=target_schedule,
                day=day,
                segment=segment,
                name=prompt_name,
                agent_id=agent_id,
                state=end_state,
                **({"facet": agent_facet} if agent_facet else {}),
            )

    return (success, failed, failed_names)


def run_prompts_by_priority(
    day: str,
    segment: str | None,
    refresh: bool,
    verbose: bool,
    max_concurrency: int = 2,
    stream: str | None = None,
) -> tuple[int, int, list[str]]:
    """Run all scheduled prompts in priority order.

    Loads all prompts for the target schedule, groups by priority, and executes
    each group with bounded concurrency. Waits for completion before proceeding
    to the next priority group. For generators (prompts with output), runs
    incremental indexing after each completes.

    Args:
        day: Day in YYYYMMDD format
        segment: Optional segment key in HHMMSS_LEN format
        refresh: Whether to regenerate existing outputs
        verbose: Verbose logging
        max_concurrency: Max agents to run concurrently per priority group.
            0 means unlimited (all agents in a group run in parallel).

    Returns:
        Tuple of (success_count, fail_count, failed_names) where
        failed_names contains descriptions like "digest (error)".
    """
    target_schedule = "segment" if segment else "daily"

    # Load ALL scheduled prompts (both generators and agents)
    all_prompts = get_muse_configs(schedule=target_schedule)

    if not all_prompts:
        logging.info(f"No prompts found for schedule: {target_schedule}")
        return (0, 0, [])

    # Group prompts by priority
    priority_groups: dict[int, list[tuple[str, dict]]] = {}
    for name, config in all_prompts.items():
        priority = config["priority"]  # Required field, validated by get_muse_configs
        priority_groups.setdefault(priority, []).append((name, config))

    # Pre-compute shared data for multi-facet prompts
    day_formatted = iso_date(day)
    input_summary = day_input_summary(day)
    enabled_facets = get_enabled_facets()

    if not segment:
        # Daily mode: use facets with activity on this day (stable for the run)
        active_facets = get_active_facets(day)

    total_prompts = sum(len(prompts) for prompts in priority_groups.values())
    num_groups = len(priority_groups)
    _update_status(
        mode=target_schedule,
        day=day,
        segment=segment,
        stream=stream,
        agents_total=total_prompts,
        agents_completed=0,
        current_agents=[],
    )

    logging.info(
        f"Running {total_prompts} prompts for {day} in {num_groups} priority groups"
    )

    emit(
        "started",
        mode=target_schedule,
        day=day,
        segment=segment,
        count=total_prompts,
        groups=num_groups,
    )

    start_time = time.time()
    total_success = 0
    total_failed = 0
    all_failed_names: list[str] = []

    # Process each priority group in order
    for priority in sorted(priority_groups.keys()):
        prompts_list = priority_groups[priority]
        _update_status(current_group_priority=priority)
        logging.info(f"Starting priority {priority} ({len(prompts_list)} prompts)")

        emit(
            "group_started",
            mode=target_schedule,
            day=day,
            segment=segment,
            priority=priority,
            count=len(prompts_list),
        )

        # Segment mode: reload active facets each group since earlier groups
        # (e.g., facets generator at priority 90) may have written facets.json
        if segment:
            raw_facets = load_segment_facets(day, segment, stream=stream)
            active_facets = set(f for f in raw_facets if f in enabled_facets)

        spawned: list[tuple[str, str, dict, str | None]] = (
            []
        )  # (agent_id, name, config, facet)
        group_success = 0
        group_failed = 0

        for prompt_name, config in prompts_list:
            is_generate = config["type"] == "generate"

            try:
                if config.get("multi_facet"):
                    always_run = config.get("always", False)

                    for facet_name in enabled_facets.keys():
                        if not always_run and facet_name not in active_facets:
                            logging.info(
                                f"Skipping {prompt_name} for {facet_name}: "
                                f"no activity on {day_formatted}"
                            )
                            continue

                        logging.info(f"Spawning {prompt_name} for facet: {facet_name}")

                        # Always pass day for instructions.day context
                        request_config: dict = {"facet": facet_name, "day": day}
                        if is_generate:
                            request_config["output"] = config.get("output", "md")
                            if refresh:
                                request_config["refresh"] = True
                        env: dict[str, str] = {
                            "SOL_DAY": day,
                            "SOL_FACET": facet_name,
                        }
                        if segment:
                            request_config["segment"] = segment
                            env["SOL_SEGMENT"] = segment
                            if stream:
                                env["SOL_STREAM"] = stream
                        request_config["env"] = env

                        prompt = (
                            ""
                            if is_generate
                            else f"Processing facet '{facet_name}' for {day_formatted}: {input_summary}. Use get_facet('{facet_name}') to load context."
                        )

                        agent_id = _cortex_request_with_retry(
                            prompt=prompt,
                            name=prompt_name,
                            config=request_config,
                        )
                        if agent_id is None:
                            group_failed += 1
                            all_failed_names.append(
                                f"{prompt_name}/{facet_name} (send)"
                            )
                            continue
                        spawned.append((agent_id, prompt_name, config, facet_name))
                        emit(
                            "agent_started",
                            mode=target_schedule,
                            day=day,
                            segment=segment,
                            name=prompt_name,
                            agent_id=agent_id,
                            facet=facet_name,
                        )
                        logging.info(
                            f"Started {prompt_name} for {facet_name} (ID: {agent_id})"
                        )

                        # Drain batch when concurrency limit reached
                        if max_concurrency and len(spawned) >= max_concurrency:
                            _update_status(
                                current_agents=[name for _, name, _, _ in spawned]
                            )
                            s, f, fn = _drain_priority_batch(
                                spawned, target_schedule, day, segment, stream
                            )
                            group_success += s
                            group_failed += f
                            all_failed_names.extend(fn)
                            spawned = []
                            _update_status(
                                agents_completed=total_success
                                + total_failed
                                + group_success
                                + group_failed,
                                current_agents=[],
                            )
                else:
                    # Regular single-instance prompt
                    logging.info(f"Spawning {prompt_name}")

                    # Always pass day for instructions.day context
                    request_config: dict = {"day": day}
                    if is_generate:
                        request_config["output"] = config.get("output", "md")
                        if refresh:
                            request_config["refresh"] = True
                    env: dict[str, str] = {"SOL_DAY": day}
                    if segment:
                        request_config["segment"] = segment
                        env["SOL_SEGMENT"] = segment
                        if stream:
                            env["SOL_STREAM"] = stream
                    request_config["env"] = env

                    prompt = (
                        ""
                        if is_generate
                        else f"Running scheduled task for {day_formatted}: {input_summary}."
                    )

                    agent_id = _cortex_request_with_retry(
                        prompt=prompt,
                        name=prompt_name,
                        config=request_config,
                    )
                    if agent_id is None:
                        group_failed += 1
                        all_failed_names.append(f"{prompt_name} (send)")
                        continue
                    spawned.append((agent_id, prompt_name, config, None))
                    emit(
                        "agent_started",
                        mode=target_schedule,
                        day=day,
                        segment=segment,
                        name=prompt_name,
                        agent_id=agent_id,
                    )
                    logging.info(f"Started {prompt_name} (ID: {agent_id})")

                    # Drain batch when concurrency limit reached
                    if max_concurrency and len(spawned) >= max_concurrency:
                        _update_status(
                            current_agents=[name for _, name, _, _ in spawned]
                        )
                        s, f, fn = _drain_priority_batch(
                            spawned, target_schedule, day, segment
                        )
                        group_success += s
                        group_failed += f
                        all_failed_names.extend(fn)
                        spawned = []
                        _update_status(
                            agents_completed=total_success
                            + total_failed
                            + group_success
                            + group_failed,
                            current_agents=[],
                        )

            except Exception as e:
                logging.error(f"Failed to spawn {prompt_name}: {e}")
                group_failed += 1
                all_failed_names.append(f"{prompt_name} (spawn)")

        # Drain any remaining agents in this priority group
        _update_status(current_agents=[name for _, name, _, _ in spawned])
        s, f, fn = _drain_priority_batch(spawned, target_schedule, day, segment, stream)
        group_success += s
        group_failed += f
        all_failed_names.extend(fn)
        _update_status(
            agents_completed=total_success
            + total_failed
            + group_success
            + group_failed,
            current_agents=[],
        )

        total_success += group_success
        total_failed += group_failed

        emit(
            "group_completed",
            mode=target_schedule,
            day=day,
            segment=segment,
            priority=priority,
            success=group_success,
            failed=group_failed,
        )

    duration_ms = int((time.time() - start_time) * 1000)
    emit(
        "completed",
        mode=target_schedule,
        day=day,
        segment=segment,
        success=total_success,
        failed=total_failed,
        failed_names=all_failed_names,
        duration_ms=duration_ms,
    )

    logging.info(f"Prompts completed: {total_success} succeeded, {total_failed} failed")
    return (total_success, total_failed, all_failed_names)


def run_single_prompt(
    day: str,
    name: str,
    segment: str | None = None,
    refresh: bool = False,
    facet: str | None = None,
    stream: str | None = None,
) -> bool:
    """Run a single prompt (generator or agent) by name.

    Args:
        day: Day in YYYYMMDD format
        name: Prompt name from muse/*.md (e.g., 'activity', 'timeline')
        segment: Optional segment key in HHMMSS_LEN format
        refresh: Whether to regenerate existing output
        facet: Optional facet name for multi-facet agents

    Returns:
        True if successful, False if failed
    """
    # Load all configs to find the prompt
    all_configs = get_muse_configs(include_disabled=True)

    if name not in all_configs:
        logging.error(f"Prompt not found: {name}")
        logging.info(f"Available prompts: {', '.join(sorted(all_configs.keys()))}")
        return False

    config = all_configs[name]

    if config.get("disabled"):
        logging.warning(f"Prompt '{name}' is disabled")
        return False

    is_generate = config["type"] == "generate"

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

    if facet and not config.get("multi_facet"):
        logging.warning(f"'{name}' is not a multi-facet agent, --facet will be ignored")
        facet = None

    day_formatted = iso_date(day)

    if is_generate:
        logging.info(f"Running generator: {name}")

        request_config = {
            "day": day,
            "output": config.get("output", "md"),
        }
        if segment:
            request_config["segment"] = segment
        if refresh:
            request_config["refresh"] = True

        try:
            agent_id = _cortex_request_with_retry(
                prompt="",
                name=name,
                config=request_config,
            )
            if agent_id is None:
                logging.error(f"Failed to send cortex request for generator '{name}'")
                return False
            logging.info(f"Spawned generator {name} (ID: {agent_id})")
            emit(
                "agent_started",
                day=day,
                segment=segment,
                name=name,
                agent_id=agent_id,
            )

            completed, timed_out = wait_for_agents([agent_id], timeout=600)

            if timed_out:
                logging.error(f"Generator {name} timed out (ID: {agent_id})")
                emit(
                    "agent_completed",
                    day=day,
                    segment=segment,
                    name=name,
                    agent_id=agent_id,
                    state="timeout",
                )
                return False

            end_state = completed.get(agent_id, "unknown")
            if end_state == "finish":
                logging.info(f"Generator {name} completed successfully")
                emit(
                    "agent_completed",
                    day=day,
                    segment=segment,
                    name=name,
                    agent_id=agent_id,
                    state="finish",
                )
                day_log(day, f"dream --run {name} ok")
                return True
            else:
                logging.error(f"Generator {name} ended with state: {end_state}")
                emit(
                    "agent_completed",
                    day=day,
                    segment=segment,
                    name=name,
                    agent_id=agent_id,
                    state=end_state,
                )
                return False

        except Exception as e:
            logging.error(f"Failed to run generator {name}: {e}")
            return False

    else:
        logging.info(f"Running agent: {name}")

        input_summary = day_input_summary(day)
        spawned_ids: list[tuple[str, str | None]] = []

        if config.get("multi_facet"):
            facets_data = get_facets()
            enabled_facets = {
                k: v for k, v in facets_data.items() if not v.get("muted", False)
            }
            active_facets = get_active_facets(day)
            always_run = config.get("always", False)

            if facet:
                if facet not in enabled_facets:
                    logging.error(f"Facet '{facet}' not found or is muted")
                    return False
                target_facets = [facet]
            else:
                target_facets = [
                    f for f in enabled_facets.keys() if always_run or f in active_facets
                ]

            if not target_facets:
                logging.warning(f"No active facets for {name} on {day_formatted}")
                return True

            for facet_name in target_facets:
                try:
                    logging.info(f"Spawning {name} for facet: {facet_name}")
                    request_config = {"facet": facet_name, "day": day}
                    env: dict[str, str] = {
                        "SOL_DAY": day,
                        "SOL_FACET": facet_name,
                    }
                    if segment:
                        request_config["segment"] = segment
                        env["SOL_SEGMENT"] = segment
                        if stream:
                            env["SOL_STREAM"] = stream
                    request_config["env"] = env
                    agent_id = _cortex_request_with_retry(
                        prompt=f"Processing facet '{facet_name}' for {day_formatted}: {input_summary}. Use get_facet('{facet_name}') to load context.",
                        name=name,
                        config=request_config,
                    )
                    if agent_id is None:
                        logging.error(
                            f"Failed to send cortex request for {name}/{facet_name}"
                        )
                        continue
                    spawned_ids.append((agent_id, facet_name))
                    emit(
                        "agent_started",
                        day=day,
                        segment=segment,
                        name=name,
                        agent_id=agent_id,
                        facet=facet_name,
                    )
                    logging.info(f"Started {name} for {facet_name} (ID: {agent_id})")
                except Exception as e:
                    logging.error(f"Failed to spawn {name} for {facet_name}: {e}")

        else:
            try:
                request_config = {"day": day}
                env: dict[str, str] = {"SOL_DAY": day}
                if segment:
                    request_config["segment"] = segment
                    env["SOL_SEGMENT"] = segment
                    if stream:
                        env["SOL_STREAM"] = stream
                request_config["env"] = env

                agent_id = _cortex_request_with_retry(
                    prompt=f"Running task for {day_formatted}: {input_summary}.",
                    name=name,
                    config=request_config,
                )
                if agent_id is None:
                    logging.error(f"Failed to send cortex request for '{name}'")
                    return False
                spawned_ids.append((agent_id, None))
                emit(
                    "agent_started",
                    day=day,
                    segment=segment,
                    name=name,
                    agent_id=agent_id,
                )
                logging.info(f"Started {name} agent (ID: {agent_id})")
            except Exception as e:
                logging.error(f"Failed to spawn {name}: {e}")
                return False

        if not spawned_ids:
            return False

        logging.info(f"Waiting for {len(spawned_ids)} agent(s)...")
        completed, timed_out = wait_for_agents(
            [agent_id for agent_id, _ in spawned_ids], timeout=600
        )

        if timed_out:
            logging.warning(f"{len(timed_out)} agent(s) timed out: {timed_out}")

        error_count = 0
        for agent_id, agent_facet in spawned_ids:
            if agent_id in timed_out:
                emit(
                    "agent_completed",
                    day=day,
                    segment=segment,
                    name=name,
                    agent_id=agent_id,
                    state="timeout",
                    **({"facet": agent_facet} if agent_facet else {}),
                )
                continue
            end_state = completed.get(agent_id, "unknown")
            emit(
                "agent_completed",
                day=day,
                segment=segment,
                name=name,
                agent_id=agent_id,
                state=end_state,
                **({"facet": agent_facet} if agent_facet else {}),
            )
            if end_state == "error":
                logging.error(f"Agent {agent_id} ended with error")
                error_count += 1

        success = len(completed) > 0 and len(timed_out) == 0 and error_count == 0
        if success:
            day_log(day, f"dream --run {name} ok")
        elif error_count > 0:
            logging.error(f"{error_count} agent(s) ended with errors")
        return success


def run_activity_prompts(
    day: str,
    activity_id: str,
    facet: str,
    refresh: bool = False,
    verbose: bool = False,
    max_concurrency: int = 2,
) -> bool:
    """Run activity-scheduled agents for a completed activity.

    Loads the activity record from the journal, filters agents whose
    schedule="activity" and whose 'activities' list matches the activity type
    (or contains "*"), then spawns each matching agent with the activity's
    segment span for transcript loading.

    Args:
        day: Day in YYYYMMDD format
        activity_id: Activity record ID (e.g., "coding_100000_300")
        facet: Facet name
        refresh: Whether to regenerate existing outputs
        verbose: Verbose logging
        max_concurrency: Max agents to run concurrently (0=unlimited)

    Returns:
        True if all agents succeeded, False if any failed
    """
    # Load activity record
    records = load_activity_records(facet, day)
    record = None
    for r in records:
        if r.get("id") == activity_id:
            record = r
            break

    if not record:
        logging.error(
            "Activity record not found: %s in facet '%s' on %s",
            activity_id,
            facet,
            day,
        )
        return False

    activity_type = record.get("activity", "")
    segments = record.get("segments", [])

    if not segments:
        logging.error("Activity record %s has no segments", activity_id)
        return False

    # Load activity-scheduled agents
    all_prompts = get_muse_configs(schedule="activity")

    if not all_prompts:
        logging.info("No activity-scheduled agents found")
        return True

    # Filter agents that match this activity type
    matching = {}
    for name, config in all_prompts.items():
        activities_filter = config.get("activities", [])
        if "*" in activities_filter or activity_type in activities_filter:
            matching[name] = config

    if not matching:
        logging.info(
            "No agents match activity type '%s' (checked %d agents)",
            activity_type,
            len(all_prompts),
        )
        return True

    # Group by priority
    priority_groups: dict[int, list[tuple[str, dict]]] = {}
    for name, config in matching.items():
        priority = config["priority"]
        priority_groups.setdefault(priority, []).append((name, config))

    total_prompts = sum(len(p) for p in priority_groups.values())
    num_groups = len(priority_groups)
    _update_status(
        mode="activity",
        day=day,
        activity=activity_id,
        facet=facet,
        agents_total=total_prompts,
        agents_completed=0,
        current_agents=[],
    )

    logging.info(
        "Running %d activity agents for %s (type=%s, %d segments) in %d groups",
        total_prompts,
        activity_id,
        activity_type,
        len(segments),
        num_groups,
    )

    emit(
        "started",
        mode="activity",
        day=day,
        activity=activity_id,
        facet=facet,
        count=total_prompts,
        groups=num_groups,
    )

    start_time = time.time()
    total_success = 0
    total_failed = 0

    day_formatted = iso_date(day)

    for priority in sorted(priority_groups.keys()):
        prompts_list = priority_groups[priority]
        _update_status(current_group_priority=priority)
        logging.info(f"Starting priority {priority} ({len(prompts_list)} agents)")

        emit(
            "group_started",
            mode="activity",
            day=day,
            activity=activity_id,
            facet=facet,
            priority=priority,
            count=len(prompts_list),
        )

        spawned: list[tuple[str, str, dict]] = []  # (agent_id, name, config)
        group_success = 0
        group_failed = 0

        def _drain_activity_batch() -> None:
            """Wait for current batch of spawned activity agents."""
            nonlocal spawned, group_success, group_failed
            if not spawned:
                return

            agent_ids = [aid for aid, _, _ in spawned]
            logging.info(f"Waiting for {len(agent_ids)} agents...")

            completed, timed_out = wait_for_agents(agent_ids, timeout=600)

            if timed_out:
                logging.warning(f"{len(timed_out)} agents timed out")
                group_failed += len(timed_out)
                for agent_id in timed_out:
                    timed_name = next(
                        (n for aid, n, _ in spawned if aid == agent_id), "unknown"
                    )
                    emit(
                        "agent_completed",
                        mode="activity",
                        day=day,
                        activity=activity_id,
                        facet=facet,
                        name=timed_name,
                        agent_id=agent_id,
                        state="timeout",
                    )

            for agent_id, prompt_name, config in spawned:
                if agent_id in timed_out:
                    continue

                end_state = completed.get(agent_id, "unknown")
                if end_state == "finish":
                    logging.info(f"{prompt_name} completed successfully")
                    group_success += 1

                    # Incremental indexing for generators (skip JSON)
                    is_generate = config["type"] == "generate"
                    output_format = config.get("output", "md")
                    if is_generate and output_format != "json":
                        output_path = get_activity_output_path(
                            facet,
                            day,
                            activity_id,
                            prompt_name,
                            output_format=output_format,
                        )
                        if output_path.exists():
                            logging.debug(f"Indexing {output_path}")
                            run_queued_command(
                                ["sol", "indexer", "--rescan-file", str(output_path)],
                                day,
                                timeout=60,
                            )
                else:
                    logging.error(f"{prompt_name} ended with state: {end_state}")
                    group_failed += 1

                emit(
                    "agent_completed",
                    mode="activity",
                    day=day,
                    activity=activity_id,
                    facet=facet,
                    name=prompt_name,
                    agent_id=agent_id,
                    state=end_state,
                )

            spawned = []

        for prompt_name, config in prompts_list:
            is_generate = config["type"] == "generate"

            try:
                logging.info(f"Spawning {prompt_name} for activity {activity_id}")

                output_format = config.get("output", "md")
                request_config: dict = {
                    "facet": facet,
                    "day": day,
                    "span": segments,
                    "activity": record,
                    "output_path": str(
                        get_activity_output_path(
                            facet,
                            day,
                            activity_id,
                            prompt_name,
                            output_format=output_format,
                        )
                    ),
                    "env": {
                        "SOL_DAY": day,
                        "SOL_FACET": facet,
                        "SOL_ACTIVITY": activity_id,
                    },
                }
                if is_generate:
                    request_config["output"] = output_format
                    if refresh:
                        request_config["refresh"] = True

                prompt = (
                    ""
                    if is_generate
                    else f"Processing activity '{activity_id}' ({activity_type}) in facet '{facet}' for {day_formatted}."
                )

                agent_id = _cortex_request_with_retry(
                    prompt=prompt,
                    name=prompt_name,
                    config=request_config,
                )
                if agent_id is None:
                    total_failed += 1
                    continue
                spawned.append((agent_id, prompt_name, config))
                emit(
                    "agent_started",
                    mode="activity",
                    day=day,
                    activity=activity_id,
                    facet=facet,
                    name=prompt_name,
                    agent_id=agent_id,
                )
                logging.info(f"Started {prompt_name} (ID: {agent_id})")

                # Drain batch when concurrency limit reached
                if max_concurrency and len(spawned) >= max_concurrency:
                    _update_status(current_agents=[name for _, name, _ in spawned])
                    _drain_activity_batch()
                    _update_status(
                        agents_completed=total_success
                        + total_failed
                        + group_success
                        + group_failed,
                        current_agents=[],
                    )

            except Exception as e:
                logging.error(f"Failed to spawn {prompt_name}: {e}")
                total_failed += 1

        # Drain any remaining agents
        _update_status(current_agents=[name for _, name, _ in spawned])
        _drain_activity_batch()
        _update_status(
            agents_completed=total_success
            + total_failed
            + group_success
            + group_failed,
            current_agents=[],
        )

        total_success += group_success
        total_failed += group_failed

        emit(
            "group_completed",
            mode="activity",
            day=day,
            activity=activity_id,
            facet=facet,
            priority=priority,
            success=group_success,
            failed=group_failed,
        )

    duration_ms = int((time.time() - start_time) * 1000)
    emit(
        "completed",
        mode="activity",
        day=day,
        activity=activity_id,
        facet=facet,
        success=total_success,
        failed=total_failed,
        duration_ms=duration_ms,
    )

    logging.info(
        f"Activity agents completed: {total_success} succeeded, {total_failed} failed"
    )

    msg = f"dream --activity {activity_id}"
    if total_failed:
        msg += f" failed={total_failed}"
    day_log(day, msg)

    return total_failed == 0


def run_flush_prompts(
    day: str,
    segment: str,
    verbose: bool,
    stream: str | None = None,
) -> bool:
    """Run flush hooks for segment agents that declare flush support.

    Triggered by supervisor when no new segments arrive after a timeout.
    Only runs agents with hook.flush=true, passing flush=True so their
    pre-hooks can close out dangling state.

    Args:
        day: Day in YYYYMMDD format
        segment: Last observed segment key
        verbose: Verbose logging

    Returns:
        True if all flush agents succeeded, False if any failed
    """
    all_prompts = get_muse_configs(schedule="segment")

    # Filter to only agents with flush hooks
    flush_prompts = {
        name: config
        for name, config in all_prompts.items()
        if isinstance(config.get("hook"), dict) and config["hook"].get("flush")
    }

    if not flush_prompts:
        logging.info("No flush-eligible agents found")
        return True

    logging.info(
        f"Flushing {len(flush_prompts)} agents for {day}/{segment}: "
        f"{', '.join(flush_prompts.keys())}"
    )

    emit("started", mode="flush", day=day, segment=segment, count=len(flush_prompts))
    start_time = time.time()
    total_success = 0
    total_failed = 0

    spawned: list[tuple[str, str, dict]] = []  # (agent_id, name, config)
    _update_status(
        mode="flush",
        day=day,
        segment=segment,
        stream=stream,
        agents_total=len(flush_prompts),
        agents_completed=0,
        current_agents=[],
    )

    for prompt_name, config in flush_prompts.items():
        is_generate = config["type"] == "generate"

        try:
            env: dict[str, str] = {
                "SOL_SEGMENT": segment,
                "SOL_DAY": day,
            }
            if stream:
                env["SOL_STREAM"] = stream
            request_config: dict = {
                "day": day,
                "segment": segment,
                "flush": True,
                "refresh": True,
                "env": env,
            }
            if is_generate:
                request_config["output"] = config.get("output", "md")

            agent_id = _cortex_request_with_retry(
                prompt="",
                name=prompt_name,
                config=request_config,
            )
            if agent_id is None:
                total_failed += 1
                continue
            spawned.append((agent_id, prompt_name, config))
            emit(
                "agent_started",
                mode="flush",
                day=day,
                segment=segment,
                name=prompt_name,
                agent_id=agent_id,
            )
            logging.info(f"Started flush agent {prompt_name} (ID: {agent_id})")

        except Exception as e:
            logging.error(f"Failed to spawn flush agent {prompt_name}: {e}")
            total_failed += 1

    if spawned:
        _update_status(current_agents=[name for _, name, _ in spawned])
        agent_ids = [aid for aid, _, _ in spawned]
        completed, timed_out = wait_for_agents(agent_ids, timeout=600)

        if timed_out:
            logging.warning(f"Flush: {len(timed_out)} agents timed out")
            total_failed += len(timed_out)

        for agent_id, prompt_name, config in spawned:
            if agent_id in timed_out:
                continue
            end_state = completed.get(agent_id, "unknown")
            if end_state == "finish":
                logging.info(f"Flush agent {prompt_name} completed")
                total_success += 1
            else:
                logging.error(
                    f"Flush agent {prompt_name} ended with state: {end_state}"
                )
                total_failed += 1

            emit(
                "agent_completed",
                mode="flush",
                day=day,
                segment=segment,
                name=prompt_name,
                agent_id=agent_id,
                state=end_state,
            )
        _update_status(
            agents_completed=total_success + total_failed,
            current_agents=[],
        )
    if not spawned and total_failed:
        _update_status(agents_completed=total_failed, current_agents=[])

    duration_ms = int((time.time() - start_time) * 1000)
    emit(
        "completed",
        mode="flush",
        day=day,
        segment=segment,
        success=total_success,
        failed=total_failed,
        duration_ms=duration_ms,
    )

    logging.info(
        f"Flush completed in {duration_ms}ms: "
        f"{total_success} succeeded, {total_failed} failed"
    )

    msg = f"dream --flush {segment}"
    if total_failed:
        msg += f" failed={total_failed}"
    day_log(day, msg)

    return total_failed == 0


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
    parser.add_argument(
        "--refresh", action="store_true", help="Refresh existing outputs"
    )
    parser.add_argument(
        "--segments",
        action="store_true",
        help="Re-process all segments for the day (incompatible with --segment, --run, --facet)",
    )
    parser.add_argument(
        "--run",
        metavar="NAME",
        help="Run a single prompt by name (e.g., 'activity', 'timeline')",
    )
    parser.add_argument(
        "--facet",
        metavar="NAME",
        help="Target a specific facet (only used with --run or --activity)",
    )
    parser.add_argument(
        "--activity",
        metavar="ID",
        help="Run activity-scheduled agents for a completed activity record (requires --facet and --day)",
    )
    parser.add_argument(
        "--stream",
        help="Stream name (e.g., 'archon', 'import.apple'). Passed to agents as SOL_STREAM env var.",
    )
    parser.add_argument(
        "--flush",
        action="store_true",
        help="Run flush hooks on segment agents to close out dangling state (requires --segment)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=2,
        metavar="N",
        help="Max concurrent agents per priority group (0=unlimited, default: 2)",
    )
    return parser


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

    if args.facet and not args.run and not args.activity:
        parser.error("--facet requires --run or --activity")

    if args.activity and not args.facet:
        parser.error("--activity requires --facet")

    # Auto-enable refresh for dirty days (full daily runs only)
    if not args.refresh and not args.segment and not args.segments:
        health_dir = day_dir / "health"
        stream_marker = health_dir / "stream.updated"
        daily_marker = health_dir / "daily.updated"
        if stream_marker.is_file() and (
            not daily_marker.is_file()
            or stream_marker.stat().st_mtime > daily_marker.stat().st_mtime
        ):
            args.refresh = True
            logging.info("Day %s has pending stream data, enabling refresh", day)

    if args.activity and not args.day:
        parser.error("--activity requires --day")

    if args.activity and (args.segment or args.run or args.segments or args.flush):
        parser.error(
            "--activity is incompatible with --segment, --run, --segments, and --flush"
        )

    if args.flush and not args.segment:
        parser.error("--flush requires --segment")

    if args.flush and (args.run or args.segments or args.refresh):
        parser.error("--flush is incompatible with --run, --segments, and --refresh")

    if args.segments and (args.segment or args.run or args.facet):
        parser.error("--segments is incompatible with --segment, --run, and --facet")

    # Start callosum connection
    _callosum = CallosumConnection(defaults={"rev": get_rev()})
    _callosum.start()
    _stop_status.clear()
    status_thread = threading.Thread(target=_emit_periodic_status, daemon=True)
    status_thread.start()

    try:
        # Handle activity-triggered execution mode
        if args.activity:
            success = run_activity_prompts(
                day=day,
                activity_id=args.activity,
                facet=args.facet,
                refresh=args.refresh,
                verbose=args.verbose,
                max_concurrency=args.jobs,
            )
            sys.exit(0 if success else 1)

        # Handle flush mode
        if args.flush:
            if not check_callosum_available():
                logging.warning("Callosum socket not found - prompts may fail to spawn")
            success = run_flush_prompts(
                day=day,
                segment=args.segment,
                verbose=args.verbose,
                stream=args.stream,
            )
            sys.exit(0 if success else 1)

        # Handle single prompt execution mode
        if args.run:
            success = run_single_prompt(
                day=day,
                name=args.run,
                segment=args.segment,
                refresh=args.refresh,
                facet=args.facet,
                stream=args.stream,
            )
            sys.exit(0 if success else 1)

        # Handle batch segment re-processing mode
        if args.segments:
            if not check_callosum_available():
                logging.warning("Callosum socket not found - prompts may fail to spawn")

            segments = cluster_segments(day)
            if not segments:
                logging.info(f"No segments found for {day}")
                sys.exit(0)

            total = len(segments)
            logging.info(f"Processing {total} segments for {day}")
            emit("segments_started", day=day, count=total)
            _update_status(segments_total=total, segments_completed=0)

            batch_start = time.time()
            batch_success = 0
            batch_failed = 0

            for i, seg in enumerate(segments, 1):
                seg_key = seg["key"]
                seg_stream = seg.get("stream")
                logging.info(
                    f"Processing segment {i}/{total}: {seg_key} ({seg['start']}-{seg['end']})"
                )
                try:
                    success, failed, _fn = run_prompts_by_priority(
                        day=day,
                        segment=seg_key,
                        refresh=args.refresh,
                        verbose=args.verbose,
                        max_concurrency=args.jobs,
                        stream=seg_stream,
                    )
                    # Touch stream.updated marker after each segment
                    try:
                        health_dir = day_path(day) / "health"
                        health_dir.mkdir(parents=True, exist_ok=True)
                        (health_dir / "stream.updated").touch()
                    except Exception:
                        pass
                    batch_success += success
                    batch_failed += failed
                    _update_status(segments_completed=i, segments_total=total)
                except Exception:
                    logging.exception(f"Segment {seg_key} failed with exception")
                    batch_failed += 1
                    _update_status(segments_completed=i, segments_total=total)

            duration_ms = int((time.time() - batch_start) * 1000)
            logging.info(
                f"All segments completed in {duration_ms}ms: "
                f"{batch_success} succeeded, {batch_failed} failed across {total} segments"
            )
            emit(
                "segments_completed",
                day=day,
                count=total,
                success=batch_success,
                failed=batch_failed,
                duration_ms=duration_ms,
            )

            if args.refresh:
                day_log(day, f"dream --segments --refresh failed={batch_failed}")
            else:
                day_log(day, f"dream --segments failed={batch_failed}")

            if batch_failed > 0:
                sys.exit(1)
            sys.exit(0)

        # Check callosum availability
        if not check_callosum_available():
            logging.warning("Callosum socket not found - prompts may fail to spawn")

        start_time = time.time()

        # PRE-PHASE: Run sense repair (daily only)
        if not args.segment:
            logging.info("Running pre-phase: sense repair")
            cmd = ["sol", "sense", "--day", day]
            if args.verbose:
                cmd.append("-v")
            day_log(day, f"starting: {' '.join(cmd)}")
            if not run_command(cmd, day):
                logging.warning("Sense repair failed, continuing anyway")

        # MAIN PHASE: Run all prompts by priority
        success_count, fail_count, failed_names = run_prompts_by_priority(
            day=day,
            segment=args.segment,
            refresh=args.refresh,
            verbose=args.verbose,
            max_concurrency=args.jobs,
            stream=args.stream,
        )

        # Touch stream.updated marker after segment processing
        if args.segment:
            try:
                health_dir = day_path(day) / "health"
                health_dir.mkdir(parents=True, exist_ok=True)
                (health_dir / "stream.updated").touch()
            except Exception:
                pass

        # POST-PHASE: Final indexing and stats (daily only)
        if not args.segment:
            logging.info("Running post-phase: indexer rescan")
            rescan_cmd = ["sol", "indexer", "--rescan"]
            if args.verbose:
                rescan_cmd.append("--verbose")
            run_queued_command(rescan_cmd, day, timeout=3600)

            logging.info("Running post-phase: journal stats")
            stats_cmd = ["sol", "journal-stats"]
            if args.verbose:
                stats_cmd.append("--verbose")
            run_command(stats_cmd, day)

            # Touch daily.updated marker after daily schedule completion
            try:
                health_dir = day_path(day) / "health"
                health_dir.mkdir(parents=True, exist_ok=True)
                (health_dir / "daily.updated").touch()
            except Exception:
                pass

        # Build log message
        msg = "dream"
        if args.refresh:
            msg += " --refresh"
        if fail_count:
            msg += f" failed={fail_count}"
        day_log(day, msg)

        duration_ms = int((time.time() - start_time) * 1000)
        logging.info(
            f"Dream completed in {duration_ms}ms: {success_count} succeeded, {fail_count} failed"
        )

        if fail_count > 0:
            names = ", ".join(failed_names)
            logging.error(f"{fail_count} prompt(s) failed: {names}")
            sys.exit(1)

    finally:
        _clear_status()
        _stop_status.set()
        status_thread.join(timeout=2)
        _callosum.stop()


if __name__ == "__main__":
    main()
