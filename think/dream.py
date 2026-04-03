# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unified prompt execution pipeline for solstone.

Runs all scheduled prompts (generators and agents) in priority order.
Lower priority numbers run first. All prompts at the same priority
run in parallel, then dream waits for completion before the next group.
"""

import argparse
import fnmatch
import json
import logging
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from think.activities import get_activity_output_path, load_activity_records
from think.callosum import CallosumConnection
from think.cluster import cluster_segments
from think.cortex_client import cortex_request, wait_for_agents
from think.facets import (
    get_active_facets,
    get_enabled_facets,
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
    iter_segments,
    segment_parse,
    setup_cli,
    updated_days,
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
    timeout: int | None = 610,
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

    completed, timed_out = wait_for_agents(agent_ids, timeout=timeout)

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


def _segment_dir(day: str, segment: str, stream: str | None) -> Path:
    """Return the expected segment directory without creating it."""
    return day_path(day) / (stream or "default") / segment


def _resolve_segment_dir(
    day: str,
    segment: str,
    stream: str | None,
) -> Path | None:
    """Resolve a segment directory, searching across streams when needed."""
    if stream:
        path = _segment_dir(day, segment, stream)
        return path if path.is_dir() else None

    for seg_stream, seg_key, seg_path in iter_segments(day):
        if seg_key == segment:
            return seg_path
    return None


def _load_json_file(path: Path, default: object) -> object:
    """Load JSON from a file, returning the provided default on failure."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _write_json_atomic(path: Path, data: object) -> None:
    """Atomically write JSON data to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def _count_nonempty_lines(path: Path) -> list[str]:
    """Return non-empty lines from a text file."""
    try:
        return [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    except OSError:
        return []


def _load_segment_facet_rows(
    day: str,
    segment: str,
    stream: str | None,
) -> list[dict]:
    """Load raw facets.json rows for a segment."""
    seg_dir = _resolve_segment_dir(day, segment, stream)
    if not seg_dir:
        return []

    facets_path = seg_dir / "agents" / "facets.json"
    data = _load_json_file(facets_path, [])
    return data if isinstance(data, list) else []


def _has_audio_embeddings(seg_dir: Path) -> bool:
    """Return True when a segment has audio embedding files."""
    for npz_path in seg_dir.glob("*.npz"):
        if npz_path.stem == "audio" or npz_path.stem.endswith("_audio"):
            return True
    return False


def _should_skip_preflight(
    prompt_name: str,
    *,
    day: str,
    segment: str | None,
    stream: str | None,
) -> tuple[bool, str | None]:
    """Return whether a prompt can be skipped before sending a cortex request."""
    if not segment:
        return (False, None)

    if prompt_name == "firstday_checkin":
        from think.awareness import get_onboarding

        onboarding = get_onboarding()
        if onboarding.get("status") != "complete":
            return (True, "preflight:not_complete")
        if onboarding.get("firstday_checkin_sent"):
            return (True, "preflight:already_sent")
        return (False, None)

    if prompt_name == "observation":
        from think.awareness import get_onboarding

        onboarding = get_onboarding()
        if onboarding.get("status") != "observing":
            return (True, "preflight:not_observing")
        return (False, None)

    seg_dir = _resolve_segment_dir(day, segment, stream)
    if seg_dir is None:
        return (False, None)

    if prompt_name == "speaker_attribution":
        if not _has_audio_embeddings(seg_dir):
            return (True, "preflight:no_embeddings")
        return (False, None)

    if prompt_name == "speakers":
        transcript_files = sorted(seg_dir.glob("audio.jsonl"))
        transcript_files.extend(sorted(seg_dir.glob("*_audio.jsonl")))
        transcript_files.extend(sorted(seg_dir.glob("*_transcript.jsonl")))
        if not transcript_files:
            return (True, "preflight:no_transcripts")

        speakers: set[str] = set()
        for transcript_path in transcript_files:
            lines = _count_nonempty_lines(transcript_path)
            for line in lines[1:]:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                speaker = entry.get("speaker")
                if speaker is not None:
                    speakers.add(str(speaker))
        if len(speakers) < 2:
            return (True, "preflight:single_speaker")
        return (False, None)

    if prompt_name == "activities":
        from muse.activity_state import find_previous_segment

        previous_segment = find_previous_segment(day, segment, stream=stream)
        if not previous_segment:
            return (True, "preflight:no_previous_segment")

        prev_dir = _resolve_segment_dir(day, previous_segment, stream)
        if prev_dir is None:
            return (True, "preflight:no_previous_activity_state")

        agents_dir = prev_dir / "agents"
        for facet_dir in sorted(agents_dir.iterdir()) if agents_dir.is_dir() else []:
            if facet_dir.is_dir() and (facet_dir / "activity_state.json").exists():
                return (False, None)
        return (True, "preflight:no_previous_activity_state")

    return (False, None)


def _classify_segment_density(
    day: str,
    segment: str,
    stream: str | None,
) -> str:
    """Classify segment content density as 'idle', 'low_change', or 'active'."""
    seg_dir = _segment_dir(day, segment, stream)
    if not seg_dir.exists():
        return "active"

    transcript_lines = 0
    transcript_files = sorted(seg_dir.glob("audio.jsonl"))
    transcript_files.extend(sorted(seg_dir.glob("*_audio.jsonl")))
    transcript_files.extend(sorted(seg_dir.glob("*_transcript.jsonl")))
    for transcript_path in transcript_files:
        lines = _count_nonempty_lines(transcript_path)
        transcript_lines += max(0, len(lines) - 1)

    imported_md = seg_dir / "imported.md"
    if imported_md.exists():
        transcript_lines += len(_count_nonempty_lines(imported_md))

    screen_frames = 0
    screen_files = sorted(seg_dir.glob("screen.jsonl"))
    screen_files.extend(sorted(seg_dir.glob("*_screen.jsonl")))
    for screen_path in screen_files:
        lines = _count_nonempty_lines(screen_path)
        if not lines:
            continue
        subtract_header = False
        try:
            first_entry = json.loads(lines[0])
            subtract_header = isinstance(first_entry, dict) and "raw" in first_entry
        except json.JSONDecodeError:
            subtract_header = False
        screen_frames += max(0, len(lines) - 1 if subtract_header else len(lines))

    if transcript_lines < 3 and screen_frames < 2:
        return "idle"
    if transcript_lines < 10 and screen_frames < 5:
        return "low_change"
    return "active"


def _activity_state_cache_path(day: str) -> Path:
    return day_path(day) / "health" / "activity_state_cache.json"


def _load_activity_state_cache(day: str) -> dict:
    data = _load_json_file(_activity_state_cache_path(day), {})
    return data if isinstance(data, dict) else {}


def _save_activity_state_cache(day: str, data: dict) -> None:
    _write_json_atomic(_activity_state_cache_path(day), data)


def _emit_activity_live_entries(
    *,
    day: str,
    segment: str,
    facet: str,
    entries: list[dict],
) -> None:
    """Emit activity.live events for active entries."""
    from think.callosum import callosum_send

    for entry in entries:
        if entry.get("state") != "active":
            continue
        try:
            callosum_send(
                "activity",
                "live",
                facet=facet,
                day=day,
                segment=segment,
                id=entry["id"],
                activity=entry["activity"],
                since=entry["since"],
                description=entry.get("description", ""),
                level=entry.get("level", "medium"),
                active_entities=entry.get("active_entities", []),
            )
        except Exception as exc:
            logging.warning(
                "Failed to emit carried activity.live for %s/%s: %s",
                facet,
                entry.get("id", "unknown"),
                exc,
            )


def _try_carry_forward_activity_state(
    *,
    day: str,
    segment: str,
    stream: str | None,
    facet: str,
    cache: dict,
) -> bool:
    """Try to carry forward cached activity_state for a facet."""
    cache_key = f"{stream or 'default'}:{facet}"
    entry = cache.get(cache_key)
    if not isinstance(entry, dict):
        return False

    if int(entry.get("carry_count", 0)) >= 6:
        return False

    cached_segment = entry.get("segment", "")
    if not cached_segment:
        return False

    current_start, _current_end = segment_parse(segment)
    _cached_start, cached_end = segment_parse(cached_segment)
    if current_start is None or cached_end is None:
        return False

    current_dt = datetime.combine(date.today(), current_start)
    cached_end_dt = datetime.combine(date.today(), cached_end)
    gap = current_dt - cached_end_dt
    if gap < timedelta(0) or gap > timedelta(minutes=10):
        return False

    facet_rows = _load_segment_facet_rows(day, segment, stream)
    facet_row = next((row for row in facet_rows if row.get("facet") == facet), None)
    if not facet_row:
        return False
    if facet_row.get("activity", "") != entry.get("facet_classification", ""):
        return False

    state = entry.get("state")
    if not isinstance(state, list):
        return False

    seg_dir = _segment_dir(day, segment, stream)
    output_path = seg_dir / "agents" / facet / "activity_state.json"
    _write_json_atomic(output_path, state)
    _emit_activity_live_entries(day=day, segment=segment, facet=facet, entries=state)

    entry["carry_count"] = int(entry.get("carry_count", 0)) + 1
    entry["segment"] = segment
    entry["updated_at"] = int(time.time())
    cache[cache_key] = entry
    return True


def _refresh_activity_state_cache(
    *,
    day: str,
    segment: str,
    stream: str | None,
    facets: list[str],
    cache: dict,
) -> None:
    """Refresh activity_state cache entries from newly-written output files."""
    if not facets:
        return

    facet_rows = _load_segment_facet_rows(day, segment, stream)
    facet_activity = {
        row.get("facet"): row.get("activity", "")
        for row in facet_rows
        if isinstance(row, dict) and row.get("facet")
    }
    seg_dir = _segment_dir(day, segment, stream)

    for facet in facets:
        state_path = seg_dir / "agents" / facet / "activity_state.json"
        data = _load_json_file(state_path, None)
        if not isinstance(data, list):
            continue

        cache[f"{stream or 'default'}:{facet}"] = {
            "state": data,
            "facet_classification": facet_activity.get(facet, ""),
            "segment": segment,
            "carry_count": 0,
            "updated_at": int(time.time()),
        }


def _pulse_counter_path(day: str) -> Path:
    return day_path(day) / "health" / "pulse_counter.json"


def _load_pulse_counter(day: str) -> dict:
    data = _load_json_file(_pulse_counter_path(day), {"count": 0, "last_segment": ""})
    return data if isinstance(data, dict) else {"count": 0, "last_segment": ""}


def _save_pulse_counter(day: str, data: dict) -> None:
    _write_json_atomic(_pulse_counter_path(day), data)


def _active_activity_keys_for_segment(
    day: str,
    segment: str,
    stream: str | None,
) -> set[tuple[str, str, str]]:
    """Return active (facet, activity, since) tuples for a segment."""
    seg_dir = _resolve_segment_dir(day, segment, stream)
    if not seg_dir:
        return set()

    keys: set[tuple[str, str, str]] = set()
    agents_dir = seg_dir / "agents"
    if not agents_dir.is_dir():
        return keys

    for facet_dir in sorted(agents_dir.iterdir()):
        if not facet_dir.is_dir():
            continue
        state_path = facet_dir / "activity_state.json"
        data = _load_json_file(state_path, None)
        if not isinstance(data, list):
            continue
        for item in data:
            if item.get("state") == "active":
                keys.add(
                    (
                        facet_dir.name,
                        str(item.get("activity", "")),
                        str(item.get("since", "")),
                    )
                )
    return keys


def _detect_activity_state_change(
    day: str,
    segment: str,
    stream: str | None,
) -> bool:
    """Check whether activity_state changed for this segment."""
    from muse.activity_state import find_previous_segment

    previous_segment = find_previous_segment(day, segment, stream=stream)
    if not previous_segment:
        return True

    current_keys = _active_activity_keys_for_segment(day, segment, stream)
    previous_keys = _active_activity_keys_for_segment(day, previous_segment, stream)
    return current_keys != previous_keys


def run_prompts_by_priority(
    day: str,
    segment: str | None,
    refresh: bool,
    verbose: bool,
    max_concurrency: int = 2,
    stream: str | None = None,
    timeout: int | None = 610,
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

    segment_density = "active"
    if segment:
        segment_density = _classify_segment_density(day, segment, stream)
        if segment_density != "active":
            logging.info("Segment %s classified as %s", segment, segment_density)

    activity_changed = False
    as_cache = _load_activity_state_cache(day) if segment else {}

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

        if priority == 10 and segment_density != "active":
            logging.info(
                "Skipping priority-10 group (%d agents): segment %s is %s",
                len(prompts_list),
                segment,
                segment_density,
            )
            continue

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
        activity_state_llm_facets: set[str] = set()

        spawned: list[
            tuple[str, str, dict, str | None]
        ] = []  # (agent_id, name, config, facet)
        group_success = 0
        group_failed = 0

        for prompt_name, config in prompts_list:
            is_generate = config["type"] == "generate"

            # Check exclude_streams filter
            exclude_patterns = config.get("exclude_streams")
            if exclude_patterns and stream:
                if any(fnmatch.fnmatch(stream, pat) for pat in exclude_patterns):
                    logging.info(
                        f"Skipping {prompt_name}: stream '{stream}' matches exclude_streams"
                    )
                    continue

            skip, skip_reason = _should_skip_preflight(
                prompt_name,
                day=day,
                segment=segment,
                stream=stream,
            )
            if skip:
                logging.info("Skipping %s: %s", prompt_name, skip_reason)
                continue

            # Skip pulse when sol/pulse.md is already current for this segment
            if prompt_name == "pulse" and segment and not refresh:
                try:
                    pulse_path = Path(get_journal()) / "sol" / "pulse.md"
                    if pulse_path.exists():
                        seg_start, _ = segment_parse(segment)
                        if seg_start:
                            day_date = datetime.strptime(day, "%Y%m%d").date()
                            seg_dt = datetime.combine(day_date, seg_start)
                            if pulse_path.stat().st_mtime >= seg_dt.timestamp():
                                logging.info(
                                    f"Skipping pulse: sol/pulse.md current for {segment}"
                                )
                                continue
                except Exception:
                    pass

                if not activity_changed:
                    pulse_counter = _load_pulse_counter(day)
                    if pulse_counter.get("count", 0) < 5:
                        pulse_counter["count"] = int(pulse_counter.get("count", 0)) + 1
                        pulse_counter["last_segment"] = segment
                        _save_pulse_counter(day, pulse_counter)
                        logging.info(
                            "Skipping pulse: counter %d/6 for %s",
                            pulse_counter["count"],
                            segment,
                        )
                        continue
                    _save_pulse_counter(day, {"count": 0, "last_segment": segment})
                else:
                    _save_pulse_counter(day, {"count": 0, "last_segment": segment})

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
                        if prompt_name == "activity_state" and segment:
                            if not refresh and _try_carry_forward_activity_state(
                                day=day,
                                segment=segment,
                                stream=stream,
                                facet=facet_name,
                                cache=as_cache,
                            ):
                                logging.info(
                                    "Carry-forward activity_state for %s", facet_name
                                )
                                continue
                            activity_state_llm_facets.add(facet_name)

                        # Always pass day for instructions.day context
                        request_config: dict = {"facet": facet_name, "day": day}
                        if is_generate:
                            request_config["output"] = config.get("output", "md")
                            if refresh:
                                request_config["refresh"] = True
                        elif config.get("output"):
                            # Cogitate agents with explicit output get auto-persisted
                            request_config["output"] = config["output"]
                        env: dict[str, str] = {
                            "SOL_DAY": day,
                            "SOL_FACET": facet_name,
                        }
                        if segment:
                            request_config["segment"] = segment
                            env["SOL_SEGMENT"] = segment
                            if stream:
                                request_config["stream"] = stream
                                env["SOL_STREAM"] = stream
                        request_config["env"] = env
                        request_config["schedule"] = target_schedule

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
                            request_config["stream"] = stream
                            env["SOL_STREAM"] = stream
                    request_config["env"] = env
                    request_config["schedule"] = target_schedule

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
                            spawned, target_schedule, day, segment, stream, timeout
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
        s, f, fn = _drain_priority_batch(
            spawned, target_schedule, day, segment, stream, timeout
        )
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

        if priority == 95 and segment:
            _refresh_activity_state_cache(
                day=day,
                segment=segment,
                stream=stream,
                facets=sorted(activity_state_llm_facets),
                cache=as_cache,
            )
            _save_activity_state_cache(day, as_cache)
            activity_changed = _detect_activity_state_change(day, segment, stream)

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

            completed, timed_out = wait_for_agents(agent_ids, timeout=610)

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
                request_config["schedule"] = "activity"
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
            if stream:
                request_config["stream"] = stream
            request_config["schedule"] = "segment"
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
        completed, timed_out = wait_for_agents(agent_ids, timeout=610)

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


def dry_run(
    day: str,
    *,
    segment: str | None = None,
    segments: bool = False,
    facet: str | None = None,
    activity: str | None = None,
    flush: bool = False,
    refresh: bool = False,
    stream: str | None = None,
) -> None:
    """Print what dream would execute without spawning any agents."""
    day_formatted = iso_date(day)

    if activity:
        _dry_run_activity(day, day_formatted, activity, facet or "", refresh)
        return

    if flush:
        _dry_run_flush(day, segment or "")
        return

    if segments:
        segs = cluster_segments(day)
        if not segs:
            print(f"No segments found for {day}")
            return
        print(f"Day {day_formatted} — re-process {len(segs)} segments\n")
        for i, seg in enumerate(segs, 1):
            seg_key = seg["key"]
            seg_stream = seg.get("stream")
            label = f"  [{i}/{len(segs)}] {seg_key} ({seg['start']}-{seg['end']})"
            if seg_stream:
                label += f" stream={seg_stream}"
            print(label)
        print()
        # Show what prompts would run per segment
        all_prompts = get_muse_configs(schedule="segment")
        if all_prompts:
            _print_prompt_table(all_prompts, day, segment="<each>", stream=stream)
        return

    # Default: full daily or segment run
    target_schedule = "segment" if segment else "daily"
    all_prompts = get_muse_configs(schedule=target_schedule)

    header = f"Day {day_formatted}"
    if segment:
        header += f" segment {segment}"
    if refresh:
        header += " (refresh)"
    print(header + "\n")

    if not segment:
        print("Pre-phase:  sol sense --day " + day)

    if not all_prompts:
        print(f"No prompts for schedule: {target_schedule}")
    else:
        _print_prompt_table(
            all_prompts, day, segment=segment, refresh=refresh, stream=stream
        )

    if not segment:
        print("Post-phase: sol indexer --rescan")
        print("Post-phase: sol journal-stats")


def _print_prompt_table(
    prompts: dict[str, dict],
    day: str,
    *,
    segment: str | None = None,
    refresh: bool = False,
    stream: str | None = None,
) -> None:
    """Print a grouped-by-priority table of prompts."""
    enabled_facets = get_enabled_facets()

    if segment and segment != "<each>":
        active_facets = set(
            f
            for f in load_segment_facets(day, segment, stream=stream)
            if f in enabled_facets
        )
    else:
        active_facets = get_active_facets(day)

    # Group by priority
    groups: dict[int, list[tuple[str, dict]]] = {}
    for name, config in prompts.items():
        pri = config["priority"]
        groups.setdefault(pri, []).append((name, config))

    total = 0
    for priority in sorted(groups.keys()):
        items = groups[priority]
        print(f"Priority {priority}:")
        for name, config in items:
            is_gen = config["type"] == "generate"
            type_label = "gen" if is_gen else "agent"
            output_fmt = config.get("output", "md") if is_gen else None

            if config.get("multi_facet"):
                always = config.get("always", False)
                target_facets = [
                    f for f in enabled_facets if always or f in active_facets
                ]
                skipped = [f for f in enabled_facets if f not in target_facets]
                for f in target_facets:
                    status = (
                        _output_status(
                            day, name, segment, output_fmt, facet=f, stream=stream
                        )
                        if is_gen
                        else ""
                    )
                    print(f"  {type_label}  {name}/{f}{status}")
                    total += 1
                if skipped:
                    print(f"  skip {name} — no activity: {', '.join(skipped)}")
            else:
                status = (
                    _output_status(day, name, segment, output_fmt, stream=stream)
                    if is_gen
                    else ""
                )
                print(f"  {type_label}  {name}{status}")
                total += 1
        print()

    print(f"Total: {total} agents")


def _output_status(
    day: str,
    name: str,
    segment: str | None,
    output_format: str | None,
    *,
    facet: str | None = None,
    stream: str | None = None,
) -> str:
    """Return a short status suffix for a generator output file."""
    if segment == "<each>":
        return ""
    path = get_output_path(
        day_path(day),
        name,
        segment=segment,
        output_format=output_format,
        facet=facet,
        stream=stream,
    )
    if path.exists():
        return " (exists)"
    return " (new)"


def _dry_run_activity(
    day: str, day_formatted: str, activity_id: str, facet: str, refresh: bool
) -> None:
    """Dry-run for --activity mode."""
    records = load_activity_records(facet, day)
    record = next((r for r in records if r.get("id") == activity_id), None)

    if not record:
        print(f"Activity not found: {activity_id} in facet '{facet}' on {day}")
        return

    activity_type = record.get("activity", "")
    segments = record.get("segments", [])

    print(
        f"Day {day_formatted} --activity {activity_id} --facet {facet}"
        + (" (refresh)" if refresh else "")
        + "\n"
    )
    print(f"  type:     {activity_type}")
    print(f"  segments: {len(segments)}")

    all_prompts = get_muse_configs(schedule="activity")
    matching = {
        n: c
        for n, c in all_prompts.items()
        if "*" in c.get("activities", []) or activity_type in c.get("activities", [])
    }

    if not matching:
        print(f"\n  No agents match activity type '{activity_type}'")
        return

    groups: dict[int, list[tuple[str, dict]]] = {}
    for n, c in matching.items():
        groups.setdefault(c["priority"], []).append((n, c))

    print()
    total = 0
    for priority in sorted(groups.keys()):
        items = groups[priority]
        print(f"Priority {priority}:")
        for n, c in items:
            is_gen = c["type"] == "generate"
            type_label = "gen" if is_gen else "agent"
            output_fmt = c.get("output", "md") if is_gen else None
            status = ""
            if is_gen:
                path = get_activity_output_path(
                    facet, day, activity_id, n, output_format=output_fmt
                )
                status = " (exists)" if path.exists() else " (new)"
            print(f"  {type_label}  {n}{status}")
            total += 1
        print()

    print(f"Total: {total} agents")


def _dry_run_flush(day: str, segment: str) -> None:
    """Dry-run for --flush mode."""
    all_prompts = get_muse_configs(schedule="segment")
    flush_prompts = {
        n: c
        for n, c in all_prompts.items()
        if isinstance(c.get("hook"), dict) and c["hook"].get("flush")
    }

    day_formatted = iso_date(day)
    print(f"Day {day_formatted} --flush segment {segment}\n")

    if not flush_prompts:
        print("  No flush-eligible agents")
        return

    for n, c in flush_prompts.items():
        type_label = "gen" if c["type"] == "generate" else "agent"
        print(f"  {type_label}  {n}")

    print(f"\nTotal: {len(flush_prompts)} agents")


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
        help="Segment key in HHMMSS_LEN format (processes segment agents only)",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Refresh existing outputs"
    )
    parser.add_argument(
        "--segments",
        action="store_true",
        help="Re-process all segments for the day (incompatible with --segment, --facet)",
    )
    parser.add_argument(
        "--facet",
        metavar="NAME",
        help="Target a specific facet (only used with --activity)",
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
    parser.add_argument(
        "--no-timeout",
        action="store_true",
        help="Disable per-batch agent wait timeout in --segments mode",
    )
    parser.add_argument(
        "--updated",
        action="store_true",
        help="List days with pending daily processing and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing anything",
    )
    return parser


def main() -> None:
    global _callosum

    parser = parse_args()
    args = setup_cli(parser)

    from think.awareness import ensure_sol_directory

    ensure_sol_directory()

    if args.updated:
        incompatible = []
        if args.day:
            incompatible.append("--day")
        if args.segment:
            incompatible.append("--segment")
        if args.facet:
            incompatible.append("--facet")
        if args.activity:
            incompatible.append("--activity")
        if args.flush:
            incompatible.append("--flush")
        if args.segments:
            incompatible.append("--segments")
        if incompatible:
            parser.error(f"--updated is incompatible with {', '.join(incompatible)}")
        today = date.today().strftime("%Y%m%d")
        for d in updated_days(exclude={today}):
            print(d)
        sys.exit(0)

    day = args.day
    if day is None:
        day = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    day_dir = day_path(day)

    if not day_dir.is_dir():
        parser.error(f"Day folder not found: {day_dir}")

    if args.facet and not args.activity:
        parser.error("--facet requires --activity")

    if args.activity and not args.facet:
        parser.error("--activity requires --facet")

    # Auto-enable refresh for updated days (full daily runs only)
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

    if args.activity and (args.segment or args.segments or args.flush):
        parser.error(
            "--activity is incompatible with --segment, --segments, and --flush"
        )

    if args.flush and not args.segment:
        parser.error("--flush requires --segment")

    if args.flush and (args.segments or args.refresh):
        parser.error("--flush is incompatible with --segments and --refresh")

    if args.segments and (args.segment or args.facet):
        parser.error("--segments is incompatible with --segment and --facet")

    if args.dry_run:
        dry_run(
            day,
            segment=args.segment,
            segments=args.segments,
            facet=args.facet,
            activity=args.activity,
            flush=args.flush,
            refresh=args.refresh,
            stream=args.stream,
        )
        sys.exit(0)

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
                        timeout=None if args.no_timeout else 610,
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
        resolved_stream = args.stream
        if args.segment and args.stream is None:
            matches = [(s, k) for s, k, _ in iter_segments(day) if k == args.segment]
            if not matches:
                parser.error(
                    f"Segment {args.segment} not found in any stream under {day_dir}"
                )
            resolved_stream = matches[0][0]

        success_count, fail_count, failed_names = run_prompts_by_priority(
            day=day,
            segment=args.segment,
            refresh=args.refresh,
            verbose=args.verbose,
            max_concurrency=args.jobs,
            stream=resolved_stream,
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

            # Set first_daily_ready awareness flag after first post-onboarding daily
            try:
                from think.awareness import get_current, get_onboarding, update_state

                ob = get_onboarding()
                if ob.get("status") == "complete":
                    cur = get_current()
                    if not cur.get("journal", {}).get("first_daily_ready"):
                        update_state(
                            "journal",
                            {
                                "first_daily_ready": True,
                                "first_daily_ready_at": datetime.now().strftime(
                                    "%Y%m%dT%H:%M:%S"
                                ),
                            },
                        )
            except Exception:
                pass

            # Notify supervisor that daily dream processing is complete
            emit(
                "daily_complete",
                day=day,
                success=success_count,
                failed=fail_count,
                duration_ms=int((time.time() - start_time) * 1000),
            )

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
