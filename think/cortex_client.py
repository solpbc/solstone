# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Cortex client for managing AI agent requests."""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from think.callosum import CallosumConnection, callosum_send
from think.utils import get_journal, now_ms

logger = logging.getLogger(__name__)

# Module-level state for monotonic timestamp generation
_last_ts = 0


def _find_agent_file(talents_dir: Path, use_id: str) -> tuple[Path | None, str]:
    """Find an agent log file in per-agent subdirectories.

    Returns:
        Tuple of (file_path, status) where status is
        "completed", "running", or "not_found".
    """
    for match in talents_dir.glob(f"*/{use_id}.jsonl"):
        return match, "completed"
    for match in talents_dir.glob(f"*/{use_id}_active.jsonl"):
        return match, "running"
    return None, "not_found"


def cortex_request(
    prompt: str,
    name: str,
    provider: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str | None:
    """Create a Cortex agent request via Callosum broadcast.

    Args:
        prompt: The task or question for the agent
        name: Agent name - system (e.g., "unified") or app-qualified (e.g., "entities:entity_assist")
        provider: AI provider - openai, google, or anthropic
        config: Provider-specific configuration (model, max_output_tokens, thinking_budget, etc.)

    Returns:
        Agent ID (timestamp-based string), or None if the Callosum send failed.
    """
    # Get journal path (for use_id uniqueness check)
    journal_path = get_journal()

    # Create agents directory if it doesn't exist
    talents_dir = Path(journal_path) / "talents"
    talents_dir.mkdir(parents=True, exist_ok=True)

    # Generate monotonic timestamp in milliseconds, ensuring uniqueness
    global _last_ts
    ts = now_ms()

    # If same or earlier than last used, increment to ensure uniqueness
    if ts <= _last_ts:
        ts = _last_ts + 1

    _last_ts = ts
    use_id = str(ts)

    # Build request object
    request = {
        "event": "request",
        "ts": ts,
        "use_id": use_id,
        "prompt": prompt,
        "provider": provider,
        "name": name,
    }

    # Add optional fields
    if config:
        if not isinstance(config, dict):
            raise ValueError("config must be a dictionary")
        # Merge config overrides directly into the request for a flat schema
        request.update(config)

    # Broadcast request to Callosum
    # Note: callosum_send() signature is send(tract, event, **fields)
    # Remove "event" from request dict to avoid conflict
    request_fields = {k: v for k, v in request.items() if k != "event"}
    sent = callosum_send("cortex", "request", **request_fields)

    if not sent:
        logger.info("Failed to send cortex request for agent '%s'", name)
        return None

    return use_id


def get_use_log_status(use_id: str) -> str:
    """Get the status of a specific agent from its log file.

    Args:
        use_id: The agent ID (timestamp)

    Returns:
        "completed" - Agent finished (*.jsonl exists)
        "running" - Agent still active (*_active.jsonl exists)
        "not_found" - No agent file exists
    """
    talents_dir = Path(get_journal()) / "talents"
    _, status = _find_agent_file(talents_dir, use_id)
    return status


def wait_for_agents(
    use_ids: list[str],
    timeout: int | None = 600,
) -> tuple[dict[str, str], list[str]]:
    """Wait for agents to complete via Callosum events.

    Listens for cortex.finish and cortex.error events. Sets up the event
    listener first, then does an initial file check for agents that may have
    already completed, and a final file check at timeout as a backstop for
    any missed events.

    Args:
        use_ids: List of agent IDs to wait for
        timeout: Maximum wait time in seconds (default 600 = 10 minutes)

    Returns:
        Tuple of (completed, timed_out) where completed is a dict mapping
        use_id to end state ("finish" or "error"), and timed_out is a
        list of agent IDs that did not complete within the timeout.
    """
    pending = set(use_ids)
    completed: dict[str, str] = {}
    lock = threading.Lock()
    all_done = threading.Event()

    def on_message(msg: dict) -> None:
        if msg.get("tract") != "cortex":
            return
        use_id = msg.get("use_id")
        if not use_id:
            return

        event_type = msg.get("event")
        if event_type in ("finish", "error"):
            with lock:
                if use_id in pending:
                    completed[use_id] = event_type
                    pending.discard(use_id)
                    if not pending:
                        all_done.set()

    # Start listener BEFORE initial check to avoid race condition
    listener = CallosumConnection()
    listener.start(callback=on_message)

    try:
        # Initial file check (with lock since callback may be running)
        with lock:
            for use_id in list(pending):
                end_state = get_use_end_state(use_id)
                if end_state in ("finish", "error"):
                    completed[use_id] = end_state
                    pending.discard(use_id)

            if not pending:
                return completed, []

        # Wait for all completions or timeout
        all_done.wait(timeout=timeout)

    finally:
        listener.stop()

    # Final file check for any remaining (backstop for missed events)
    # Listener is stopped, so no lock needed
    for use_id in list(pending):
        end_state = get_use_end_state(use_id)
        if end_state in ("finish", "error"):
            logger.info(
                f"Agent {use_id} completion event not received but agent completed"
            )
            completed[use_id] = end_state
            pending.discard(use_id)

    return completed, list(pending)


def get_use_end_state(use_id: str) -> str:
    """Get how a completed agent ended (finish or error).

    Checks file contents for terminal events even if file is still _active.jsonl,
    since Callosum broadcasts happen before file rename.

    Args:
        use_id: The agent ID (timestamp)

    Returns:
        "finish" - Agent completed successfully
        "error" - Agent ended with an error
        "running" - Agent is still active (no terminal event in file)
        "unknown" - Agent file not found
    """
    status = get_use_log_status(use_id)
    if status == "not_found":
        return "unknown"

    # Read events to find terminal state (even for "running" files that may
    # have finish event - Callosum broadcast happens before file rename)
    try:
        events = read_agent_events(use_id)
        # Find last finish or error event
        for event in reversed(events):
            event_type = event.get("event")
            if event_type == "finish":
                return "finish"
            if event_type == "error":
                return "error"
        # No terminal event found - still running
        return "running"
    except FileNotFoundError:
        return "unknown"


def read_agent_events(use_id: str) -> list[Dict[str, Any]]:
    """Read all events from an agent's JSONL log file.

    Args:
        use_id: The agent ID (timestamp)

    Returns:
        List of event dictionaries in chronological order

    Raises:
        FileNotFoundError: If agent log doesn't exist
    """
    talents_dir = Path(get_journal()) / "talents"
    agent_file, _status = _find_agent_file(talents_dir, use_id)
    if agent_file is None:
        raise FileNotFoundError(f"Agent log not found: {use_id}")

    events = []
    with open(agent_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError:
                logger.debug(f"Skipping malformed JSON in {agent_file}")
                continue

    return events


def cortex_agents(
    limit: int = 10,
    offset: int = 0,
    agent_type: str = "all",
    facet: Optional[str] = None,
) -> Dict[str, Any]:
    """List agents from the journal with pagination and filtering.

    Args:
        limit: Maximum number of agents to return (1-100)
        offset: Number of agents to skip
        agent_type: Filter by "live", "historical", or "all"
        facet: Optional facet to filter by. If provided, only returns agents
               that were run in this facet context. None means no filtering.

    Returns:
        Dictionary with agents list and pagination info
    """
    # Validate parameters
    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    talents_dir = Path(get_journal()) / "talents"
    if not talents_dir.exists():
        return {
            "talents": [],
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": 0,
                "has_more": False,
            },
            "live_count": 0,
            "historical_count": 0,
        }

    # Collect all agent files
    all_agents = []
    live_count = 0
    historical_count = 0

    for agent_file in talents_dir.glob("*/*.jsonl"):
        # Determine status from filename
        is_active = "_active.jsonl" in agent_file.name
        is_pending = "_pending.jsonl" in agent_file.name

        # Skip pending files
        if is_pending:
            continue

        status = "running" if is_active else "completed"

        # Count by type
        if status == "running":
            live_count += 1
        else:
            historical_count += 1

        # Filter by requested type
        if agent_type == "live" and status != "running":
            continue
        if agent_type == "historical" and status != "completed":
            continue

        # Extract agent ID from filename
        use_id = agent_file.stem.replace("_active", "")

        # Read agent file to get request info and calculate runtime
        try:
            with open(agent_file, "r") as f:
                lines = f.readlines()
                if not lines:
                    continue

                # Parse first line (request)
                first_line = lines[0].strip()
                if not first_line:
                    continue

                request = json.loads(first_line)
                if request.get("event") != "request":
                    continue

                # Extract facet from request
                agent_facet = request.get("facet")

                # Filter by facet if specified
                if facet is not None and agent_facet != facet:
                    continue

                # Extract basic info
                agent_info = {
                    "id": use_id,
                    "name": request.get("name", "unified"),
                    "start": request.get("ts", 0),
                    "status": status,
                    "prompt": request.get("prompt", ""),
                    "provider": request.get("provider", "openai"),
                    "facet": agent_facet,
                }

                # For completed agents, find finish event to calculate runtime
                if status == "completed" and len(lines) > 1:
                    # Read last few lines to find finish event (reading backwards is more efficient)
                    for line in reversed(lines[-10:]):  # Check last 10 lines
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                            if event.get("event") == "finish":
                                end_ts = event.get("ts", 0)
                                if end_ts and agent_info["start"]:
                                    # Calculate runtime in seconds
                                    agent_info["runtime_seconds"] = (
                                        end_ts - agent_info["start"]
                                    ) / 1000.0
                                break
                        except json.JSONDecodeError:
                            continue

                all_agents.append(agent_info)
        except (json.JSONDecodeError, IOError):
            # Skip malformed files
            continue

    # Sort by start time (newest first)
    all_agents.sort(key=lambda x: x["start"], reverse=True)

    # Apply pagination
    total = len(all_agents)
    paginated = all_agents[offset : offset + limit]

    return {
        "talents": paginated,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": total,
            "has_more": (offset + limit) < total,
        },
        "live_count": live_count,
        "historical_count": historical_count,
    }
