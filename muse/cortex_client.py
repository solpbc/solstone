# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Cortex client for managing AI agent requests."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from think.callosum import callosum_send
from think.utils import get_journal

logger = logging.getLogger(__name__)

# Module-level state for monotonic timestamp generation
_last_ts = 0


def cortex_request(
    prompt: str,
    persona: str,
    provider: Optional[str] = None,
    handoff_from: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    save: Optional[str] = None,
) -> str:
    """Create a Cortex agent request via Callosum broadcast.

    Args:
        prompt: The task or question for the agent
        persona: Agent persona - system (e.g., "default") or app-qualified (e.g., "entities:entity_assist")
        provider: AI provider - openai, google, anthropic, or claude
        handoff_from: Previous agent ID if this is a handoff request
        config: Provider-specific configuration (model, max_tokens, facet for Claude)
        save: Optional filename to save result to in current day directory

    Returns:
        Agent ID (timestamp-based string)
    """
    # Get journal path (for agent_id uniqueness check)
    journal_path = get_journal()

    # Create agents directory if it doesn't exist
    agents_dir = Path(journal_path) / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    # Generate monotonic timestamp in milliseconds, ensuring uniqueness
    global _last_ts
    ts = int(time.time() * 1000)

    # If same or earlier than last used, increment to ensure uniqueness
    if ts <= _last_ts:
        ts = _last_ts + 1

    _last_ts = ts
    agent_id = str(ts)

    # Build request object
    request = {
        "event": "request",
        "ts": ts,
        "agent_id": agent_id,
        "prompt": prompt,
        "provider": provider,
        "persona": persona,
    }

    # Add optional fields
    if config:
        if not isinstance(config, dict):
            raise ValueError("config must be a dictionary")
        # Merge config overrides directly into the request for a flat schema
        request.update(config)

    if handoff_from:
        request["handoff_from"] = handoff_from

    if save:
        request["save"] = save

    # Broadcast request to Callosum
    # Note: callosum_send() signature is send(tract, event, **fields)
    # Remove "event" from request dict to avoid conflict
    request_fields = {k: v for k, v in request.items() if k != "event"}
    callosum_send("cortex", "request", **request_fields)

    return agent_id


def create_synthetic_agent(result: str) -> str:
    """Create a synthetic agent with only a finish event.

    This is used for system-generated messages that appear as completed agents
    but don't have thinking/tool use steps. The agent file contains a single
    finish event with the result.

    Apps using this should create their own metadata/records as needed.

    Args:
        result: The message content (will be the finish event result)

    Returns:
        Agent ID (timestamp-based string)
    """
    journal_path = get_journal()

    # Create agents directory if it doesn't exist
    agents_dir = Path(journal_path) / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique monotonic timestamp
    global _last_ts
    ts = int(time.time() * 1000)
    if ts <= _last_ts:
        ts = _last_ts + 1
    _last_ts = ts
    agent_id = str(ts)

    # Create agent file with single finish event
    agent_file = agents_dir / f"{agent_id}.jsonl"
    finish_event = {"event": "finish", "result": result, "ts": ts, "agent_id": agent_id}

    with open(agent_file, "w", encoding="utf-8") as f:
        json.dump(finish_event, f)
        f.write("\n")

    # Broadcast finish event to Callosum so listeners are notified
    callosum_send("cortex", "finish", agent_id=agent_id, result=result, ts=ts)

    return agent_id


def get_agent_status(agent_id: str) -> str:
    """Get the status of a specific agent.

    Args:
        agent_id: The agent ID (timestamp)

    Returns:
        "completed" - Agent finished (*.jsonl exists)
        "running" - Agent still active (*_active.jsonl exists)
        "not_found" - No agent file exists
    """
    agents_dir = Path(get_journal()) / "agents"

    if (agents_dir / f"{agent_id}.jsonl").exists():
        return "completed"
    if (agents_dir / f"{agent_id}_active.jsonl").exists():
        return "running"
    return "not_found"


def get_agent_end_state(agent_id: str) -> str:
    """Get how a completed agent ended (finish or error).

    Args:
        agent_id: The agent ID (timestamp)

    Returns:
        "finish" - Agent completed successfully
        "error" - Agent ended with an error
        "running" - Agent is still active
        "unknown" - Agent file exists but no terminal event found
    """
    status = get_agent_status(agent_id)
    if status == "running":
        return "running"
    if status == "not_found":
        return "unknown"

    # Read events to find terminal state
    try:
        events = read_agent_events(agent_id)
        # Find last finish or error event
        for event in reversed(events):
            event_type = event.get("event")
            if event_type == "finish":
                return "finish"
            if event_type == "error":
                return "error"
        return "unknown"
    except FileNotFoundError:
        return "unknown"


def read_agent_events(agent_id: str) -> list[Dict[str, Any]]:
    """Read all events from an agent's JSONL log file.

    Args:
        agent_id: The agent ID (timestamp)

    Returns:
        List of event dictionaries in chronological order

    Raises:
        FileNotFoundError: If agent log doesn't exist
    """
    agents_dir = Path(get_journal()) / "agents"

    # Check for completed agent first, then active if not found
    agent_file = agents_dir / f"{agent_id}.jsonl"
    if not agent_file.exists():
        agent_file = agents_dir / f"{agent_id}_active.jsonl"
        if not agent_file.exists():
            raise FileNotFoundError(f"Agent log not found: {agent_id}")

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


def get_agent_thread(agent_id: str) -> list[str]:
    """Get all agent IDs in a conversation thread, in chronological order.

    Given any agent ID in a thread, walks backward to find the root,
    then forward to collect all agents in the thread.

    Args:
        agent_id: Any agent ID in the thread

    Returns:
        List of agent IDs in chronological order (oldest first)

    Raises:
        FileNotFoundError: If the specified agent doesn't exist
    """
    # Walk backward to find the root
    current = agent_id
    seen = {current}

    while True:
        try:
            events = read_agent_events(current)
        except FileNotFoundError:
            if current == agent_id:
                raise  # Original agent not found
            break  # Referenced agent doesn't exist, stop here

        # Find continue_from in request event
        request = next((e for e in events if e.get("event") == "request"), None)
        if not request:
            break

        continue_from = request.get("continue_from")
        if not continue_from or continue_from in seen:
            break

        seen.add(continue_from)
        current = continue_from

    root = current

    # Walk forward from root to build the thread
    thread = [root]
    seen = {root}

    while True:
        try:
            events = read_agent_events(thread[-1])
        except FileNotFoundError:
            break

        # Find continue event(s) - take the first one for linear threads
        continue_event = next(
            (e for e in events if e.get("event") == "continue" and e.get("to")), None
        )
        if not continue_event:
            break

        next_agent = continue_event["to"]
        if next_agent in seen:
            break  # Prevent cycles

        seen.add(next_agent)
        thread.append(next_agent)

    return thread


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

    agents_dir = Path(get_journal()) / "agents"
    if not agents_dir.exists():
        return {
            "agents": [],
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

    for agent_file in agents_dir.glob("*.jsonl"):
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
        agent_id = agent_file.stem.replace("_active", "")

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
                    "id": agent_id,
                    "persona": request.get("persona", "default"),
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
        "agents": paginated,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": total,
            "has_more": (offset + limit) < total,
        },
        "live_count": live_count,
        "historical_count": historical_count,
    }
