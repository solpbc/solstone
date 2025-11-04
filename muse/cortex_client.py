"""Cortex client for managing AI agent requests."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from think.callosum import callosum_send

logger = logging.getLogger(__name__)


def cortex_request(
    prompt: str,
    persona: str,
    backend: Optional[str] = None,
    handoff_from: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    save: Optional[str] = None,
) -> str:
    """Create a Cortex agent request via Callosum broadcast.

    Args:
        prompt: The task or question for the agent
        persona: Agent persona from think/agents/*.txt
        backend: AI backend - openai, google, anthropic, or claude
        handoff_from: Previous agent ID if this is a handoff request
        config: Backend-specific configuration (model, max_tokens, facet for Claude)
        save: Optional filename to save result to in current day directory

    Returns:
        Agent ID (timestamp-based string)
    """
    # Get journal path from environment (for agent_id uniqueness check)
    journal_path = os.environ.get("JOURNAL_PATH")
    if not journal_path:
        raise ValueError("JOURNAL_PATH environment variable not set")

    # Create agents directory if it doesn't exist
    agents_dir = Path(journal_path) / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp in milliseconds, ensuring uniqueness
    ts = int(time.time() * 1000)
    while (agents_dir / f"{ts}_active.jsonl").exists() or (
        agents_dir / f"{ts}.jsonl"
    ).exists():
        ts += 1

    agent_id = str(ts)

    # Build request object
    request = {
        "event": "request",
        "ts": ts,
        "agent_id": agent_id,
        "prompt": prompt,
        "backend": backend,
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


def read_agent_events(agent_id: str) -> list[Dict[str, Any]]:
    """Read all events from an agent's JSONL log file.

    Args:
        agent_id: The agent ID (timestamp)

    Returns:
        List of event dictionaries in chronological order

    Raises:
        FileNotFoundError: If agent log doesn't exist
    """
    journal_path = os.environ.get("JOURNAL_PATH")
    if not journal_path:
        raise ValueError("JOURNAL_PATH environment variable not set")

    agents_dir = Path(journal_path) / "agents"

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


def cortex_agents(
    limit: int = 10,
    offset: int = 0,
    agent_type: str = "all",
) -> Dict[str, Any]:
    """List agents from the journal with pagination and filtering.

    Args:
        limit: Maximum number of agents to return (1-100)
        offset: Number of agents to skip
        agent_type: Filter by "live", "historical", or "all"

    Returns:
        Dictionary with agents list and pagination info
    """
    # Get journal path from environment
    journal_path = os.environ.get("JOURNAL_PATH")
    if not journal_path:
        raise ValueError("JOURNAL_PATH environment variable not set")

    # Validate parameters
    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    agents_dir = Path(journal_path) / "agents"
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

                # Extract basic info
                agent_info = {
                    "id": agent_id,
                    "persona": request.get("persona", "default"),
                    "start": request.get("ts", 0),
                    "status": status,
                    "prompt": request.get("prompt", ""),
                    "model": request.get(
                        "backend", "openai"
                    ),  # Backend is the model provider
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
