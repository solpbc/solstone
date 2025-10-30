"""Cortex client for managing AI agent requests."""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from watchfiles import Change, watch

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Track reading state for an agent."""

    position: int = 0
    partial_line: str = ""


def cortex_request(
    prompt: str,
    persona: str,
    backend: Optional[str] = None,
    handoff_from: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    save: Optional[str] = None,
) -> str:
    """Create a Cortex agent request file.

    Args:
        prompt: The task or question for the agent
        persona: Agent persona from think/agents/*.txt
        backend: AI backend - openai, google, anthropic, or claude
        handoff_from: Previous agent ID if this is a handoff request
        config: Backend-specific configuration (model, max_tokens, domain for Claude)
        save: Optional filename to save result to in current day directory

    Returns:
        Path to the created request file
    """
    # Get journal path from environment
    journal_path = os.environ.get("JOURNAL_PATH")
    if not journal_path:
        raise ValueError("JOURNAL_PATH environment variable not set")

    # Create agents directory if it doesn't exist
    agents_dir = Path(journal_path) / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp in milliseconds, ensuring uniqueness
    ts = int(time.time() * 1000)
    while (agents_dir / f"{ts}_active.jsonl").exists() or (
        agents_dir / f"{ts}_pending.jsonl"
    ).exists():
        ts += 1

    # Build request object
    request = {
        "event": "request",
        "ts": ts,
        "agent_id": str(ts),
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

    # Write to pending file
    pending_file = agents_dir / f"{ts}_pending.jsonl"
    with open(pending_file, "w") as f:
        json.dump(request, f)
        f.write("\n")

    # Rename to active to trigger Cortex processing (atomic operation)
    active_file = agents_dir / f"{ts}_active.jsonl"
    pending_file.rename(active_file)

    return str(active_file)


def _extract_agent_id(path: Path) -> Optional[str]:
    """Extract agent_id from filename. Returns None for _pending files."""
    name = path.name
    if name.endswith("_pending.jsonl"):
        return None
    if name.endswith("_active.jsonl"):
        return name[:-13]  # Remove "_active.jsonl"
    if name.endswith(".jsonl"):
        return name[:-6]  # Remove ".jsonl"
    return None


def _process_agent_file(
    path: Path,
    agent_id: str,
    state: AgentState,
    on_event: Callable[[Dict[str, Any]], Optional[bool]],
) -> bool:
    """Process agent file and return True if should stop tracking.

    Returns True if saw finish/error event or callback requested stop.
    """
    try:
        with open(path, "r") as f:
            f.seek(state.position)
            new_content = f.read()

            if not new_content:
                return False

            content = state.partial_line + new_content
            lines = content.split("\n")

            if not content.endswith("\n"):
                state.partial_line = lines[-1]
                lines = lines[:-1]
            else:
                state.partial_line = ""

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)

                    try:
                        result = on_event(event)
                        if result is False:
                            state.position = f.tell()
                            return True
                    except Exception:
                        state.position = f.tell()
                        return True
                except json.JSONDecodeError:
                    continue

            state.position = f.tell()
            return False

    except (OSError, IOError):
        # File doesn't exist (might be renamed), just skip
        return False


def cortex_watch(
    on_event: Callable[[Dict[str, Any]], Optional[bool]],
    stop_event: Optional[Any] = None,
) -> None:
    """Watch for Cortex agent events and emit callbacks.

    This function blocks and watches for any agent files in the agents
    directory, efficiently tailing them for new events. When new events are
    written, they are parsed and passed to the on_event callback.

    Args:
        on_event: Callback function that receives each event as a dict.
                 Should return False to stop watching, True/None to continue.
        stop_event: Optional threading.Event() to stop watching cleanly.
                   When set(), the watcher will exit its loop gracefully.

    The callback receives event dictionaries with at least:
        - event: Event type (request, start, tool_start, tool_end, finish, error, etc.)
        - ts: Millisecond timestamp
        - Additional fields depend on event type

    Clean shutdown example:
        import threading
        stop_ev = threading.Event()
        t = threading.Thread(target=lambda: cortex_watch(on_event, stop_ev))
        t.start()
        # Later, to stop:
        stop_ev.set()
        t.join()
    """
    # Get journal path from environment
    journal_path = os.environ.get("JOURNAL_PATH")
    if not journal_path:
        raise ValueError("JOURNAL_PATH environment variable not set")

    agents_dir = Path(journal_path) / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    # Track state by agent_id (stable across renames)
    agent_states: Dict[str, AgentState] = {}

    # Initial scan for existing active files
    for active_file in agents_dir.glob("*_active.jsonl"):
        if active_file.is_file():
            agent_id = _extract_agent_id(active_file)
            if agent_id:
                # Start from current end of file for existing files
                stat = active_file.stat()
                agent_states[agent_id] = AgentState(position=stat.st_size)

    # Watch for changes in the agents directory
    watch_kwargs = {"raise_interrupt": False}
    if stop_event is not None:
        watch_kwargs["stop_event"] = stop_event

    for changes in watch(agents_dir, **watch_kwargs):
        for change_type, path_str in changes:
            path = Path(path_str)

            # Extract agent_id from filename
            agent_id = _extract_agent_id(path)
            if not agent_id:
                continue

            # Ignore delete events - state persists by agent_id
            if change_type == Change.deleted:
                continue

            # Process added or modified events
            if change_type in (Change.added, Change.modified):
                # Get or create state for this agent
                if agent_id not in agent_states:
                    agent_states[agent_id] = AgentState()

                # Process the file
                should_stop = _process_agent_file(
                    path, agent_id, agent_states[agent_id], on_event
                )

                # Clean up agent state when done, but continue watching other agents
                if should_stop:
                    del agent_states[agent_id]


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
    agent_file = agents_dir / f"{agent_id}.jsonl"

    if not agent_file.exists():
        raise FileNotFoundError(f"Agent log not found: {agent_file}")

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
