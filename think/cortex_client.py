"""Cortex client for managing AI agent requests."""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from watchfiles import Change, watch


@dataclass
class FileState:
    """Track state of a watched file to handle atomic renames properly."""

    position: int
    inode: int
    size: int
    partial_line: str = ""  # Buffer for incomplete lines


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
    while (agents_dir / f"{ts}_active.jsonl").exists() or (agents_dir / f"{ts}_pending.jsonl").exists():
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


def cortex_watch(
    on_event: Callable[[Dict[str, Any]], Optional[bool]],
    stop_event: Optional[Any] = None,
) -> None:
    """Watch for Cortex agent events and emit callbacks.

    This function blocks and watches for any *_active.jsonl files in the agents
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

    # Track file state to handle atomic renames properly
    file_states: Dict[Path, FileState] = {}

    # Initial scan for existing active files
    for active_file in agents_dir.glob("*_active.jsonl"):
        if active_file.is_file():
            # Start from current end of file for existing files
            stat = active_file.stat()
            file_states[active_file] = FileState(
                position=stat.st_size, inode=stat.st_ino, size=stat.st_size
            )

    # Watch for changes in the agents directory
    watch_kwargs = {"raise_interrupt": False}
    if stop_event is not None:
        watch_kwargs["stop_event"] = stop_event

    for changes in watch(agents_dir, **watch_kwargs):
        for change_type, path_str in changes:
            path = Path(path_str)

            # Only process .jsonl files
            if not path.name.endswith(".jsonl"):
                continue

            # Handle new active files
            if change_type == Change.added and path.name.endswith("_active.jsonl"):
                try:
                    stat = path.stat()
                    # Check if this is an atomic rename (same inode we've seen before)
                    existing_state = None
                    for tracked_path, state in list(file_states.items()):
                        if state.inode == stat.st_ino:
                            # This file was renamed, preserve reading position
                            existing_state = state
                            # Remove old path entry
                            del file_states[tracked_path]
                            break

                    if existing_state:
                        # File was renamed atomically, continue from last position
                        file_states[path] = FileState(
                            position=existing_state.position,
                            inode=stat.st_ino,
                            size=stat.st_size,
                            partial_line=existing_state.partial_line,  # Preserve partial line
                        )
                    else:
                        # Truly new file, start from beginning
                        file_states[path] = FileState(
                            position=0, inode=stat.st_ino, size=stat.st_size
                        )
                except (OSError, IOError):
                    pass  # File might have been removed already

            # Handle modified files (new content appended)
            if change_type in (Change.added, Change.modified):
                if path in file_states or path.name.endswith("_active.jsonl"):
                    # Read new content from last position
                    try:
                        with open(path, "r") as f:
                            # Get current state or create new one
                            state = file_states.get(path)
                            if not state:
                                stat = path.stat()
                                state = FileState(
                                    position=0, inode=stat.st_ino, size=stat.st_size
                                )
                                file_states[path] = state

                            # Seek to last read position
                            f.seek(state.position)

                            # Read new content
                            new_content = f.read()
                            if not new_content:
                                continue

                            # Combine with any partial line from last read
                            content = state.partial_line + new_content

                            # Split into lines, keeping the last incomplete line
                            lines = content.split("\n")

                            # If content doesn't end with newline, last element is partial
                            if not content.endswith("\n"):
                                state.partial_line = lines[-1]
                                lines = lines[:-1]
                            else:
                                state.partial_line = ""

                            # Process complete lines
                            for line in lines:
                                line = line.strip()
                                if not line:
                                    continue

                                try:
                                    # Parse JSON event
                                    event = json.loads(line)

                                    # Call the callback with error protection
                                    try:
                                        result = on_event(event)

                                        # If callback returns False, stop watching
                                        if result is False:
                                            return
                                    except Exception:
                                        # Treat callback errors as request to stop
                                        return
                                except json.JSONDecodeError:
                                    # Skip malformed JSON lines
                                    continue

                            # Update position to end of file
                            state.position = f.tell()
                            state.size = f.tell()
                    except (OSError, IOError):
                        # File might have been renamed/removed, remove from tracking
                        file_states.pop(path, None)
                        continue  # Don't re-raise, just skip this file
                    except Exception:
                        # Unexpected error, log and continue
                        file_states.pop(path, None)
                        continue  # Continue watching other files

            # Handle renamed/removed files
            if change_type == Change.deleted:
                # File was likely renamed from active to completed
                # Don't remove immediately - might be atomic rename
                # The inode tracking will handle this
                pass


def cortex_run(
    prompt: str,
    persona: str = "default",
    backend: str = "openai",
    config: Optional[Dict[str, Any]] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> str:
    """Run an agent synchronously and return the result.

    This function creates a Cortex request, watches for events, and returns
    the final result when the agent completes.

    Args:
        prompt: The task or question for the agent
        persona: Agent persona from think/agents/*.txt (default: "default")
        backend: AI backend - openai, google, anthropic, or claude (default: "openai")
        config: Backend-specific configuration
        on_event: Optional callback for streaming events

    Returns:
        The agent's final result text

    Raises:
        RuntimeError: If agent errors or times out
    """
    # Create the request
    active_file = cortex_request(
        prompt=prompt,
        persona=persona,
        backend=backend,
        config=config,
    )

    # Extract agent_id from the filename (timestamp)
    agent_id = Path(active_file).stem.replace("_active", "")

    # Track result and errors
    result = None
    error = None
    seen_finish = False

    def event_handler(event: Dict[str, Any]) -> Optional[bool]:
        nonlocal result, error, seen_finish

        # Check if this event is for our agent
        event_agent_id = event.get("agent_id", "")
        if not event_agent_id:
            # Try to infer from the event if no agent_id field
            # Events in the same file are for the same agent
            pass
        elif event_agent_id != agent_id:
            return True  # Continue watching, not our agent

        # Call the user's callback if provided
        if on_event:
            on_event(event)

        # Handle finish and error events
        event_type = event.get("event")
        if event_type == "finish":
            result = event.get("result", "")
            seen_finish = True
            return False  # Stop watching
        elif event_type == "error":
            error = event.get("error", "Agent error")
            seen_finish = True
            return False  # Stop watching

        return True  # Continue watching

    # Watch for events
    cortex_watch(event_handler)

    # Check results
    if error:
        raise RuntimeError(f"Agent error: {error}")
    elif result is not None:
        return result
    else:
        raise RuntimeError("Agent did not complete properly")


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
