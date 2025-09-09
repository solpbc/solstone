"""Cortex client for managing AI agent requests."""

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from watchfiles import Change, watch


def cortex_request(
    prompt: str,
    persona: str,
    backend: Optional[str] = None,
    handoff_from: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a Cortex agent request file.
    
    Args:
        prompt: The task or question for the agent
        persona: Agent persona from think/agents/*.txt
        backend: AI backend - openai, google, anthropic, or claude
        handoff_from: Previous agent ID if this is a handoff request
        config: Backend-specific configuration (model, max_tokens, domain for Claude)
    
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
    
    # Generate timestamp in milliseconds
    ts = int(time.time() * 1000)
    
    # Build request object
    request = {
        "event": "request",
        "ts": ts,
        "prompt": prompt,
        "backend": backend,
        "persona": persona,
    }
    
    # Add optional fields
    if config:
        request["config"] = config
    
    if handoff_from:
        request["handoff_from"] = handoff_from
    
    # Write to pending file
    pending_file = agents_dir / f"{ts}_pending.jsonl"
    with open(pending_file, "w") as f:
        json.dump(request, f)
        f.write("\n")
    
    # Rename to active to trigger Cortex processing (atomic operation)
    active_file = agents_dir / f"{ts}_active.jsonl"
    pending_file.rename(active_file)
    
    return str(active_file)


def cortex_watch(on_event: Callable[[Dict[str, Any]], Optional[bool]]) -> None:
    """Watch for Cortex agent events and emit callbacks.
    
    This function blocks and watches for any *_active.jsonl files in the agents
    directory, efficiently tailing them for new events. When new events are
    written, they are parsed and passed to the on_event callback.
    
    Args:
        on_event: Callback function that receives each event as a dict.
                 Should return False to stop watching, True/None to continue.
    
    The callback receives event dictionaries with at least:
        - event: Event type (request, start, tool_start, tool_end, finish, error, etc.)
        - ts: Millisecond timestamp
        - Additional fields depend on event type
    """
    # Get journal path from environment
    journal_path = os.environ.get("JOURNAL_PATH")
    if not journal_path:
        raise ValueError("JOURNAL_PATH environment variable not set")
    
    agents_dir = Path(journal_path) / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    
    # Track file positions to only read new content
    file_positions: Dict[Path, int] = {}
    
    # Initial scan for existing active files
    for active_file in agents_dir.glob("*_active.jsonl"):
        if active_file.is_file():
            # Start from current end of file for existing files
            file_positions[active_file] = active_file.stat().st_size
    
    # Watch for changes in the agents directory
    for changes in watch(agents_dir):
        for change_type, path_str in changes:
            path = Path(path_str)
            
            # Only process .jsonl files
            if not path.name.endswith(".jsonl"):
                continue
            
            # Handle new active files
            if change_type == Change.added and path.name.endswith("_active.jsonl"):
                # Start reading from beginning for new files
                file_positions[path] = 0
            
            # Handle modified files (new content appended)
            if change_type in (Change.added, Change.modified):
                if path in file_positions or path.name.endswith("_active.jsonl"):
                    # Read new content from last position
                    try:
                        with open(path, "r") as f:
                            # Seek to last read position
                            last_pos = file_positions.get(path, 0)
                            f.seek(last_pos)
                            
                            # Read new lines
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                
                                try:
                                    # Parse JSON event
                                    event = json.loads(line)
                                    
                                    # Call the callback
                                    result = on_event(event)
                                    
                                    # If callback returns False, stop watching
                                    if result is False:
                                        return
                                except json.JSONDecodeError:
                                    # Skip malformed JSON lines
                                    continue
                            
                            # Update position
                            file_positions[path] = f.tell()
                    except (OSError, IOError):
                        # File might have been renamed/removed, remove from tracking
                        file_positions.pop(path, None)
            
            # Handle renamed/removed files
            if change_type == Change.deleted:
                # File was likely renamed from active to completed
                file_positions.pop(path, None)


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