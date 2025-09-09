"""Cortex client for managing AI agent requests."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


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