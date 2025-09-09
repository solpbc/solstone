"""Flask-specific synchronous utilities for Cortex agent interactions."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from think.cortex_client import cortex_run as cortex_run_agent


# Note: The async infrastructure has been removed since cortex_client.py
# now provides synchronous functions directly


def run_agent_via_cortex(
    prompt: str,
    persona: str = "default",
    backend: str = "openai",
    config: Optional[Dict[str, Any]] = None,
    attachments: Optional[List[str]] = None,
    timeout: int = 60,
    on_event: Optional[Callable[[dict], None]] = None,
    journal_path: Optional[str] = None,
) -> str:
    """Synchronous helper to run an agent via Cortex.

    This is a convenience function that uses the cortex_client.run_agent
    function to run agents synchronously.

    Args:
        prompt: The prompt to send to the agent
        persona: The persona to use (default: "default")
        backend: The backend to use (default: "openai")
        config: Optional configuration dictionary for the agent
        attachments: Optional list of attachments to include with prompt
        timeout: Maximum time to wait for completion in seconds (default: 60)
        on_event: Optional callback for agent events
        journal_path: Optional journal path (deprecated, ignored)

    Returns:
        The agent's response text

    Raises:
        Exception: If agent spawn fails or agent errors
    """
    # Prepare full prompt with attachments
    if attachments:
        full_prompt = "\n".join([prompt] + attachments)
    else:
        full_prompt = prompt

    # Set timeout in config if specified
    if config is None:
        config = {}
    if timeout and "timeout" not in config:
        config["timeout"] = timeout

    # Run agent using the synchronous cortex_client function
    # Note: journal_path parameter is ignored as JOURNAL_PATH is always in environment
    return cortex_run_agent(
        prompt=full_prompt,
        persona=persona,
        backend=backend,
        config=config,
        on_event=on_event,
    )
