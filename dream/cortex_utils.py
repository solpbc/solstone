"""Shared utilities for Cortex agent interactions."""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

from .cortex_client import get_global_cortex_client


class CortexAgentRunner:
    """Helper class for running agents via Cortex with event handling."""

    def __init__(self, timeout: int = 60):
        """Initialize the runner.

        Args:
            timeout: Maximum time to wait for agent completion in seconds
        """
        self.timeout = timeout
        self._result: Optional[str] = None
        self._finished = threading.Event()
        self._lock = threading.Lock()
        self._error_message: Optional[str] = None

    def _handle_event(self, event: dict) -> None:
        """Handle events from cortex agent."""
        event_type = event.get("event")

        if event_type == "finish":
            with self._lock:
                self._result = event.get("result", "")
                self._finished.set()
        elif event_type == "error":
            with self._lock:
                error_msg = event.get("error", "Unknown error")
                trace = event.get("trace", "")
                self._error_message = f"{error_msg}\n{trace}" if trace else error_msg
                self._result = None
                self._finished.set()

    def run(
        self,
        prompt: str,
        persona: str = "default",
        backend: str = "openai",
        config: Optional[Dict[str, Any]] = None,
        attachments: Optional[List[str]] = None,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> str:
        """Run an agent via Cortex and wait for result.

        Args:
            prompt: The prompt to send to the agent
            persona: The persona to use (default: "default")
            backend: The backend to use (default: "openai")
            config: Optional configuration dictionary for the agent
            attachments: Optional list of attachments to include with prompt
            on_event: Optional callback for all events (in addition to internal handling)

        Returns:
            The agent's response text

        Raises:
            Exception: If connection fails, agent spawn fails, or agent errors
        """
        # Reset state
        with self._lock:
            self._result = None
            self._error_message = None
            self._finished.clear()

        # Get cortex client
        client = get_global_cortex_client()
        if not client:
            raise Exception("Could not connect to cortex server")

        # Set up event callback - wrap user callback if provided
        def event_handler(event: dict) -> None:
            self._handle_event(event)
            if on_event:
                on_event(event)

        client.set_event_callback(event_handler)

        # Prepare full prompt with attachments
        if attachments:
            full_prompt = "\n".join([prompt] + attachments)
        else:
            full_prompt = prompt

        # Spawn agent via cortex
        agent_id = client.spawn_agent(
            prompt=full_prompt,
            backend=backend,
            persona=persona,
            config=config or {},
        )

        if not agent_id:
            raise Exception("Failed to spawn agent")

        # Attach to the spawned agent to receive events
        if not client.attach_agent(agent_id):
            raise Exception(f"Failed to attach to agent {agent_id}")

        # Wait for agent to finish
        if not self._finished.wait(self.timeout):
            raise Exception(f"Agent timed out after {self.timeout} seconds")

        with self._lock:
            if self._error_message:
                raise Exception(f"Agent error: {self._error_message}")
            if self._result is None:
                raise Exception("Agent completed but no result received")
            return self._result


def run_agent_via_cortex(
    prompt: str,
    persona: str = "default",
    backend: str = "openai",
    config: Optional[Dict[str, Any]] = None,
    attachments: Optional[List[str]] = None,
    timeout: int = 60,
    on_event: Optional[Callable[[dict], None]] = None,
) -> str:
    """Convenience function to run an agent via Cortex.

    Args:
        prompt: The prompt to send to the agent
        persona: The persona to use (default: "default")
        backend: The backend to use (default: "openai")
        config: Optional configuration dictionary for the agent
        attachments: Optional list of attachments to include with prompt
        timeout: Maximum time to wait for completion in seconds (default: 60)
        on_event: Optional callback for agent events

    Returns:
        The agent's response text

    Raises:
        Exception: If connection fails, agent spawn fails, or agent errors
    """
    runner = CortexAgentRunner(timeout=timeout)
    return runner.run(
        prompt=prompt,
        persona=persona,
        backend=backend,
        config=config,
        attachments=attachments,
        on_event=on_event,
    )
