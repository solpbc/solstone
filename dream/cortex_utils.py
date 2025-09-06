"""Flask-specific synchronous utilities for Cortex agent interactions."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from think.cortex_client import CortexClient as AsyncCortexClient
from think.cortex_client import SimpleAgentWatcher
from think.cortex_client import run_agent as async_run_agent
from think.cortex_client import run_agent_with_events as async_run_agent_with_events


class SyncCortexClient:
    """Synchronous wrapper around the async CortexClient for Flask usage."""

    def __init__(self, journal_path: Optional[str] = None):
        self.journal_path = journal_path
        self.client = AsyncCortexClient(journal_path)
        # Create the event loop and thread once during initialization
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._initialized = True

    def _ensure_loop(self):
        """Ensure async event loop is running (now just checks it's still alive)."""
        if not self._initialized:
            raise RuntimeError("SyncCortexClient not properly initialized")
        if self._loop is None:
            raise RuntimeError("Event loop was cleaned up but client is still in use")
        if not self._loop.is_running():
            raise RuntimeError("Event loop stopped unexpectedly")

    def _run_async(self, coro):
        """Run an async coroutine and return the result."""
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def spawn(
        self,
        prompt: str,
        persona: str = "default",
        backend: str = "openai",
        config: Optional[Dict[str, Any]] = None,
        handoff: Optional[Dict[str, Any]] = None,
        handoff_from: Optional[str] = None,
    ) -> str:
        """Spawn a new agent and return agent_id."""
        return self._run_async(
            self.client.spawn(
                prompt=prompt,
                persona=persona,
                backend=backend,
                config=config,
                handoff=handoff,
                handoff_from=handoff_from,
            )
        )

    def list_agents(
        self,
        limit: int = 10,
        offset: int = 0,
        include_active: bool = True,
        agent_type: str = "all",
    ) -> Dict[str, Any]:
        """List agents with pagination and filtering."""
        return self._run_async(
            self.client.list_agents(
                limit=limit,
                offset=offset,
                include_active=include_active,
                agent_type=agent_type,
            )
        )

    def wait_for_completion(self, agent_id: str, timeout: float = 60) -> str:
        """Wait for agent to complete and return result."""
        return self._run_async(
            self.client.wait_for_completion(agent_id, timeout=timeout)
        )

    def get_agent_status(self, agent_id: str) -> str:
        """Get the current status of an agent."""
        return self._run_async(self.client.get_agent_status(agent_id))

    def get_agent_events(self, agent_id: str) -> List[dict]:
        """Get all events for an agent."""
        return self._run_async(self.client.get_agent_events(agent_id))

    def read_events(
        self, agent_id: str, on_event: Callable[[dict], None], follow: bool = True
    ) -> None:
        """Read events from an agent synchronously."""
        return self._run_async(
            self.client.read_events(agent_id, on_event, follow=follow)
        )

    def run_agent(
        self,
        prompt: str,
        persona: str = "default",
        backend: str = "openai",
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Run an agent and wait for completion."""
        # Use the async helper through our event loop
        return self._run_async(
            async_run_agent(
                prompt,
                journal_path=self.journal_path,  # Pass journal_path explicitly
                persona=persona,
                backend=backend,
                config=config,
            )
        )

    def run_agent_with_events(
        self,
        prompt: str,
        on_event: Callable[[dict], None],
        persona: str = "default",
        backend: str = "openai",
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Run an agent and stream events to callback."""
        # Use the async helper through our event loop
        return self._run_async(
            async_run_agent_with_events(
                prompt,
                on_event,
                journal_path=self.journal_path,  # Pass journal_path explicitly
                persona=persona,
                backend=backend,
                config=config,
            )
        )

    def cleanup(self) -> None:
        """Clean up the event loop and thread."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        self._loop = None
        self._thread = None
        self._initialized = False


# Global client instance for Flask app
_sync_client: Optional[SyncCortexClient] = None
_client_lock = threading.Lock()


def get_global_cortex_client(
    journal_path: Optional[str] = None,
) -> Optional[SyncCortexClient]:
    """Get or create global synchronous cortex client instance.

    Args:
        journal_path: Optional journal path to use. If not provided,
                     uses JOURNAL_PATH environment variable.

    Returns:
        SyncCortexClient instance or None if creation fails
    """
    global _sync_client

    with _client_lock:
        if _sync_client is None:
            try:
                _sync_client = SyncCortexClient(journal_path=journal_path)
            except Exception as e:
                # Log the error for debugging
                logging.error(f"Failed to create SyncCortexClient: {e}")
                return None

        return _sync_client


def cleanup_global_cortex_client() -> None:
    """Clean up the global cortex client instance.

    This should be called when the Flask app is shutting down to properly
    clean up the event loop and thread resources.
    """
    global _sync_client

    with _client_lock:
        if _sync_client is not None:
            _sync_client.cleanup()
            _sync_client = None


# Global agent watcher - much simpler!
_agent_watcher_task = None


def start_global_agent_watcher(callback: Callable[[dict], None]) -> None:
    """Start the global agent watcher that broadcasts all events.

    Args:
        callback: Function to call with ALL agent events
    """
    global _agent_watcher_task

    if _agent_watcher_task is not None:
        return  # Already running

    journal_path = os.getenv("JOURNAL_PATH", ".")
    agents_dir = Path(journal_path) / "agents"

    # Get the global client's event loop
    client = get_global_cortex_client()
    if not client or not client._loop:
        raise RuntimeError("Cortex client not initialized")

    # Create and start watcher in the existing event loop
    async def run_watcher():
        watcher = SimpleAgentWatcher(agents_dir, callback)
        await watcher.start()
        # Keep watcher alive
        while True:
            await asyncio.sleep(60)

    _agent_watcher_task = asyncio.run_coroutine_threadsafe(run_watcher(), client._loop)


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

    This is a convenience function that uses the global SyncCortexClient
    to run agents. For better control, use SyncCortexClient directly.

    Args:
        prompt: The prompt to send to the agent
        persona: The persona to use (default: "default")
        backend: The backend to use (default: "openai")
        config: Optional configuration dictionary for the agent
        attachments: Optional list of attachments to include with prompt
        timeout: Maximum time to wait for completion in seconds (default: 60)
        on_event: Optional callback for agent events
        journal_path: Optional journal path (uses JOURNAL_PATH env var if not set)

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

    # Get or create the sync client
    client = get_global_cortex_client(journal_path=journal_path)
    if not client:
        raise RuntimeError("Failed to create Cortex client")

    # Set timeout in config if specified
    if config is None:
        config = {}
    if timeout and "timeout" not in config:
        config["timeout"] = timeout

    # Run agent with or without event streaming
    if on_event:
        return client.run_agent_with_events(
            full_prompt,
            on_event,
            persona=persona,
            backend=backend,
            config=config,
        )
    else:
        return client.run_agent(
            full_prompt,
            persona=persona,
            backend=backend,
            config=config,
        )
