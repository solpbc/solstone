"""Synchronous wrapper for the async Cortex client for use in Flask apps."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Dict, List, Optional

from .cortex_client import CortexClient as AsyncCortexClient
from .cortex_client import run_agent as async_run_agent
from .cortex_client import run_agent_with_events as async_run_agent_with_events


class SyncCortexClient:
    """Synchronous wrapper around the async CortexClient for Flask usage."""

    def __init__(self, uri: Optional[str] = None):
        self.client = AsyncCortexClient(uri)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = False

    def _ensure_loop(self):
        """Ensure async event loop is running."""
        if self._loop is None or not self._loop.is_running():
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._thread.start()

    def _run_async(self, coro):
        """Run an async coroutine and return the result."""
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def connect(self) -> None:
        """Connect to Cortex server."""
        if not self._connected:
            self._run_async(self.client.connect())
            self._connected = True

    def disconnect(self) -> None:
        """Disconnect from Cortex server."""
        if self._connected:
            self._run_async(self.client.disconnect())
            self._connected = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def spawn_agent(
        self,
        prompt: str,
        backend: str = "openai",
        persona: str = "default",
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Spawn a new agent and return agent_id."""
        self.connect()
        return self._run_async(
            self.client.spawn(
                prompt=prompt,
                persona=persona,
                backend=backend,
                config=config,
            )
        )

    def list_agents(self, limit: int = 10, offset: int = 0) -> Optional[Dict[str, Any]]:
        """List agents with pagination."""
        self.connect()
        return self._run_async(self.client.list_agents(limit=limit, offset=offset))

    def wait_for_completion(self, agent_id: str, timeout: float = 60) -> str:
        """Wait for agent to complete and return result."""
        self.connect()
        return self._run_async(
            self.client.wait_for_completion(agent_id, timeout=timeout)
        )


# Global client instance for Flask app
_sync_client: Optional[SyncCortexClient] = None
_client_lock = threading.Lock()


def get_global_cortex_client() -> Optional[SyncCortexClient]:
    """Get or create global synchronous cortex client instance."""
    global _sync_client

    with _client_lock:
        if _sync_client is None:
            try:
                _sync_client = SyncCortexClient()
                _sync_client.connect()
            except Exception:
                return None

        return _sync_client


def run_agent_via_cortex(
    prompt: str,
    persona: str = "default",
    backend: str = "openai",
    config: Optional[Dict[str, Any]] = None,
    attachments: Optional[List[str]] = None,
    timeout: int = 60,
    on_event: Optional[Callable[[dict], None]] = None,
) -> str:
    """Synchronous helper to run an agent via Cortex.

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
    # Prepare full prompt with attachments
    if attachments:
        full_prompt = "\n".join([prompt] + attachments)
    else:
        full_prompt = prompt

    # Get or create event loop for async execution
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop running, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Run the async function
    if on_event:
        coro = async_run_agent_with_events(
            full_prompt,
            on_event,
            persona=persona,
            backend=backend,
            config=config,
        )
    else:
        coro = async_run_agent(
            full_prompt,
            persona=persona,
            backend=backend,
            config=config,
        )

    # Execute in the appropriate way
    try:
        # If we're already in an async context, run in executor
        if asyncio.iscoroutinefunction(lambda: None):
            return asyncio.run(coro)
        else:
            # Run in new event loop
            return asyncio.run(coro)
    except RuntimeError:
        # Fallback to thread-based execution
        thread_loop = asyncio.new_event_loop()

        def run_in_thread():
            asyncio.set_event_loop(thread_loop)
            return thread_loop.run_until_complete(coro)

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result(timeout=timeout)
