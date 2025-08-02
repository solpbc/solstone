"""WebSocket client for connecting dream backend to cortex server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import websockets


class CortexClient:
    """Synchronous WebSocket client for communicating with cortex server."""

    def __init__(self, uri: str = "ws://127.0.0.1:2468/ws/cortex"):
        self.uri = uri
        self.connected = False
        self.lock = threading.RLock()
        self.attached_agent: Optional[str] = None
        self.event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.logger = logging.getLogger(__name__)

        # Async components
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connection_task: Optional[asyncio.Task] = None
        self._thread: Optional[threading.Thread] = None

        # Response handling
        self._responses: Dict[str, Any] = {}
        self._response_event = threading.Event()

    def connect(self) -> bool:
        """Connect to cortex server (synchronous)."""
        if self.connected:
            return True

        # Start async event loop in background thread
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()

        # Wait for connection
        timeout = 10
        start_time = time.time()
        while not self.connected and time.time() - start_time < timeout:
            time.sleep(0.1)

        return self.connected

    def disconnect(self) -> None:
        """Disconnect from cortex server."""
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._async_disconnect(), self._loop)

        # Wait a bit for cleanup
        time.sleep(0.5)
        self.connected = False

    def spawn_agent(
        self,
        prompt: str,
        backend: str = "openai",
        model: str = "",
        persona: str = "default",
        max_tokens: int = 0,
    ) -> Optional[str]:
        """Spawn a new agent and return agent_id (synchronous)."""
        if not self.connected or not self._loop:
            return None

        request = {
            "action": "spawn",
            "prompt": prompt,
            "backend": backend,
            "persona": persona,
        }

        if model:
            request["model"] = model
        if max_tokens:
            request["max_tokens"] = max_tokens

        # Send request asynchronously
        future = asyncio.run_coroutine_threadsafe(
            self._send_request(request), self._loop
        )

        try:
            future.result(timeout=5)
            return None  # Agent ID will come via callback
        except Exception as e:
            self.logger.error(f"Failed to spawn agent: {e}")
            return None

    def attach_agent(self, agent_id: str) -> bool:
        """Attach to an agent for real-time events."""
        if not self.connected or not self._loop:
            return False

        request = {"action": "attach", "agent_id": agent_id}

        future = asyncio.run_coroutine_threadsafe(
            self._send_request(request), self._loop
        )

        try:
            future.result(timeout=5)
            with self.lock:
                self.attached_agent = agent_id
            return True
        except Exception as e:
            self.logger.error(f"Failed to attach to agent: {e}")
            return False

    def detach_agent(self) -> bool:
        """Detach from current agent."""
        if not self.connected or not self._loop:
            return False

        request = {"action": "detach"}

        future = asyncio.run_coroutine_threadsafe(
            self._send_request(request), self._loop
        )

        try:
            future.result(timeout=5)
            with self.lock:
                self.attached_agent = None
            return True
        except Exception as e:
            self.logger.error(f"Failed to detach from agent: {e}")
            return False

    def list_agents(self, limit: int = 10, offset: int = 0) -> Optional[Dict[str, Any]]:
        """List agents with pagination (synchronous)."""
        if not self.connected or not self._loop:
            return None

        request = {
            "action": "list",
            "limit": limit,
            "offset": offset,
        }

        # Clear previous response
        self._responses.clear()
        self._response_event.clear()

        # Send request asynchronously
        future = asyncio.run_coroutine_threadsafe(
            self._send_request(request), self._loop
        )

        try:
            future.result(timeout=5)
            # Wait for response
            if self._response_event.wait(timeout=5):
                return self._responses.get("agent_list")
            return None
        except Exception as e:
            self.logger.error(f"Failed to list agents: {e}")
            return None

    def set_event_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Set callback function for agent events."""
        self.event_callback = callback

    def _run_async_loop(self) -> None:
        """Run the async event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._async_connect_and_listen())
        except Exception as e:
            self.logger.error(f"Error in async loop: {e}")
        finally:
            self._loop.close()

    async def _async_connect_and_listen(self) -> None:
        """Connect to cortex and listen for messages."""
        try:
            self._ws = await websockets.connect(self.uri)
            self.connected = True
            self.logger.info("Connected to cortex server")

            # Listen for messages
            async for message in self._ws:
                await self._handle_message(message)

        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Cortex connection closed")
            self.connected = False
        except Exception as e:
            self.logger.error(f"Cortex connection error: {e}")
            self.connected = False

    async def _async_disconnect(self) -> None:
        """Disconnect asynchronously."""
        if self._ws:
            await self._ws.close()
        self.connected = False

    async def _send_request(self, request: Dict[str, Any]) -> None:
        """Send request to cortex."""
        if self._ws:
            await self._ws.send(json.dumps(request))

    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "agent_event" and self.event_callback:
                # Forward agent events to callback
                event_data = data.get("event", {})
                self.event_callback(event_data)

            elif msg_type == "agent_spawned":
                agent_id = data.get("agent_id")
                self.logger.info(f"Agent spawned: {agent_id}")

            elif msg_type == "attached":
                agent_id = data.get("agent_id")
                self.logger.info(f"Attached to agent: {agent_id}")

            elif msg_type == "detached":
                self.logger.info("Detached from agent")

            elif msg_type == "agent_finished":
                agent_id = data.get("agent_id")
                self.logger.info(f"Agent finished: {agent_id}")

            elif msg_type == "agent_list":
                # Store agent list response
                self._responses["agent_list"] = data
                self._response_event.set()

            elif msg_type == "error":
                error_msg = data.get("message", "Unknown error")
                self.logger.error(f"Cortex error: {error_msg}")

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON from cortex: {e}")
        except Exception as e:
            self.logger.error(f"Error handling cortex message: {e}")


def get_cortex_client() -> Optional[CortexClient]:
    """Get cortex client with URI from journal agents directory."""
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return None

    uri_file = Path(journal) / "agents" / "cortex.uri"
    if uri_file.exists():
        try:
            cortex_uri = uri_file.read_text().strip()
            return CortexClient(cortex_uri)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to read cortex URI: {e}")

    # Fallback to default
    return CortexClient()


# Global client instance
_cortex_client: Optional[CortexClient] = None
_client_lock = threading.Lock()


def get_global_cortex_client() -> Optional[CortexClient]:
    """Get or create global cortex client instance."""
    global _cortex_client

    with _client_lock:
        if _cortex_client is None:
            _cortex_client = get_cortex_client()
            if _cortex_client and not _cortex_client.connected:
                if not _cortex_client.connect():
                    _cortex_client = None

        return _cortex_client
