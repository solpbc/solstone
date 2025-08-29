"""Async WebSocket client for Cortex agent system."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import websockets


class CortexClient:
    """Async WebSocket client for communicating with Cortex server."""

    def __init__(self, uri: Optional[str] = None):
        """Initialize client with URI discovery."""
        self.uri = uri or self._discover_uri()
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _discover_uri() -> str:
        """Discover Cortex URI from journal or use default."""
        journal = os.getenv("JOURNAL_PATH")
        if journal:
            uri_file = Path(journal) / "agents" / "cortex.uri"
            if uri_file.exists():
                try:
                    return uri_file.read_text().strip()
                except Exception:
                    pass
        return "ws://127.0.0.1:2468/ws/cortex"

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Connect to Cortex server."""
        if self.ws:
            return  # Already connected

        try:
            self.ws = await websockets.connect(self.uri)
            self.logger.info(f"Connected to Cortex at {self.uri}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Cortex: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Cortex server."""
        if self.ws:
            await self.ws.close()
            self.ws = None
            self.logger.info("Disconnected from Cortex")

    async def spawn(
        self,
        prompt: str,
        persona: str = "default",
        backend: str = "openai",
        config: Optional[Dict[str, Any]] = None,
        handoff: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Spawn a new agent and return agent_id."""
        if not self.ws:
            raise RuntimeError("Not connected to Cortex")

        request = {
            "action": "spawn",
            "prompt": prompt,
            "persona": persona,
            "backend": backend,
            "config": config or {},
        }
        if handoff:
            request["handoff"] = handoff

        await self.ws.send(json.dumps(request))

        # Wait for agent_spawned response
        response = await self.ws.recv()
        data = json.loads(response)

        if data.get("type") == "agent_spawned":
            return data.get("agent_id")
        elif data.get("type") == "error":
            raise RuntimeError(f"Spawn failed: {data.get('message')}")
        else:
            raise RuntimeError(f"Unexpected response: {data}")

    async def attach(self, agent_id: str, on_event: Callable[[dict], None]) -> None:
        """Attach to agent for real-time events."""
        if not self.ws:
            raise RuntimeError("Not connected to Cortex")

        request = {"action": "attach", "agent_id": agent_id}
        await self.ws.send(json.dumps(request))

        # Start listening for events
        async for message in self.ws:
            data = json.loads(message)

            if data.get("type") == "attached":
                self.logger.info(f"Attached to agent {agent_id}")
            elif data.get("type") == "agent_event":
                event = data.get("event", {})
                if asyncio.iscoroutinefunction(on_event):
                    await on_event(event)
                else:
                    on_event(event)
            elif data.get("type") == "agent_finished":
                self.logger.info(f"Agent {agent_id} finished")
                break
            elif data.get("type") == "error":
                self.logger.error(f"Error: {data.get('message')}")
                break

    async def detach(self) -> None:
        """Detach from current agent."""
        if not self.ws:
            return

        request = {"action": "detach"}
        await self.ws.send(json.dumps(request))

        response = await self.ws.recv()
        data = json.loads(response)

        if data.get("type") == "detached":
            self.logger.info("Detached from agent")
        elif data.get("type") == "error":
            self.logger.error(f"Detach failed: {data.get('message')}")

    async def list_agents(
        self, limit: int = 10, offset: int = 0
    ) -> Optional[Dict[str, Any]]:
        """List agents with pagination."""
        if not self.ws:
            raise RuntimeError("Not connected to Cortex")

        request = {
            "action": "list",
            "limit": limit,
            "offset": offset,
        }
        await self.ws.send(json.dumps(request))

        response = await self.ws.recv()
        data = json.loads(response)

        if data.get("type") == "agent_list":
            return data
        elif data.get("type") == "error":
            self.logger.error(f"List failed: {data.get('message')}")
            return None
        else:
            return None

    async def wait_for_completion(self, agent_id: str, timeout: float = 60) -> str:
        """Wait for agent to complete and return result."""
        result = None
        error = None
        finished = asyncio.Event()

        async def handle_event(event: dict):
            nonlocal result, error
            event_type = event.get("event")

            if event_type == "finish":
                result = event.get("result", "")
                finished.set()
            elif event_type == "error":
                error = event.get("error", "Unknown error")
                finished.set()

        # Attach to agent and wait
        attach_task = asyncio.create_task(self.attach(agent_id, handle_event))

        try:
            await asyncio.wait_for(finished.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Agent {agent_id} timed out after {timeout} seconds")
        finally:
            attach_task.cancel()
            try:
                await attach_task
            except asyncio.CancelledError:
                pass

        if error:
            raise RuntimeError(f"Agent error: {error}")

        return result or ""


# Helper functions for common patterns
async def run_agent(prompt: str, **kwargs) -> str:
    """Simple helper to spawn and wait for an agent."""
    async with CortexClient() as client:
        agent_id = await client.spawn(prompt, **kwargs)
        return await client.wait_for_completion(agent_id)


async def run_agent_with_events(
    prompt: str, on_event: Callable[[dict], None], **kwargs
) -> str:
    """Run agent and stream events to callback."""
    async with CortexClient() as client:
        agent_id = await client.spawn(prompt, **kwargs)

        result = None
        error = None
        finished = asyncio.Event()

        async def handle_event(event: dict):
            nonlocal result, error
            # Forward to user callback
            if asyncio.iscoroutinefunction(on_event):
                await on_event(event)
            else:
                on_event(event)

            # Track completion
            event_type = event.get("event")
            if event_type == "finish":
                result = event.get("result", "")
                finished.set()
            elif event_type == "error":
                error = event.get("error", "Unknown error")
                finished.set()

        attach_task = asyncio.create_task(client.attach(agent_id, handle_event))

        await finished.wait()
        attach_task.cancel()
        try:
            await attach_task
        except asyncio.CancelledError:
            pass

        if error:
            raise RuntimeError(f"Agent error: {error}")

        return result or ""
