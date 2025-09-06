"""File-based client for Cortex agent system."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class AgentEventWatcher(FileSystemEventHandler):
    """Watch for changes to agent JSONL files."""

    def __init__(
        self,
        file_path: Path,
        event_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ):
        self.file_path = file_path
        self.event_queue = event_queue
        self.last_position = 0
        self.loop = loop

    def on_modified(self, event):
        """Handle file modification events."""
        if event.src_path == str(self.file_path):
            # Queue a check for new events (thread-safe)
            asyncio.run_coroutine_threadsafe(self.event_queue.put("check"), self.loop)


class SimpleAgentWatcher:
    """Simple watcher that monitors all active agents and broadcasts all events."""

    def __init__(
        self,
        agents_dir: Path,
        callback: Callable[[dict], None],
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize watcher for agent directory.

        Args:
            agents_dir: Directory containing agent files
            callback: Function to call with ALL events from ALL agents
            logger: Optional logger instance
        """
        self.agents_dir = agents_dir
        self.callback = callback
        self.logger = logger or logging.getLogger(__name__)
        self.observer = None
        self.active_agents = {}  # agent_id -> last_position
        self.running = False

    async def start(self):
        """Start watching all active agents."""
        if self.running:
            return

        self.running = True
        self.observer = Observer()

        # Set up directory watcher
        handler = FileSystemEventHandler()
        handler.on_modified = self._on_file_modified
        handler.on_created = self._on_file_created
        self.observer.schedule(handler, str(self.agents_dir), recursive=False)
        self.observer.start()

        # Initial scan for active agents
        await self._scan_active_agents()

        # Start polling task
        asyncio.create_task(self._poll_agents())

    async def stop(self):
        """Stop watching."""
        self.running = False
        if self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join(timeout=1)

    def _on_file_modified(self, event):
        """Handle file modification - just mark for checking."""
        if not event.is_directory and event.src_path.endswith(".jsonl"):
            # We'll check it in the next poll cycle
            pass

    def _on_file_created(self, event):
        """Handle new file creation."""
        if not event.is_directory and "_active.jsonl" in event.src_path:
            # New active agent, will be picked up in next poll cycle
            pass

    async def _scan_active_agents(self):
        """Scan for all active agent files."""
        try:
            for path in self.agents_dir.glob("*_active.jsonl"):
                agent_id = path.stem.replace("_active", "")
                if agent_id not in self.active_agents:
                    self.active_agents[agent_id] = 0
                    self.logger.info(f"Started watching agent {agent_id}")
        except Exception as e:
            self.logger.error(f"Error scanning agents: {e}")

    async def _poll_agents(self):
        """Poll all active agents for new events."""
        while self.running:
            try:
                # Check each active agent
                completed = []
                for agent_id, last_pos in list(self.active_agents.items()):
                    active_path = self.agents_dir / f"{agent_id}_active.jsonl"
                    completed_path = self.agents_dir / f"{agent_id}.jsonl"

                    # Determine current file
                    if active_path.exists():
                        file_path = active_path
                    elif completed_path.exists():
                        file_path = completed_path
                    else:
                        # File disappeared
                        completed.append(agent_id)
                        continue

                    # Read new events
                    try:
                        with open(file_path, "r") as f:
                            f.seek(last_pos)
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        event = json.loads(line)
                                        event["agent_id"] = agent_id

                                        # Broadcast event
                                        if asyncio.iscoroutinefunction(self.callback):
                                            await self.callback(event)
                                        else:
                                            self.callback(event)

                                        # Check for completion
                                        if event.get("event") in ["finish", "error"]:
                                            completed.append(agent_id)
                                    except json.JSONDecodeError:
                                        pass

                            self.active_agents[agent_id] = f.tell()

                    except Exception as e:
                        self.logger.error(f"Error reading agent {agent_id}: {e}")

                # Remove completed agents
                for agent_id in completed:
                    del self.active_agents[agent_id]
                    self.logger.info(f"Agent {agent_id} completed")

                # Rescan for new agents periodically
                await self._scan_active_agents()

            except Exception as e:
                self.logger.error(f"Poll error: {e}")

            # Poll every 0.5 seconds
            await asyncio.sleep(0.5)


class CortexClient:
    """File-based client for communicating with Cortex system."""

    def __init__(self, journal_path: Optional[str] = None):
        """Initialize client with journal path."""
        self.journal_path = Path(journal_path or os.getenv("JOURNAL_PATH", "."))
        self.agents_dir = self.journal_path / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass  # No cleanup needed for file-based approach

    async def spawn(
        self,
        prompt: str,
        persona: str = "default",
        backend: str = "openai",
        config: Optional[Dict[str, Any]] = None,
        handoff: Optional[Dict[str, Any]] = None,
        handoff_from: Optional[str] = None,
    ) -> str:
        """Spawn a new agent by writing a request file."""
        # Generate agent ID from timestamp
        agent_id = str(int(time.time() * 1000))

        # Create request object
        request = {
            "event": "request",
            "ts": int(agent_id),
            "prompt": prompt,
            "backend": backend,
            "persona": persona,
            "config": config or {},
        }
        if handoff:
            request["handoff"] = handoff
        if handoff_from:
            request["handoff_from"] = handoff_from

        # Write to pending file
        pending_path = self.agents_dir / f"{agent_id}_pending.jsonl"
        with open(pending_path, "w") as f:
            f.write(json.dumps(request) + "\n")

        # Atomic rename to active (triggers Cortex processing)
        active_path = self.agents_dir / f"{agent_id}_active.jsonl"
        pending_path.rename(active_path)

        self.logger.info(f"Spawned agent {agent_id}")
        return agent_id

    async def list_agents(
        self,
        limit: int = 10,
        offset: int = 0,
        include_active: bool = True,
        agent_type: str = "all",  # "all", "live", "historical"
    ) -> Dict[str, Any]:
        """List agents from the file system with full metadata.

        Args:
            limit: Maximum number of agents to return
            offset: Number of agents to skip
            include_active: Whether to include active agents (deprecated, use agent_type)
            agent_type: Filter by "all", "live", or "historical"
        """
        agents = []

        # Handle backward compatibility - if include_active is False, use historical
        if not include_active and agent_type == "all":
            agent_type = "historical"

        # Collect all agent files based on type
        all_files = []

        if agent_type in ["all", "historical"]:
            # Get completed files (*.jsonl but not *_active.jsonl)
            for file_path in self.agents_dir.glob("*.jsonl"):
                if not file_path.name.endswith("_active.jsonl"):
                    all_files.append(file_path)

        if agent_type in ["all", "live"]:
            # Add active files
            all_files.extend(self.agents_dir.glob("*_active.jsonl"))

        # Sort by modification time (newest first)
        all_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Apply pagination
        total = len(all_files)
        paginated_files = all_files[offset : offset + limit]

        # Build detailed agent info
        for file_path in paginated_files:
            agent_id = file_path.stem.replace("_active", "")
            is_active = "_active" in file_path.name

            try:
                with open(file_path, "r") as f:
                    lines = f.readlines()
                    if not lines:
                        continue

                    # Parse first and last events
                    first_event = json.loads(lines[0])
                    last_event = (
                        json.loads(lines[-1]) if len(lines) > 1 else first_event
                    )

                    # Extract timing info
                    start_ts = first_event.get("ts", int(agent_id))
                    end_ts = last_event.get("ts", start_ts)

                    # Determine status
                    if is_active:
                        status = "running"
                        runtime_seconds = (time.time() * 1000 - start_ts) / 1000
                    else:
                        # Check last event type
                        last_event_type = last_event.get("event")
                        if last_event_type == "error":
                            status = "error"
                        elif last_event_type == "finish":
                            status = "completed"
                        else:
                            status = "interrupted"
                        runtime_seconds = (end_ts - start_ts) / 1000

                    agents.append(
                        {
                            "id": agent_id,
                            "status": status,
                            "is_live": is_active,
                            "persona": first_event.get("persona", "default"),
                            "backend": first_event.get("backend", "openai"),
                            "model": first_event.get("model", ""),
                            "prompt": first_event.get("prompt", ""),
                            "ts": start_ts,
                            "start": start_ts / 1000,
                            "end": end_ts / 1000 if not is_active else None,
                            "runtime_seconds": runtime_seconds,
                            "event_count": len(lines),
                        }
                    )
            except Exception as e:
                self.logger.warning(f"Failed to read agent file {file_path}: {e}")

        # Separate counts
        live_count = sum(1 for a in agents if a["is_live"])
        historical_count = len(agents) - live_count

        return {
            "agents": agents,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total,
                "has_more": (offset + limit) < total,
            },
            "live_count": live_count,
            "historical_count": historical_count,
        }

    async def read_events(
        self, agent_id: str, on_event: Callable[[dict], None], follow: bool = True
    ) -> None:
        """Read events from an agent's JSONL file."""
        # Determine file path (could be active or completed)
        active_path = self.agents_dir / f"{agent_id}_active.jsonl"
        completed_path = self.agents_dir / f"{agent_id}.jsonl"

        file_path = active_path if active_path.exists() else completed_path
        if not file_path.exists():
            raise FileNotFoundError(f"Agent file not found: {agent_id}")

        last_position = 0
        finished = False

        while not finished:
            # Read new events from file
            try:
                with open(file_path, "r") as f:
                    f.seek(last_position)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            event = json.loads(line)

                            # Call event handler
                            if asyncio.iscoroutinefunction(on_event):
                                await on_event(event)
                            else:
                                on_event(event)

                            # Check for completion
                            if event.get("event") in ["finish", "error"]:
                                finished = True

                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Invalid JSON in agent file: {e}")

                    last_position = f.tell()

            except FileNotFoundError:
                # File may have been renamed (active -> completed)
                if file_path == active_path and completed_path.exists():
                    file_path = completed_path
                    continue
                else:
                    raise

            # If not following or finished, break
            if not follow or finished:
                break

            # Check if file was completed (renamed from active)
            if file_path == active_path and not active_path.exists():
                if completed_path.exists():
                    file_path = completed_path
                else:
                    # File disappeared unexpectedly
                    break

            # Small delay before checking for new events
            await asyncio.sleep(0.1)

    async def wait_for_completion(self, agent_id: str, timeout: float = 60) -> str:
        """Wait for agent to complete and return result."""
        result = None
        error = None

        async def handle_event(event: dict):
            nonlocal result, error
            event_type = event.get("event")

            if event_type == "finish":
                result = event.get("result", "")
            elif event_type == "error":
                error = event.get("error", "Unknown error")

        # Read events with timeout
        try:
            await asyncio.wait_for(
                self.read_events(agent_id, handle_event, follow=True), timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Agent {agent_id} timed out after {timeout} seconds")

        if error:
            raise RuntimeError(f"Agent error: {error}")

        return result or ""

    async def get_agent_status(self, agent_id: str) -> str:
        """Get the current status of an agent."""
        active_path = self.agents_dir / f"{agent_id}_active.jsonl"
        completed_path = self.agents_dir / f"{agent_id}.jsonl"

        if active_path.exists():
            return "running"
        elif completed_path.exists():
            # Check if it finished successfully or with error
            with open(completed_path, "r") as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if event.get("event") == "finish":
                            return "completed"
                        elif event.get("event") == "error":
                            return "failed"
                    except json.JSONDecodeError:
                        continue
            return "completed"  # Default if no finish/error event
        else:
            return "not_found"

    async def get_agent_events(self, agent_id: str) -> List[dict]:
        """Get all events for an agent."""
        events = []

        # Determine file path
        active_path = self.agents_dir / f"{agent_id}_active.jsonl"
        completed_path = self.agents_dir / f"{agent_id}.jsonl"

        file_path = active_path if active_path.exists() else completed_path
        if not file_path.exists():
            raise FileNotFoundError(f"Agent file not found: {agent_id}")

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON in agent file: {e}")

        return events


# Helper functions for common patterns
async def run_agent(prompt: str, **kwargs) -> str:
    """Simple helper to spawn and wait for an agent."""
    # Extract journal_path if present
    journal_path = kwargs.pop("journal_path", None)

    async with CortexClient(journal_path=journal_path) as client:
        agent_id = await client.spawn(prompt, **kwargs)
        return await client.wait_for_completion(agent_id)


async def run_agent_with_events(
    prompt: str, on_event: Callable[[dict], None], **kwargs
) -> str:
    """Run agent and stream events to callback."""
    # Extract journal_path if present
    journal_path = kwargs.pop("journal_path", None)

    async with CortexClient(journal_path=journal_path) as client:
        agent_id = await client.spawn(prompt, **kwargs)

        result = None
        error = None

        async def handle_event(event: dict):
            nonlocal result, error

            # Add agent_id to event if not present
            if "agent_id" not in event:
                event["agent_id"] = agent_id

            # Forward to user callback
            if asyncio.iscoroutinefunction(on_event):
                await on_event(event)
            else:
                on_event(event)

            # Track completion
            event_type = event.get("event")
            if event_type == "finish":
                result = event.get("result", "")
            elif event_type == "error":
                error = event.get("error", "Unknown error")

        await client.read_events(agent_id, handle_event, follow=True)

        if error:
            raise RuntimeError(f"Agent error: {error}")

        return result or ""
