"""WebSocket server API for managing running think/agents instances.

This module provides a WebSocket API to:
- List running agents
- Attach to running agents for live events
- Detach from live events
- Spawn new agents

Cortex only manages actively running agents. Historical agent data should be
accessed directly from <journal>/agents/<ts>.jsonl files.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from flask_sock import Sock
from simple_websocket import ConnectionClosed

from think.utils import get_personas


class RunningAgent:
    """Represents a running agent process."""

    def __init__(
        self,
        agent_id: str,
        process: subprocess.Popen,
        log_path: Path,
        request: Optional[Dict[str, Any]] = None,
    ):
        self.agent_id = agent_id
        self.process = process
        self.log_path = log_path
        self.started_at = int(time.time() * 1000)
        self.status = "running"
        self.watchers: Set[Any] = set()  # WebSocket connections watching this agent
        self.stop_event = threading.Event()
        self.events: List[Dict[str, Any]] = []  # All events from this agent in memory
        self.lock = threading.RLock()  # Lock for thread-safe event list access
        self.request = request or {}  # Store original request for handoff detection

    def is_running(self) -> bool:
        """Check if the agent process is still running."""
        return self.process.poll() is None and not self.stop_event.is_set()

    def stop(self) -> None:
        """Stop the agent process."""
        self.stop_event.set()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.status = "stopped"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.agent_id,
            "status": self.status,
            "started_at": self.started_at,
            "pid": self.process.pid if self.process.poll() is None else None,
            "metadata": {},  # Running agents don't have stored metadata
        }


class CortexServer:
    """WebSocket server for managing think/agents instances."""

    def __init__(self, path: str = "/ws/cortex"):
        self.path = path
        self.running_agents: Dict[str, RunningAgent] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

    def register(self, sock: Sock) -> None:
        """Register WebSocket routes with Flask-Sock."""

        @sock.route(self.path, endpoint="cortex_ws")
        def _handler(ws) -> None:
            self._handle_connection(ws)

    def start(self) -> None:
        """Start the cortex server (for backwards compatibility)."""

    def _handle_connection(self, ws) -> None:
        """Handle a new WebSocket connection."""
        attached_agent: Optional[str] = None

        try:
            while ws.connected:
                try:
                    msg = ws.receive(timeout=1)
                    if msg is None:
                        continue

                    req = json.loads(msg)
                    action = req.get("action")

                    if action == "list":
                        self._handle_list(ws, req)
                    elif action == "attach":
                        agent_id = req.get("agent_id")
                        if agent_id:
                            attached_agent = self._handle_attach(
                                ws, agent_id, attached_agent
                            )
                    elif action == "detach":
                        attached_agent = self._handle_detach(ws, attached_agent)
                    elif action == "spawn":
                        self._handle_spawn(ws, req)
                    else:
                        self._send_error(ws, f"Unknown action: {action}")

                except json.JSONDecodeError as e:
                    self._send_error(ws, f"Invalid JSON: {e}")
                except Exception as e:
                    self.logger.exception("Error handling WebSocket message")
                    self._send_error(ws, f"Internal error: {e}")

        except ConnectionClosed:
            pass
        finally:
            # Clean up: remove from any agent watchers
            if attached_agent:
                self._cleanup_watcher(ws, attached_agent)

    def _handle_list(self, ws, req: Dict[str, Any]) -> None:
        """Handle list agents request with pagination support."""
        # Extract pagination parameters
        limit = req.get("limit", 10)
        offset = req.get("offset", 0)

        # Validate parameters
        try:
            limit = max(1, min(int(limit), 100))  # Limit between 1-100
            offset = max(0, int(offset))
        except (ValueError, TypeError):
            self._send_error(ws, "Invalid limit or offset parameters")
            return

        try:
            agents, total_count = self._get_running_agents_with_pagination(
                limit, offset
            )
            response = {
                "type": "agent_list",
                "agents": agents,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": total_count,
                    "has_more": offset + limit < total_count,
                },
            }
            self._send_message(ws, response)
        except Exception as e:
            self.logger.exception("Error handling list request")
            self._send_error(ws, f"Error fetching agents: {e}")

    def _handle_attach(self, ws, agent_id: str, current_attached: Optional[str]) -> str:
        """Handle attach to agent request."""
        # First detach from current agent if any
        if current_attached:
            self._handle_detach(ws, current_attached)

        # Only support running agents
        with self.lock:
            running_agent = self.running_agents.get(agent_id)
            if not running_agent:
                self._send_error(ws, f"Agent {agent_id} not found or not running")
                return current_attached or ""

            if not running_agent.is_running():
                self._send_error(ws, f"Agent {agent_id} is not running")
                return current_attached or ""

            # Add to watchers for live updates
            running_agent.watchers.add(ws)

            # Send attach confirmation
            self._send_message(ws, {"type": "attached", "agent_id": agent_id})

            # Send all in-memory events for running agent
            with running_agent.lock:
                for event_data in running_agent.events:
                    message = {
                        "type": "agent_event",
                        "agent_id": agent_id,
                        "event": event_data,
                    }
                    self._send_message(ws, message)

            return agent_id

    def _handle_detach(self, ws, attached_agent: Optional[str]) -> None:
        """Handle detach from agent request."""
        if attached_agent:
            self._cleanup_watcher(ws, attached_agent)
            self._send_message(ws, {"type": "detached"})
        return None

    def _handle_spawn(self, ws, req: Dict[str, Any]) -> None:
        """Handle spawn new agent request."""
        prompt = req.get("prompt", "")
        backend = req.get("backend", "openai")
        persona = req.get("persona", "default")
        config = req.get("config", {})

        if not prompt:
            self._send_error(ws, "prompt is required")
            return

        # Generate agent ID (timestamp)
        agent_id = str(int(time.time() * 1000))

        # Set up journal log path
        journal = os.getenv("JOURNAL_PATH")
        if not journal:
            self._send_error(ws, "JOURNAL_PATH not configured")
            return

        log_path = Path(journal) / "agents" / f"{agent_id}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create empty log file - agent will write its own start event
        try:
            # Just ensure the file exists
            log_path.touch()
            self.logger.debug(f"Created log file for agent {agent_id}: {log_path}")
        except Exception as e:
            self.logger.error(f"Failed to create log file for agent {agent_id}: {e}")
            self._send_error(ws, f"Failed to create agent log file: {e}")
            return

        # Spawn the agent process
        try:
            # Build NDJSON request with config
            request = {
                "prompt": prompt,
                "backend": backend,
                "persona": persona,
                "config": config,
            }

            ndjson_input = json.dumps(request)

            cmd = ["think-agents"]

            env = os.environ.copy()
            env["JOURNAL_PATH"] = journal

            # Log the command and input in verbose mode
            self.logger.info(f"Spawning agent {agent_id} with NDJSON: {ndjson_input}")

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,
            )

            # Send the NDJSON input and close stdin
            process.stdin.write(ndjson_input + "\n")
            process.stdin.close()

            # Create running agent entry (agent will emit its own start event)
            # Store the original request for handoff detection
            agent = RunningAgent(agent_id, process, log_path, request=request)

            with self.lock:
                self.running_agents[agent_id] = agent

            # Start monitoring threads
            threading.Thread(
                target=self._monitor_stdout, args=(agent,), daemon=True
            ).start()
            threading.Thread(
                target=self._monitor_stderr, args=(agent,), daemon=True
            ).start()

            self._send_message(ws, {"type": "agent_spawned", "agent_id": agent_id})

        except Exception as e:
            self.logger.exception(f"Failed to spawn agent: {e}")
            self._send_error(ws, f"Failed to spawn agent: {e}")

    def _monitor_stdout(self, agent: RunningAgent) -> None:
        """Monitor agent stdout and write events to log file."""
        if not agent.process.stdout:
            return

        try:
            for line in agent.process.stdout:
                if not line:
                    continue

                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse JSON event from stdout
                    event_data = json.loads(line)

                    # Check for handoff on finish events
                    if event_data.get("event") == "finish" and "result" in event_data:
                        handoff = self._check_for_handoff(agent)
                        if handoff:
                            # Add handoff info to finish event
                            event_data["handoff"] = handoff

                    # Check for start events to add handoff_from if applicable
                    elif event_data.get("event") == "start":
                        if agent.request.get("handoff_from"):
                            event_data["handoff_from"] = agent.request["handoff_from"]

                    # Store in memory for running agents
                    with agent.lock:
                        agent.events.append(event_data)

                    # Write to log file
                    with open(agent.log_path, "a") as f:
                        f.write(json.dumps(event_data) + "\n")

                    # Broadcast to watchers
                    self._broadcast_agent_event(agent.agent_id, event_data)

                    # Spawn handoff agent after finish event is fully processed
                    if event_data.get("event") == "finish" and "handoff" in event_data:
                        # Slight delay to ensure finish event is fully processed
                        time.sleep(0.1)
                        self._spawn_handoff_agent(
                            agent.agent_id, event_data["result"], event_data["handoff"]
                        )

                except json.JSONDecodeError as e:
                    self.logger.debug(
                        f"Agent {agent.agent_id} invalid JSON output: {e}, line: {line}"
                    )
                    # Still write non-JSON output as an info event
                    info_event = {
                        "event": "info",
                        "ts": int(time.time() * 1000),
                        "message": line,
                    }

                    # Store in memory
                    with agent.lock:
                        agent.events.append(info_event)

                    with open(agent.log_path, "a") as f:
                        f.write(json.dumps(info_event) + "\n")
                    self._broadcast_agent_event(agent.agent_id, info_event)

        except Exception as e:
            self.logger.error(
                f"Error monitoring stdout for agent {agent.agent_id}: {e}"
            )
        finally:
            # Clean up when stdout closes
            exit_code = agent.process.poll()
            if exit_code is not None:
                self.logger.info(f"Agent {agent.agent_id} exited with code {exit_code}")

                # Check if agent already emitted a finish event
                has_finish_event = False
                with agent.lock:
                    for event in agent.events:
                        if event.get("event") in ["finish", "error"]:
                            has_finish_event = True
                            break

                # If no finish event was emitted, create one
                if not has_finish_event:
                    self.logger.warning(
                        f"Agent {agent.agent_id} exited without emitting finish event"
                    )
                    finish_event = {
                        "event": "error",
                        "ts": int(time.time() * 1000),
                        "exit_code": exit_code,
                        "error": f"Agent exited with code {exit_code} without emitting finish event",
                    }

                    # Store in memory
                    with agent.lock:
                        agent.events.append(finish_event)

                    # Write to log file
                    try:
                        with open(agent.log_path, "a") as f:
                            f.write(json.dumps(finish_event) + "\n")
                    except Exception as e:
                        self.logger.warning(f"Failed to write finish event: {e}")

            with self.lock:
                if agent.agent_id in self.running_agents:
                    agent.status = "finished"
                    # Notify watchers that agent finished
                    for ws in list(agent.watchers):
                        try:
                            self._send_message(
                                ws,
                                {"type": "agent_finished", "agent_id": agent.agent_id},
                            )
                        except ConnectionClosed:
                            pass
                    agent.watchers.clear()

                    # Remove finished agent from memory after notifying watchers
                    # The log file remains for historical access
                    del self.running_agents[agent.agent_id]
                    self.logger.debug(
                        f"Removed finished agent {agent.agent_id} from memory"
                    )

    def _monitor_stderr(self, agent: RunningAgent) -> None:
        """Monitor agent stderr and collect errors for final reporting."""
        if not agent.process.stderr:
            return

        stderr_lines = []
        try:
            for line in agent.process.stderr:
                if not line:
                    continue
                stripped = line.strip()
                if stripped:
                    stderr_lines.append(stripped)
                    # Log to cortex server's stderr for debugging
                    self.logger.debug(f"Agent {agent.agent_id} stderr: {stripped}")

        except Exception as e:
            self.logger.error(
                f"Error monitoring stderr for agent {agent.agent_id}: {e}"
            )
        finally:
            # If there was stderr output and the process exited with an error,
            # send a consolidated error event
            if stderr_lines:
                exit_code = agent.process.poll()
                if exit_code is not None and exit_code != 0:
                    # Create consolidated error event
                    error_event = {
                        "event": "error",
                        "ts": int(time.time() * 1000),
                        "error": "Process failed with stderr output",
                        "trace": "\n".join(stderr_lines),
                        "exit_code": exit_code,
                    }

                    # Store in memory
                    with agent.lock:
                        agent.events.append(error_event)

                    # Broadcast to watchers
                    self._broadcast_agent_event(agent.agent_id, error_event)

                    # Write to the agent's log file
                    try:
                        with open(agent.log_path, "a") as f:
                            f.write(json.dumps(error_event) + "\n")
                    except Exception as e:
                        self.logger.warning(f"Failed to write stderr event to log: {e}")

    # Remove _tail_agent_log method as stdout monitoring replaces it

    def _broadcast_agent_event(self, agent_id: str, event_data: Dict[str, Any]) -> None:
        """Broadcast an agent event to all watchers."""
        with self.lock:
            agent = self.running_agents.get(agent_id)
            if not agent:
                return

            message = {"type": "agent_event", "agent_id": agent_id, "event": event_data}

            # Send to all watchers
            for ws in list(agent.watchers):
                try:
                    self._send_message(ws, message)
                except ConnectionClosed:
                    agent.watchers.discard(ws)

    def _cleanup_watcher(self, ws, agent_id: str) -> None:
        """Remove WebSocket from agent watchers."""
        with self.lock:
            agent = self.running_agents.get(agent_id)
            if agent:
                agent.watchers.discard(ws)

    def _cleanup_dead_agents(self) -> None:
        """Remove agents that are no longer running.

        Note: This is mainly a safety net since agents should be removed
        immediately when they finish in _monitor_stdout.
        """
        dead_agents = []
        for agent_id, agent in self.running_agents.items():
            if not agent.is_running():
                dead_agents.append(agent_id)
                self.logger.warning(
                    f"Found dead agent {agent_id} that wasn't cleaned up properly"
                )

        for agent_id in dead_agents:
            del self.running_agents[agent_id]

    def _check_for_handoff(self, agent: RunningAgent) -> Optional[Dict[str, Any]]:
        """Check if the agent has a handoff configured.

        Returns the handoff configuration if found, None otherwise.
        Priority: request handoff > persona default handoff
        """
        # First check the original request for handoff
        if agent.request.get("handoff"):
            return agent.request["handoff"]

        # Then check persona metadata for default handoff
        persona = agent.request.get("persona", "default")
        try:
            personas = get_personas()
            if persona in personas:
                persona_config = personas[persona].get("config", {})
                if "handoff" in persona_config:
                    return persona_config["handoff"]
        except Exception as e:
            self.logger.debug(f"Error checking persona handoff: {e}")

        return None

    def _spawn_handoff_agent(
        self, parent_agent_id: str, result: str, handoff: Dict[str, Any]
    ) -> None:
        """Spawn a new agent as part of a handoff chain.

        Args:
            parent_agent_id: The ID of the agent that finished
            result: The result from the parent agent (becomes the prompt)
            handoff: The handoff configuration with 'persona' and optional other config
        """
        try:
            # Extract persona and build config from handoff
            persona = handoff.get("persona")
            if not persona:
                self.logger.error("Handoff missing required 'persona' field")
                return

            # Build config from handoff (excluding persona key)
            config = {k: v for k, v in handoff.items() if k != "persona"}

            # Generate new agent ID
            agent_id = str(int(time.time() * 1000))

            # Set up journal log path
            journal = os.getenv("JOURNAL_PATH")
            if not journal:
                self.logger.error("JOURNAL_PATH not configured for handoff")
                return

            log_path = Path(journal) / "agents" / f"{agent_id}.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Create empty log file
            log_path.touch()

            # Build request for the new agent
            request = {
                "prompt": result,  # Use parent's result as prompt
                "persona": persona,
                "config": config,
                "handoff_from": parent_agent_id,  # Track chain lineage
            }

            # Extract backend if specified in config
            backend = config.get("backend", "openai")
            request["backend"] = backend

            # Spawn the agent process
            ndjson_input = json.dumps(request)
            self.logger.info(
                f"Spawning handoff agent {agent_id} from {parent_agent_id}"
            )

            cmd = ["think-agents"]
            env = os.environ.copy()
            env["JOURNAL_PATH"] = journal

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,
            )

            # Send the NDJSON input and close stdin
            process.stdin.write(ndjson_input + "\n")
            process.stdin.close()

            # Create running agent entry
            agent = RunningAgent(agent_id, process, log_path, request=request)

            with self.lock:
                self.running_agents[agent_id] = agent

            # Start monitoring threads
            threading.Thread(
                target=self._monitor_stdout, args=(agent,), daemon=True
            ).start()
            threading.Thread(
                target=self._monitor_stderr, args=(agent,), daemon=True
            ).start()

            self.logger.info(f"Handoff agent {agent_id} spawned successfully")

        except Exception as e:
            self.logger.exception(f"Failed to spawn handoff agent: {e}")

    def _send_message(self, ws, message: Dict[str, Any]) -> None:
        """Send a JSON message to WebSocket."""
        try:
            ws.send(json.dumps(message))
        except ConnectionClosed:
            raise
        except Exception as e:
            self.logger.error(f"Error sending WebSocket message: {e}")
            raise ConnectionClosed()

    def _send_error(self, ws, error_message: str) -> None:
        """Send an error message to WebSocket."""
        self._send_message(ws, {"type": "error", "message": error_message})

    def stop_agent(self, agent_id: str) -> bool:
        """Stop a running agent by ID."""
        with self.lock:
            agent = self.running_agents.get(agent_id)
            if agent and agent.is_running():
                agent.stop()
                return True
        return False

    def get_agent_count(self) -> int:
        """Get count of running agents."""
        with self.lock:
            self._cleanup_dead_agents()
            return len(self.running_agents)

    def _get_running_agents_with_pagination(
        self, limit: int = 10, offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get running agents with pagination."""
        # Get running agents only
        with self.lock:
            self._cleanup_dead_agents()
            running = list(self.running_agents.values())

        # Sort by started_at (newest first)
        running.sort(key=lambda a: a.started_at, reverse=True)

        total_count = len(running)

        # Apply pagination
        paginated_agents = running[offset : offset + limit]

        # Convert to dict format
        agent_dicts = [agent.to_dict() for agent in paginated_agents]

        return agent_dicts, total_count


# Global instance
cortex_server = CortexServer()


def main() -> None:
    """CLI entry point for the cortex server."""
    import argparse

    from flask import Flask
    from flask_sock import Sock

    from think.utils import setup_cli

    parser = argparse.ArgumentParser(description="Sunstone Cortex WebSocket API Server")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=2468, help="Port to bind to (default: 2468)"
    )
    parser.add_argument(
        "--path", default="/ws/cortex", help="WebSocket path (default: /ws/cortex)"
    )

    args = setup_cli(parser)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = Flask(__name__)
    sock = Sock(app)

    # Create and register cortex server
    global cortex_server
    cortex_server = CortexServer(args.path)
    cortex_server.register(sock)

    @app.route("/health")
    def health():
        return {"status": "ok", "running_agents": cortex_server.get_agent_count()}

    @app.route("/")
    def index():
        return {
            "service": "sunstone-cortex",
            "websocket_path": args.path,
            "running_agents": cortex_server.get_agent_count(),
        }

    logging.getLogger(__name__).info(
        f"Starting Cortex server on {args.host}:{args.port} with WebSocket path {args.path}"
    )

    # Write URI file for service discovery
    journal = os.getenv("JOURNAL_PATH")
    if journal:
        from pathlib import Path

        uri_file = Path(journal) / "agents" / "cortex.uri"
        uri_file.parent.mkdir(parents=True, exist_ok=True)
        cortex_uri = f"ws://{args.host}:{args.port}{args.path}"
        uri_file.write_text(cortex_uri)
        logging.getLogger(__name__).info(f"Cortex URI written to {uri_file}")

    try:
        app.run(host=args.host, port=args.port, debug=args.verbose)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Shutting down Cortex server")


if __name__ == "__main__":
    main()
