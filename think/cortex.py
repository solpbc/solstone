"""WebSocket server API for managing running think/agents instances.

This module provides a WebSocket API to:
- List running agents
- Attach to agents and get full history + live events
- Detach from live events
- Spawn new agents

Agents are identified by their timestamp ID used in <journal>/agents/<ts>.jsonl files.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

from flask_sock import Sock
from simple_websocket import ConnectionClosed


class RunningAgent:
    """Represents a running agent process."""

    def __init__(self, agent_id: str, process: subprocess.Popen, log_path: Path):
        self.agent_id = agent_id
        self.process = process
        self.log_path = log_path
        self.started_at = int(time.time() * 1000)
        self.status = "running"
        self.watchers: Set[Any] = set()  # WebSocket connections watching this agent
        self.stop_event = threading.Event()

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
        pass

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
                        self._handle_list(ws)
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

    def _handle_list(self, ws) -> None:
        """Handle list agents request."""
        with self.lock:
            # Clean up dead agents first
            self._cleanup_dead_agents()

            agents = [agent.to_dict() for agent in self.running_agents.values()]

        response = {"type": "agent_list", "agents": agents}
        self._send_message(ws, response)

    def _handle_attach(self, ws, agent_id: str, current_attached: Optional[str]) -> str:
        """Handle attach to agent request."""
        # First detach from current agent if any
        if current_attached:
            self._handle_detach(ws, current_attached)

        with self.lock:
            agent = self.running_agents.get(agent_id)
            if not agent:
                self._send_error(ws, f"Agent {agent_id} not found")
                return current_attached or ""

            if not agent.is_running():
                self._send_error(ws, f"Agent {agent_id} is not running")
                return current_attached or ""

            # Add to watchers
            agent.watchers.add(ws)

        # Send attach confirmation
        self._send_message(ws, {"type": "attached", "agent_id": agent_id})

        # Send historical events from the agent's log file
        self._send_agent_history(ws, agent_id, agent.log_path)

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
        model = req.get("model", "")
        persona = req.get("persona", "default")
        max_tokens = req.get("max_tokens", 0)

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

        # Spawn the agent process
        try:
            cmd = [
                sys.executable,
                "-m",
                "think.agents",
                "--backend",
                backend,
                "--persona",
                persona,
                "-q",
                prompt,
            ]

            if model:
                cmd.extend(["--model", model])
            if max_tokens:
                cmd.extend(["--max-tokens", str(max_tokens)])

            env = os.environ.copy()
            env["JOURNAL_PATH"] = journal

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
            )

            # Create running agent entry
            agent = RunningAgent(agent_id, process, log_path)

            with self.lock:
                self.running_agents[agent_id] = agent

            # Start monitoring thread
            threading.Thread(
                target=self._monitor_agent, args=(agent,), daemon=True
            ).start()

            self._send_message(ws, {"type": "agent_spawned", "agent_id": agent_id})

        except Exception as e:
            self.logger.exception(f"Failed to spawn agent: {e}")
            self._send_error(ws, f"Failed to spawn agent: {e}")

    def _monitor_agent(self, agent: RunningAgent) -> None:
        """Monitor an agent process and broadcast events to watchers."""
        try:
            # Monitor the log file for new events
            self._tail_agent_log(agent)
        except Exception as e:
            self.logger.exception(f"Error monitoring agent {agent.agent_id}: {e}")
        finally:
            # Clean up when agent finishes
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

    def _tail_agent_log(self, agent: RunningAgent) -> None:
        """Tail the agent's log file and broadcast events to watchers."""
        log_path = agent.log_path

        # Wait for log file to be created
        timeout = 30  # seconds
        start_time = time.time()
        while not log_path.exists() and time.time() - start_time < timeout:
            if not agent.is_running():
                return
            time.sleep(0.1)

        if not log_path.exists():
            self.logger.warning(f"Agent log file not created: {log_path}")
            return

        # Tail the file
        with open(log_path, "r") as f:
            # Go to end of file
            f.seek(0, 2)

            while agent.is_running():
                line = f.readline()
                if line:
                    try:
                        event_data = json.loads(line.strip())
                        self._broadcast_agent_event(agent.agent_id, event_data)
                    except json.JSONDecodeError:
                        continue
                else:
                    time.sleep(0.1)

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

    def _send_agent_history(self, ws, agent_id: str, log_path: Path) -> None:
        """Send historical events from agent log file."""
        if not log_path.exists():
            return

        try:
            with open(log_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            event_data = json.loads(line.strip())
                            message = {
                                "type": "agent_event",
                                "agent_id": agent_id,
                                "event": event_data,
                            }
                            self._send_message(ws, message)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            self.logger.warning(f"Error reading agent history: {e}")

    def _cleanup_watcher(self, ws, agent_id: str) -> None:
        """Remove WebSocket from agent watchers."""
        with self.lock:
            agent = self.running_agents.get(agent_id)
            if agent:
                agent.watchers.discard(ws)

    def _cleanup_dead_agents(self) -> None:
        """Remove agents that are no longer running."""
        dead_agents = []
        for agent_id, agent in self.running_agents.items():
            if not agent.is_running():
                dead_agents.append(agent_id)

        for agent_id in dead_agents:
            del self.running_agents[agent_id]

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
        "--port", type=int, default=5001, help="Port to bind to (default: 5001)"
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

    try:
        app.run(host=args.host, port=args.port, debug=args.verbose)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Shutting down Cortex server")


if __name__ == "__main__":
    main()
