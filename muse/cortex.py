"""Callosum-based agent process manager for Sunstone.

Cortex listens for agent requests via the Callosum message bus and manages
agent process lifecycle:
- Receives requests via Callosum (tract="cortex", event="request")
- Creates <timestamp>_active.jsonl files to track active agents
- Spawns agent processes and captures their stdout events
- Broadcasts all agent events back to Callosum
- Renames to <timestamp>.jsonl when complete

Agent files provide persistence and historical record, while Callosum provides
real-time event distribution to all interested services.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from think.callosum import CallosumConnection


class AgentProcess:
    """Manages a running agent subprocess."""

    def __init__(self, agent_id: str, process: subprocess.Popen, log_path: Path):
        self.agent_id = agent_id
        self.process = process
        self.log_path = log_path
        self.stop_event = threading.Event()
        self.timeout_timer = None  # For timeout support

    def is_running(self) -> bool:
        """Check if the agent process is still running."""
        return self.process.poll() is None and not self.stop_event.is_set()

    def stop(self) -> None:
        """Stop the agent process gracefully."""
        self.stop_event.set()

        # Cancel timeout timer if it exists
        if self.timeout_timer:
            self.timeout_timer.cancel()

        if self.process.poll() is None:
            # First try SIGTERM for graceful shutdown
            self.process.terminate()
            try:
                self.process.wait(timeout=10)  # Give more time for graceful shutdown
            except subprocess.TimeoutExpired:
                logging.getLogger(__name__).warning(
                    f"Agent {self.agent_id} didn't stop gracefully, killing"
                )
                self.process.kill()
                self.process.wait()  # Ensure zombie is reaped


class CortexService:
    """Callosum-based agent process manager."""

    def __init__(self, journal_path: Optional[str] = None):
        self.journal_path = Path(journal_path or os.getenv("JOURNAL_PATH", "."))
        self.agents_dir = self.journal_path / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.running_agents: Dict[str, AgentProcess] = {}
        self.agent_requests: Dict[str, Dict[str, Any]] = {}  # Store agent configs
        self.agent_handoffs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.shutdown_requested = threading.Event()
        self.mcp_thread: Optional[threading.Thread] = None
        self.mcp_server_url: Optional[str] = None

        # Callosum connection for receiving requests and broadcasting events
        self.callosum = CallosumConnection(callback=self._handle_callosum_message)

        self._start_mcp_server()

    def _start_mcp_server(self) -> None:
        """Start the FastMCP HTTP server in a background thread."""

        if self.mcp_thread and self.mcp_thread.is_alive():
            return

        from muse.mcp_tools import mcp

        host = os.getenv("SUNSTONE_MCP_HOST", "127.0.0.1")
        port = int(os.getenv("SUNSTONE_MCP_PORT", "6270"))
        path = os.getenv("SUNSTONE_MCP_PATH", "/mcp") or "/mcp"
        if not path.startswith("/"):
            path = f"/{path}"

        self.mcp_server_url = f"http://{host}:{port}{path}"

        def _run_server() -> None:
            try:
                mcp.run(
                    transport="http",
                    host=host,
                    port=port,
                    path=path,
                    show_banner=False,
                )
            except Exception:
                self.logger.exception("MCP server thread exited unexpectedly")

        self.logger.info("Starting MCP server at %s", self.mcp_server_url)
        self.mcp_thread = threading.Thread(
            target=_run_server,
            name="sunstone-mcp-server",
            daemon=True,
        )
        self.mcp_thread.start()
        self._wait_for_mcp_server(host, port)

    def _wait_for_mcp_server(self, host: str, port: int, timeout: float = 5.0) -> None:
        """Block until MCP server socket accepts connections or timeout."""

        deadline = time.time() + timeout
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.2)
                try:
                    sock.connect((host, port))
                except OSError:
                    time.sleep(0.1)
                    continue
                else:
                    self.logger.debug("MCP server ready at %s", self.mcp_server_url)
                    return

        self.logger.warning(
            "Timed out waiting for MCP server at %s", self.mcp_server_url
        )

    def _create_error_event(
        self,
        agent_id: str,
        error: str,
        trace: Optional[str] = None,
        exit_code: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create standardized error event."""
        event = {
            "event": "error",
            "ts": int(time.time() * 1000),
            "agent_id": agent_id,
            "error": error,
        }
        if trace:
            event["trace"] = trace
        if exit_code is not None:
            event["exit_code"] = exit_code
        return event

    def start(self) -> None:
        """Start listening for agent requests via Callosum."""
        # Check for existing active files - another instance may be running
        active_files = list(self.agents_dir.glob("*_active.jsonl"))
        if active_files:
            self.logger.error(
                f"Found {len(active_files)} active agent(s) - another Cortex instance may be running!"
            )
            self.logger.error(f"Active files: {[f.name for f in active_files]}")
            self.logger.error(
                "Please ensure only one Cortex service is running at a time."
            )
            sys.exit(1)

        # Connect to Callosum to receive requests
        try:
            self.callosum.connect()
            self.logger.info("Connected to Callosum message bus")
        except Exception as e:
            self.logger.error(f"Failed to connect to Callosum: {e}")
            sys.exit(1)

        self.logger.info("Cortex service started, listening for agent requests")

        while True:
            try:
                while not self.stop_event.is_set():
                    time.sleep(1)
                    # Exit when idle during shutdown
                    if self.shutdown_requested.is_set():
                        with self.lock:
                            if len(self.running_agents) == 0:
                                self.logger.info(
                                    "No agents running, exiting gracefully"
                                )
                                return
                break
            except KeyboardInterrupt:
                self.logger.info("Shutdown requested, will exit when idle")
                self.shutdown_requested.set()

    def _handle_callosum_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming Callosum messages (callback)."""
        # Filter for cortex tract and request event
        if message.get("tract") != "cortex" or message.get("event") != "request":
            return

        # Handle the request
        try:
            self._handle_request(message)
        except Exception as e:
            self.logger.exception(f"Error handling request: {e}")

    def _handle_request(self, request: Dict[str, Any]) -> None:
        """Handle a new agent request from Callosum."""
        agent_id = request.get("agent_id")
        if not agent_id:
            self.logger.error("Received request without agent_id")
            return

        try:
            # Skip if this agent is already being processed
            with self.lock:
                if agent_id in self.running_agents:
                    self.logger.debug(
                        f"Agent {agent_id} already running, skipping duplicate"
                    )
                    return

            # Create _active.jsonl file (exclusive creation to prevent race conditions)
            file_path = self.agents_dir / f"{agent_id}_active.jsonl"
            if file_path.exists():
                self.logger.debug(
                    f"Agent {agent_id} already claimed by another process"
                )
                return

            # Write request as first line
            with open(file_path, "x") as f:  # 'x' mode fails if file exists
                f.write(json.dumps(request) + "\n")

            self.logger.info(f"Processing agent request: {agent_id}")

            # Validate request format
            if request.get("event") != "request":
                self._write_error_and_complete(file_path, "Invalid request format")
                self.logger.error(f"Invalid request format: missing 'request' event")
                return

            # Validate prompt early
            prompt = request.get("prompt")
            if not prompt:
                self.logger.error(f"Empty prompt in request {agent_id}")
                self._write_error_and_complete(file_path, "Empty prompt in request")
                return

            # Load persona and merge with request
            from muse.mcp_tools import get_tools
            from think.utils import get_agent

            persona = request.get("persona", "default")
            config = get_agent(persona)

            # Merge request into config (request values override persona defaults)
            # Only override with non-None values from request to preserve persona defaults
            config.update({k: v for k, v in request.items() if v is not None})
            config["agent_id"] = agent_id

            # Capture handoff configuration for post-run processing while
            # leaving it in the merged config for logging transparency.
            handoff_config = config.get("handoff")
            with self.lock:
                if handoff_config:
                    self.agent_handoffs[agent_id] = copy.deepcopy(handoff_config)
                else:
                    self.agent_handoffs.pop(agent_id, None)

            # Inspect previous run continuation to reuse conversation context
            self._apply_conversation_continuation(config)

            # Expand tools if it's a string (tool pack name)
            tools_config = config.get("tools")
            if isinstance(tools_config, str):
                pack_names = [p.strip() for p in tools_config.split(",") if p.strip()]
                if not pack_names:
                    pack_names = ["default"]

                expanded: list[str] = []
                for pack in pack_names:
                    try:
                        for tool in get_tools(pack):
                            if tool not in expanded:
                                expanded.append(tool)
                    except KeyError as e:
                        self.logger.warning(
                            f"Invalid tool pack '{pack}': {e}, using default"
                        )
                        for tool in get_tools("default"):
                            if tool not in expanded:
                                expanded.append(tool)

                config["tools"] = expanded
            elif tools_config is None:
                # If no tools specified, use default pack
                config["tools"] = get_tools("default")

            # Spawn the agent process with the merged config
            self._spawn_agent(agent_id, file_path, config)

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in request file {file_path}: {e}")
            self._write_error_and_complete(file_path, f"Invalid JSON in request: {e}")
        except Exception as e:
            self.logger.exception(f"Error handling active file {file_path}: {e}")
            self._write_error_and_complete(file_path, f"Failed to spawn agent: {e}")

    def _apply_conversation_continuation(self, config: Dict[str, Any]) -> None:
        """Inject conversation_id from previous agent run when requested.

        Cortex users can pass a `continue` field containing the timestamp ID of a
        completed agent run. When present, look up the resulting JSONL file and
        re-use the `conversation_id` from its final event. This keeps OpenAI
        Agents conversations threaded without exposing the concept to callers.
        """

        continue_id = config.pop("continue", None)
        if not continue_id:
            return

        if not isinstance(continue_id, str):
            self.logger.warning(
                "Ignoring non-string continue value on agent request: %s",
                continue_id,
            )
            return

        conversation_id = self._resolve_conversation_id(continue_id)
        if conversation_id:
            config.setdefault("conversation_id", conversation_id)
            self.logger.debug(
                "Resolved conversation_id %s from agent %s",
                conversation_id,
                continue_id,
            )
        else:
            self.logger.warning(
                "Could not find conversation_id to continue from agent %s", continue_id
            )

    def _resolve_conversation_id(self, agent_id: str) -> Optional[str]:
        """Load the final event for a previous agent and return its conversation ID."""

        history_path = self.agents_dir / f"{agent_id}.jsonl"
        if not history_path.exists():
            self.logger.warning(
                "Continuation requested for missing agent history file: %s",
                history_path,
            )
            return None

        last_event: Optional[Dict[str, Any]] = None
        try:
            with open(history_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        last_event = json.loads(line)
                    except json.JSONDecodeError:
                        self.logger.debug(
                            "Skipping malformed JSON line while reading %s",
                            history_path,
                        )
                        continue

        except OSError as exc:
            self.logger.error(
                "Failed to read continuation history from %s: %s", history_path, exc
            )
            return None

        if not last_event:
            self.logger.warning(
                "No events found in agent history file for continuation: %s",
                history_path,
            )
            return None

        conversation_id = last_event.get("conversation_id")
        if conversation_id:
            return str(conversation_id)

        self.logger.info(
            "Previous agent %s did not emit a conversation_id on final event",
            agent_id,
        )
        return None

    def _spawn_agent(
        self,
        agent_id: str,
        file_path: Path,
        config: Dict[str, Any],
    ) -> None:
        """Spawn an agent subprocess and monitor its output using the merged config."""
        try:
            if self.mcp_server_url and not config.get("disable_mcp", False):
                config.setdefault("mcp_server_url", self.mcp_server_url)

            # Store the config for later use (e.g., for save field) - thread safe
            with self.lock:
                self.agent_requests[agent_id] = config

            # Pass the full config through to the agent as NDJSON
            ndjson_input = json.dumps(config)

            # Prepare environment
            env = os.environ.copy()
            env["JOURNAL_PATH"] = str(self.journal_path)

            # Spawn the agent process
            cmd = ["muse-agents"]
            self.logger.info(f"Spawning agent {agent_id}: {cmd}")
            self.logger.debug(f"NDJSON input: {ndjson_input}")

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,
            )

            # Send input and close stdin
            process.stdin.write(ndjson_input + "\n")
            process.stdin.close()

            # Track the running agent
            agent = AgentProcess(agent_id, process, file_path)
            with self.lock:
                self.running_agents[agent_id] = agent

            # Set up timeout (default to 10 minutes if not specified)
            timeout_seconds = config.get(
                "timeout_seconds", 600
            )  # 600 seconds = 10 minutes
            agent.timeout_timer = threading.Timer(
                timeout_seconds,
                lambda: self._timeout_agent(agent_id, agent, timeout_seconds),
            )
            agent.timeout_timer.start()

            # Start monitoring threads
            threading.Thread(
                target=self._monitor_stdout, args=(agent,), daemon=True
            ).start()

            threading.Thread(
                target=self._monitor_stderr, args=(agent,), daemon=True
            ).start()

            self.logger.info(
                f"Agent {agent_id} spawned successfully (PID: {process.pid})"
            )

        except Exception as e:
            self.logger.exception(f"Failed to spawn agent {agent_id}: {e}")
            self._write_error_and_complete(file_path, f"Failed to spawn agent: {e}")

    def _timeout_agent(
        self, agent_id: str, agent: AgentProcess, timeout_seconds: int
    ) -> None:
        """Handle agent timeout."""
        if agent.is_running():
            self.logger.warning(
                f"Agent {agent_id} timed out after {timeout_seconds} seconds"
            )
            error_event = self._create_error_event(
                agent_id, f"Agent timed out after {timeout_seconds} seconds"
            )
            try:
                with open(agent.log_path, "a") as f:
                    f.write(json.dumps(error_event) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to write timeout event: {e}")
            agent.stop()

    def _monitor_stdout(self, agent: AgentProcess) -> None:
        """Monitor agent stdout and append events to the JSONL file."""
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
                    # Parse JSON event
                    event = json.loads(line)

                    # Ensure event has timestamp and agent_id
                    if "ts" not in event:
                        event["ts"] = int(time.time() * 1000)
                    if "agent_id" not in event:
                        event["agent_id"] = agent.agent_id

                    # Append to JSONL file
                    with open(agent.log_path, "a") as f:
                        f.write(json.dumps(event) + "\n")

                    # Broadcast event to Callosum
                    try:
                        self.callosum.emit(
                            "cortex", event.get("event", "unknown"), **event
                        )
                    except Exception as e:
                        self.logger.debug(f"Failed to broadcast event to Callosum: {e}")

                    # Handle finish or error event
                    if event.get("event") in ["finish", "error"]:
                        # Check for save and handoff (only on finish)
                        if event.get("event") == "finish":
                            result = event.get("result", "")

                            # Get original request (thread-safe access)
                            with self.lock:
                                original_request = self.agent_requests.get(
                                    agent.agent_id
                                )

                            # Log token usage if available
                            usage_data = event.get("usage")
                            if usage_data and original_request:
                                try:
                                    from think.models import log_token_usage

                                    model = original_request.get("model", "unknown")
                                    persona = original_request.get("persona", "unknown")
                                    context = f"agent.{persona}.{agent.agent_id}"

                                    log_token_usage(
                                        model=model,
                                        usage=usage_data,
                                        context=context,
                                    )
                                except Exception as e:
                                    self.logger.warning(
                                        f"Failed to log token usage for agent {agent.agent_id}: {e}"
                                    )

                            # Save result if requested
                            if original_request and original_request.get("save"):
                                self._save_agent_result(
                                    agent.agent_id,
                                    result,
                                    original_request["save"],
                                    original_request.get(
                                        "day"
                                    ),  # Pass optional day parameter
                                )

                            # Handle handoff (prefer stored config captured at startup)
                            handoff_config = None
                            with self.lock:
                                if agent.agent_id in self.agent_handoffs:
                                    handoff_config = copy.deepcopy(
                                        self.agent_handoffs.pop(agent.agent_id)
                                    )

                            if handoff_config:
                                self._spawn_handoff(
                                    agent.agent_id, result, handoff_config
                                )
                        # Break to trigger cleanup
                        break

                except json.JSONDecodeError:
                    # Non-JSON output becomes info event
                    info_event = {
                        "event": "info",
                        "ts": int(time.time() * 1000),
                        "message": line,
                        "agent_id": agent.agent_id,
                    }
                    with open(agent.log_path, "a") as f:
                        f.write(json.dumps(info_event) + "\n")

        except Exception as e:
            self.logger.error(
                f"Error monitoring stdout for agent {agent.agent_id}: {e}"
            )
        finally:
            # Wait for process to fully exit (reaps zombie)
            exit_code = agent.process.wait()
            self.logger.info(f"Agent {agent.agent_id} exited with code {exit_code}")

            # Check if finish event was emitted
            has_finish = self._has_finish_event(agent.log_path)

            if not has_finish:
                # Write error event if no finish using standardized format
                error_event = self._create_error_event(
                    agent.agent_id,
                    f"Agent exited with code {exit_code} without finish event",
                    exit_code=exit_code,
                )
                with open(agent.log_path, "a") as f:
                    f.write(json.dumps(error_event) + "\n")

            # Complete the file (rename from _active.jsonl to .jsonl)
            self._complete_agent_file(agent.agent_id, agent.log_path)

            # Remove from running agents and clean up stored request (thread-safe)
            with self.lock:
                if agent.agent_id in self.running_agents:
                    del self.running_agents[agent.agent_id]
                # Clean up stored request
                if agent.agent_id in self.agent_requests:
                    del self.agent_requests[agent.agent_id]
                # Ensure any pending handoff config is discarded
                self.agent_handoffs.pop(agent.agent_id, None)

    def _monitor_stderr(self, agent: AgentProcess) -> None:
        """Monitor agent stderr for errors."""
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
                    self.logger.debug(f"Agent {agent.agent_id} stderr: {stripped}")

        except Exception as e:
            self.logger.error(
                f"Error monitoring stderr for agent {agent.agent_id}: {e}"
            )
        finally:
            # If process failed with stderr output, write error event
            if stderr_lines:
                exit_code = agent.process.poll()
                if exit_code is not None and exit_code != 0:
                    error_event = self._create_error_event(
                        agent.agent_id,
                        "Process failed with stderr output",
                        trace="\n".join(stderr_lines),
                        exit_code=exit_code,
                    )
                    try:
                        with open(agent.log_path, "a") as f:
                            f.write(json.dumps(error_event) + "\n")
                    except Exception as e:
                        self.logger.warning(f"Failed to write stderr event: {e}")

    def _has_finish_event(self, file_path: Path) -> bool:
        """Check if the JSONL file contains a finish or error event."""
        try:
            with open(file_path, "r") as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if event.get("event") in ["finish", "error"]:
                            return True
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return False

    def _complete_agent_file(self, agent_id: str, file_path: Path) -> None:
        """Complete an agent by renaming the file from _active.jsonl to .jsonl."""
        try:
            completed_path = file_path.parent / f"{agent_id}.jsonl"
            file_path.rename(completed_path)
            self.logger.info(f"Completed agent {agent_id}: {completed_path}")
        except Exception as e:
            self.logger.error(f"Failed to complete agent file {agent_id}: {e}")

    def _write_error_and_complete(self, file_path: Path, error_message: str) -> None:
        """Write an error event to the file and mark it as complete."""
        try:
            agent_id = file_path.stem.replace("_active", "")
            error_event = self._create_error_event(agent_id, error_message)
            with open(file_path, "a") as f:
                f.write(json.dumps(error_event) + "\n")

            # Complete the file
            self._complete_agent_file(agent_id, file_path)
        except Exception as e:
            self.logger.error(f"Failed to write error and complete: {e}")

    def _save_agent_result(
        self, agent_id: str, result: str, save_filename: str, day: Optional[str] = None
    ) -> None:
        """Save agent result to a file in the specified or current day directory."""
        try:
            from think.utils import day_path

            # day_path now handles None for today, creates dir, and returns Path
            day_dir = day_path(day)

            # Write result to save file
            save_path = day_dir / save_filename
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(result)

            self.logger.info(f"Saved agent {agent_id} result to {save_path}")

        except Exception as e:
            self.logger.error(f"Failed to save agent {agent_id} result: {e}")
            # Don't raise - continue with normal flow even if save fails

    def _spawn_handoff(
        self, parent_id: str, result: str, handoff: Dict[str, Any]
    ) -> None:
        """Spawn a handoff agent from a completed agent's result."""
        try:
            from muse.cortex_client import cortex_request

            if not handoff:
                self.logger.debug(
                    "No handoff configuration provided for agent %s", parent_id
                )
                return

            # Operate on a copy so callers keep their original config untouched.
            handoff_config = copy.deepcopy(handoff)

            # Determine prompt/backends/persona before pruning extra keys.
            prompt = handoff_config.pop("prompt", None) or result
            persona = handoff_config.pop("persona", None) or "default"
            backend = handoff_config.pop("backend", None)
            if backend is None:
                with self.lock:
                    backend = self.agent_requests.get(parent_id, {}).get("backend")
            if backend is None:
                backend = "openai"

            # Ensure we do not propagate parent handoff metadata.
            handoff_config.pop("handoff", None)
            handoff_config.pop("handoff_from", None)

            # Only pass through additional overrides if any remain.
            extra_config = handoff_config or None

            # Use cortex_request to create the handoff agent
            active_path = cortex_request(
                prompt=prompt,
                persona=persona,
                backend=backend,
                handoff_from=parent_id,
                config=extra_config,
            )

            self.logger.info(f"Spawned handoff agent {active_path} from {parent_id}")

        except Exception as e:
            self.logger.error(f"Failed to spawn handoff agent: {e}")

    def stop(self) -> None:
        """Stop the Cortex service."""
        self.stop_event.set()

        # Close Callosum connection
        if self.callosum:
            self.callosum.close()

        # Stop all running agents
        with self.lock:
            for agent in self.running_agents.values():
                agent.stop()

    def get_status(self) -> Dict[str, Any]:
        """Get service status information."""
        with self.lock:
            return {
                "running_agents": len(self.running_agents),
                "agent_ids": list(self.running_agents.keys()),
            }


def main() -> None:
    """CLI entry point for the Cortex service."""
    import argparse

    from think.utils import setup_cli

    parser = argparse.ArgumentParser(description="Sunstone Cortex Agent Manager")
    args = setup_cli(parser)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Start the service
    cortex = CortexService()

    try:
        cortex.start()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Shutting down Cortex service")
        cortex.stop()


if __name__ == "__main__":
    main()
