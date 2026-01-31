# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Callosum-based agent process manager for solstone.

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
from think.utils import get_journal, now_ms


class AgentProcess:
    """Manages a running agent subprocess."""

    def __init__(self, agent_id: str, process: subprocess.Popen, log_path: Path):
        self.agent_id = agent_id
        self.process = process
        self.log_path = log_path
        self.stop_event = threading.Event()
        self.timeout_timer = None  # For timeout support
        self.start_time = time.time()  # Track when agent started

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
        self.journal_path = Path(journal_path or get_journal())
        self.agents_dir = self.journal_path / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.running_agents: Dict[str, AgentProcess] = {}
        self.agent_requests: Dict[str, Dict[str, Any]] = {}  # Store agent requests
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.shutdown_requested = threading.Event()
        self.mcp_thread: Optional[threading.Thread] = None
        self.mcp_server_url: Optional[str] = None

        # Callosum connection for receiving requests and broadcasting events
        self.callosum = CallosumConnection()

        self._start_mcp_server()

    def _start_mcp_server(self) -> None:
        """Start the FastMCP HTTP server in a background thread."""

        if self.mcp_thread and self.mcp_thread.is_alive():
            return

        from think.mcp import mcp

        host = os.getenv("SOLSTONE_MCP_HOST", "127.0.0.1")
        port = int(os.getenv("SOLSTONE_MCP_PORT", "6270"))
        path = os.getenv("SOLSTONE_MCP_PATH", "/mcp") or "/mcp"
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
            name="solstone-mcp-server",
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
            "ts": now_ms(),
            "agent_id": agent_id,
            "error": error,
        }
        if trace:
            event["trace"] = trace
        if exit_code is not None:
            event["exit_code"] = exit_code
        return event

    def _recover_orphaned_agents(self, active_files: list) -> None:
        """Recover orphaned active agent files from a previous crash.

        Appends an error event to each file and renames to completed.
        """
        for file_path in active_files:
            agent_id = file_path.stem.replace("_active", "")
            try:
                error_event = self._create_error_event(
                    agent_id, "Recovered: Cortex restarted while agent was running"
                )
                with open(file_path, "a") as f:
                    f.write(json.dumps(error_event) + "\n")

                completed_path = file_path.parent / f"{agent_id}.jsonl"
                file_path.rename(completed_path)
                self.logger.warning(f"Recovered orphaned agent: {agent_id}")
            except Exception as e:
                self.logger.error(f"Failed to recover agent {agent_id}: {e}")

    def start(self) -> None:
        """Start listening for agent requests via Callosum."""
        # Recover any orphaned active files from previous crash
        active_files = list(self.agents_dir.glob("*_active.jsonl"))
        if active_files:
            self.logger.warning(
                f"Found {len(active_files)} orphaned agent(s), recovering..."
            )
            self._recover_orphaned_agents(active_files)

        # Connect to Callosum to receive requests
        try:
            self.callosum.start(callback=self._handle_callosum_message)
            self.logger.info("Connected to Callosum message bus")
        except Exception as e:
            self.logger.error(f"Failed to connect to Callosum: {e}")
            sys.exit(1)

        # Start status emission thread
        threading.Thread(
            target=self._emit_periodic_status,
            name="cortex-status",
            daemon=True,
        ).start()

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
        """Handle a new agent request from Callosum.

        Cortex is a minimal process manager - it only handles:
        - File lifecycle (_active.jsonl -> .jsonl)
        - Process spawning and monitoring
        - Event relay to Callosum

        All config loading, validation, and hydration is done by agents.py.
        """
        agent_id = request.get("agent_id")
        if not agent_id:
            self.logger.error("Received request without agent_id")
            return

        # Skip if this agent is already being processed
        with self.lock:
            if agent_id in self.running_agents:
                self.logger.debug(f"Agent {agent_id} already running, skipping")
                return

        # Create _active.jsonl file (exclusive creation to prevent race conditions)
        file_path = self.agents_dir / f"{agent_id}_active.jsonl"
        if file_path.exists():
            self.logger.debug(f"Agent {agent_id} already claimed by another process")
            return

        try:
            with open(file_path, "x") as f:  # 'x' mode fails if file exists
                f.write(json.dumps(request) + "\n")
        except FileExistsError:
            return

        self.logger.info(f"Processing agent request: {agent_id}")

        # Store request for later use (handoffs, output writing)
        with self.lock:
            self.agent_requests[agent_id] = request

        # Inject MCP server URL
        if self.mcp_server_url:
            request["mcp_server_url"] = self.mcp_server_url

        # Spawn agent process - it handles all validation/hydration
        try:
            self._spawn_subprocess(
                agent_id, file_path, request, ["sol", "agents"], "agent"
            )
        except Exception as e:
            self.logger.exception(f"Failed to spawn agent {agent_id}: {e}")
            self._write_error_and_complete(file_path, f"Failed to spawn agent: {e}")

    def _spawn_subprocess(
        self,
        agent_id: str,
        file_path: Path,
        config: Dict[str, Any],
        cmd: list[str],
        process_type: str,
    ) -> None:
        """Spawn a subprocess and monitor its output.

        Args:
            agent_id: Unique identifier for this process
            file_path: Path to the JSONL log file
            config: Configuration dict to pass via NDJSON stdin
            cmd: Command to run (e.g., ["sol", "agents"])
            process_type: Label for logging ("agent")
        """
        try:
            # Store the config for later use - thread safe
            with self.lock:
                self.agent_requests[agent_id] = config

            # Pass the full config through as NDJSON
            ndjson_input = json.dumps(config)

            # Prepare environment - apply config overrides first, then force JOURNAL_PATH
            env = os.environ.copy()
            env_overrides = config.get("env")
            if env_overrides and isinstance(env_overrides, dict):
                env.update({k: str(v) for k, v in env_overrides.items()})
            env["JOURNAL_PATH"] = str(self.journal_path)

            # Spawn the subprocess
            self.logger.info(f"Spawning {process_type} {agent_id}: {cmd}")
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

            # Track the running process
            agent = AgentProcess(agent_id, process, file_path)
            with self.lock:
                self.running_agents[agent_id] = agent

            # Set up timeout (default to 10 minutes if not specified)
            timeout_seconds = config.get("timeout_seconds", 600)
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
                f"{process_type.capitalize()} {agent_id} spawned successfully "
                f"(PID: {process.pid})"
            )

        except Exception as e:
            self.logger.exception(f"Failed to spawn {process_type} {agent_id}: {e}")
            self._write_error_and_complete(
                file_path, f"Failed to spawn {process_type}: {e}"
            )

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
            with agent.process.stdout:
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
                            event["ts"] = now_ms()
                        if "agent_id" not in event:
                            event["agent_id"] = agent.agent_id

                        # Append to JSONL file
                        with open(agent.log_path, "a") as f:
                            f.write(json.dumps(event) + "\n")

                        # Broadcast event to Callosum
                        try:
                            event_copy = event.copy()
                            event_type = event_copy.pop("event", "unknown")
                            self.callosum.emit("cortex", event_type, **event_copy)
                        except Exception as e:
                            self.logger.info(
                                f"Failed to broadcast event to Callosum: {e}"
                            )

                        # Handle start event
                        if event.get("event") == "start":
                            # Capture model for token usage logging
                            model = event.get("model")
                            if model:
                                with self.lock:
                                    if agent.agent_id in self.agent_requests:
                                        self.agent_requests[agent.agent_id][
                                            "model"
                                        ] = model

                            # Write continue event to source file if continuing
                            continue_from = event.get("continue_from")
                            if continue_from:
                                continue_event = {
                                    "event": "continue",
                                    "ts": now_ms(),
                                    "agent_id": continue_from,
                                    "to": agent.agent_id,
                                }
                                source_file = self.agents_dir / f"{continue_from}.jsonl"
                                if source_file.exists():
                                    with open(source_file, "a") as f:
                                        f.write(json.dumps(continue_event) + "\n")
                                    self.logger.info(
                                        f"Linked continuation: {continue_from} -> {agent.agent_id}"
                                    )

                        # Handle finish or error event
                        if event.get("event") in ["finish", "error"]:
                            # Check for output and handoff (only on finish)
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
                                        from think.utils import key_to_context

                                        model = original_request.get("model", "unknown")
                                        name = original_request.get("name", "unknown")
                                        context = key_to_context(name)

                                        # Extract segment from env if set (flat merge puts env at top level)
                                        env_config = original_request.get("env", {})
                                        segment = (
                                            env_config.get("SEGMENT_KEY")
                                            if env_config
                                            else None
                                        )

                                        log_token_usage(
                                            model=model,
                                            usage=usage_data,
                                            context=context,
                                            segment=segment,
                                        )
                                    except Exception as e:
                                        self.logger.warning(
                                            f"Failed to log token usage for agent {agent.agent_id}: {e}"
                                        )

                                # Write output if requested
                                if original_request and original_request.get("output"):
                                    self._write_output(
                                        agent.agent_id,
                                        result,
                                        original_request,
                                    )

                                # Handle handoff from finish event
                                handoff_config = event.get("handoff")
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
                            "ts": now_ms(),
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

    def _monitor_stderr(self, agent: AgentProcess) -> None:
        """Monitor agent stderr for errors."""
        if not agent.process.stderr:
            return

        stderr_lines = []
        try:
            with agent.process.stderr:
                for line in agent.process.stderr:
                    if not line:
                        continue
                    stripped = line.strip()
                    if stripped:
                        stderr_lines.append(stripped)
                        # Pass through to cortex stderr with agent prefix for traceability
                        print(
                            f"[agent:{agent.agent_id}:stderr] {stripped}",
                            file=sys.stderr,
                            flush=True,
                        )

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

    def _write_output(self, agent_id: str, result: str, config: Dict[str, Any]) -> None:
        """Write agent output to the appropriate location.

        Output path is either:
        - Explicit: config["output_path"] (for multi-segment and custom paths)
        - Derived: from name + output format + schedule + facet:
          - Daily agents: YYYYMMDD/agents/{name}.{ext}
          - Segment agents: YYYYMMDD/{segment}/{name}.{ext}
          - Multi-facet: {name}_{facet}.{ext} instead of {name}.{ext}
        """
        try:
            from think.utils import day_path, get_output_path

            # Check for explicit output_path override first
            if config.get("output_path"):
                output_path = Path(config["output_path"])
            else:
                output_format = config.get("output", "md")
                name = config.get("name", "default")
                segment = config.get("segment")  # Set for segment agents
                facet = config.get("facet")  # Set for multi-facet agents
                day = config.get("day")

                # Get day directory
                day_dir = day_path(day)

                # Derive output path using shared utility
                output_path = get_output_path(
                    day_dir,
                    name,
                    segment=segment,
                    output_format=output_format,
                    facet=facet,
                )

            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)

            self.logger.info(f"Wrote agent {agent_id} output to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to write agent {agent_id} output: {e}")
            # Don't raise - continue with normal flow even if write fails

    def _spawn_handoff(
        self, parent_id: str, result: str, handoff: Dict[str, Any]
    ) -> None:
        """Spawn a handoff agent from a completed agent's result."""
        try:
            from think.cortex_client import cortex_request

            if not handoff:
                self.logger.debug(
                    "No handoff configuration provided for agent %s", parent_id
                )
                return

            # Operate on a copy so callers keep their original config untouched.
            handoff_config = copy.deepcopy(handoff)

            # Determine prompt/provider/name before pruning extra keys.
            prompt = handoff_config.pop("prompt", None) or result
            name = handoff_config.pop("name", None) or "default"

            # Provider can be explicitly set in handoff config, otherwise let
            # the handoff agent resolve its own provider from context
            provider = handoff_config.pop("provider", None)

            # Ensure we do not propagate parent handoff metadata.
            handoff_config.pop("handoff", None)
            handoff_config.pop("handoff_from", None)
            handoff_config.pop("model", None)

            # Inherit env from parent if not explicitly set in handoff config
            if "env" not in handoff_config:
                with self.lock:
                    parent_env = self.agent_requests.get(parent_id, {}).get("env")
                if parent_env:
                    handoff_config["env"] = parent_env

            # Only pass through additional overrides if any remain.
            extra_config = handoff_config or None

            # Use cortex_request to create the handoff agent
            agent_id = cortex_request(
                prompt=prompt,
                name=name,
                provider=provider,
                handoff_from=parent_id,
                config=extra_config,
            )

            self.logger.info(f"Spawned handoff agent {agent_id} from {parent_id}")

        except Exception as e:
            self.logger.error(f"Failed to spawn handoff agent: {e}")

    def stop(self) -> None:
        """Stop the Cortex service."""
        self.stop_event.set()

        # Close Callosum connection
        if self.callosum:
            self.callosum.stop()

        # Stop all running agents
        with self.lock:
            for agent in self.running_agents.values():
                agent.stop()

    def _emit_periodic_status(self) -> None:
        """Emit status events every 5 seconds (runs in background thread)."""
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    agents = []
                    for agent_id, agent_proc in self.running_agents.items():
                        config = self.agent_requests.get(agent_id, {})
                        agents.append(
                            {
                                "agent_id": agent_id,
                                "name": config.get("name", "unknown"),
                                "provider": config.get("provider", "unknown"),
                                "elapsed_seconds": int(
                                    time.time() - agent_proc.start_time
                                ),
                            }
                        )

                # Only emit status when there are active agents
                if agents:
                    self.callosum.emit(
                        "cortex",
                        "status",
                        running_agents=len(agents),
                        agents=agents,
                    )
            except Exception as e:
                self.logger.debug(f"Status emission failed: {e}")

            time.sleep(5)

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

    parser = argparse.ArgumentParser(description="solstone Cortex Agent Manager")
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
