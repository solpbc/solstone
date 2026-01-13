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
from think.utils import get_journal


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
        self.agent_requests: Dict[str, Dict[str, Any]] = {}  # Store agent configs
        self.agent_handoffs: Dict[str, Dict[str, Any]] = {}
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

        from muse.mcp import mcp

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
                self.logger.error("Invalid request format: missing 'request' event")
                return

            # Validate prompt early
            prompt = request.get("prompt")
            if not prompt:
                self.logger.error(f"Empty prompt in request {agent_id}")
                self._write_error_and_complete(file_path, "Empty prompt in request")
                return

            # Validate and link continue_from if specified
            continue_from = request.get("continue_from")
            if continue_from:
                from muse.cortex_client import get_agent_status

                status = get_agent_status(continue_from)
                if status != "completed":
                    error_msg = f"Cannot continue from agent {continue_from}: " + (
                        "agent is still running"
                        if status == "running"
                        else "agent not found"
                    )
                    self.logger.error(error_msg)
                    self._write_error_and_complete(file_path, error_msg)
                    return

                # Append continue event to the source agent's file
                continue_event = {
                    "event": "continue",
                    "ts": int(time.time() * 1000),
                    "agent_id": continue_from,
                    "to": agent_id,
                }
                source_file = self.agents_dir / f"{continue_from}.jsonl"
                with open(source_file, "a") as f:
                    f.write(json.dumps(continue_event) + "\n")
                self.logger.info(f"Linked continuation: {continue_from} -> {agent_id}")

            # Load persona and merge with request
            from muse.mcp import get_tools
            from think.utils import get_agent

            persona = request.get("persona", "default")
            facet = request.get("facet")
            config = get_agent(persona, facet=facet)

            # Merge request into config (request values override persona defaults)
            # Only override with non-None values from request to preserve persona defaults
            config.update({k: v for k, v in request.items() if v is not None})
            config["agent_id"] = agent_id

            # Resolve provider and model from context
            # Context format: agent.{app}.{name} where app="system" for system agents
            from muse.models import resolve_model_for_provider, resolve_provider

            if ":" in persona:
                app, name = persona.split(":", 1)
            else:
                app, name = "system", persona
            agent_context = f"agent.{app}.{name}"

            # Check for claude: true flag (special case for Claude Code SDK)
            if config.get("claude"):
                config["provider"] = "claude"
                # Claude SDK doesn't need model - it uses its own
            else:
                # Resolve default provider and model from context
                default_provider, model = resolve_provider(agent_context)

                # Provider can be overridden by request or persona config
                # Model is always resolved from context tier + final provider
                provider = config.get("provider") or default_provider

                # If provider was overridden, re-resolve model for that provider
                if provider != default_provider:
                    model = resolve_model_for_provider(agent_context, provider)

                config["provider"] = provider
                config["model"] = model

            # Capture handoff configuration for post-run processing while
            # leaving it in the merged config for logging transparency.
            handoff_config = config.get("handoff")
            with self.lock:
                if handoff_config:
                    self.agent_handoffs[agent_id] = copy.deepcopy(handoff_config)
                else:
                    self.agent_handoffs.pop(agent_id, None)

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

            # Prepare environment - apply config overrides first, then force JOURNAL_PATH
            env = os.environ.copy()
            env_overrides = config.get("env")
            if env_overrides and isinstance(env_overrides, dict):
                env.update({k: str(v) for k, v in env_overrides.items()})
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
                        event_copy = event.copy()
                        event_type = event_copy.pop("event", "unknown")
                        self.callosum.emit("cortex", event_type, **event_copy)
                    except Exception as e:
                        self.logger.info(f"Failed to broadcast event to Callosum: {e}")

                    # Capture model from start event (needed for token usage logging)
                    if event.get("event") == "start":
                        model = event.get("model")
                        if model:
                            with self.lock:
                                if agent.agent_id in self.agent_requests:
                                    self.agent_requests[agent.agent_id]["model"] = model

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
                                    from muse.models import log_token_usage

                                    model = original_request.get("model", "unknown")
                                    persona = original_request.get("persona", "unknown")

                                    # Build context in same format as model resolution:
                                    # agent.{app}.{name} where app="system" for system agents
                                    if ":" in persona:
                                        app, name = persona.split(":", 1)
                                    else:
                                        app, name = "system", persona
                                    context = f"agent.{app}.{name}"

                                    # Extract segment from config env if set
                                    config = original_request.get("config", {})
                                    env_config = config.get("env", {})
                                    segment = env_config.get("SEGMENT_KEY")

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

            # Determine prompt/provider/persona before pruning extra keys.
            prompt = handoff_config.pop("prompt", None) or result
            persona = handoff_config.pop("persona", None) or "default"

            # Provider can be explicitly set in handoff config, otherwise let
            # the handoff persona resolve its own provider from context
            provider = handoff_config.pop("provider", None)

            # Ensure we do not propagate parent handoff metadata or claude flag.
            # Each persona must declare claude: true in its own config.
            handoff_config.pop("handoff", None)
            handoff_config.pop("handoff_from", None)
            handoff_config.pop("claude", None)
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
                persona=persona,
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
                                "persona": config.get("persona", "unknown"),
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


def format_agent(
    entries: list[dict],
    context: dict | None = None,
) -> tuple[list[dict], dict]:
    """Format agent JSONL entries to markdown chunks.

    This is the formatter function used by the formatters registry.

    Args:
        entries: Raw JSONL entries (agent events)
        context: Optional context dict

    Returns:
        Tuple of (chunks, meta) where:
            - chunks: List of {"timestamp": int, "markdown": str} dicts
            - meta: Dict with optional "header" and "error" keys
    """
    from datetime import datetime
    from typing import Any

    _ = context  # Reserved for future context support
    meta: dict[str, Any] = {}
    chunks: list[dict[str, Any]] = []
    skipped_count = 0

    # Track tool_start events by call_id for pairing
    pending_tools: dict[str, dict] = {}

    # Extract request/start events for header
    request_event: dict | None = None
    start_event: dict | None = None
    agent_id: str | None = None

    def ts_to_time(ts: int) -> str:
        """Convert millisecond timestamp to HH:MM:SS."""
        return datetime.fromtimestamp(ts / 1000).strftime("%H:%M:%S")

    def truncate_result(text: str, max_len: int = 500) -> str:
        """Truncate text and note how many chars were truncated."""
        if len(text) <= max_len:
            return text
        truncated = len(text) - max_len
        return f"{text[:max_len]}... ({truncated} chars truncated)"

    for entry in entries:
        event_type = entry.get("event")
        if not event_type:
            skipped_count += 1
            continue

        ts = entry.get("ts", 0)

        if event_type == "request":
            request_event = entry
            agent_id = entry.get("agent_id") or str(ts)
            continue

        if event_type == "start":
            start_event = entry
            if not agent_id:
                agent_id = entry.get("agent_id") or str(ts)
            continue

        if event_type == "agent_updated":
            agent_name = entry.get("agent")
            if agent_name:
                chunks.append(
                    {
                        "timestamp": ts,
                        "markdown": f"*Switched to agent: {agent_name}*\n",
                    }
                )
            continue

        if event_type == "thinking":
            content = entry.get("content", "")
            if content:
                lines = [
                    f"### {ts_to_time(ts)} - Thinking\n",
                    "",
                ]
                # Format as blockquote
                for line in content.split("\n"):
                    lines.append(f"> {line}")
                lines.append("")
                chunks.append(
                    {
                        "timestamp": ts,
                        "markdown": "\n".join(lines),
                    }
                )
            continue

        if event_type == "tool_start":
            call_id = entry.get("call_id")
            if call_id:
                pending_tools[call_id] = entry
            else:
                # No call_id - emit as standalone
                tool_name = entry.get("tool", "unknown")
                args = entry.get("args", {})
                lines = [
                    f"### {ts_to_time(ts)} - Tool: {tool_name}\n",
                    "",
                    "**Args:**",
                    "```json",
                    json.dumps(args, indent=2),
                    "```",
                    "",
                    "*Tool call in progress...*",
                    "",
                ]
                chunks.append(
                    {
                        "timestamp": ts,
                        "markdown": "\n".join(lines),
                    }
                )
            continue

        if event_type == "tool_end":
            call_id = entry.get("call_id")
            result_str = entry.get("result", "")

            # Try to find matching tool_start
            start_entry = pending_tools.pop(call_id, None) if call_id else None

            if start_entry:
                # Paired tool call
                tool_name = start_entry.get("tool", "unknown")
                args = start_entry.get("args", {})
                start_ts = start_entry.get("ts", ts)
            else:
                # Unpaired tool_end
                tool_name = entry.get("tool", "unknown")
                args = entry.get("args") or {}
                start_ts = ts

            lines = [
                f"### {ts_to_time(start_ts)} - Tool: {tool_name}\n",
                "",
            ]

            if args:
                lines.extend(
                    [
                        "**Args:**",
                        "```json",
                        json.dumps(args, indent=2),
                        "```",
                        "",
                    ]
                )

            if result_str:
                truncated = truncate_result(result_str)
                lines.extend(
                    [
                        "**Result:**",
                        "```",
                        truncated,
                        "```",
                        "",
                    ]
                )

            chunks.append(
                {
                    "timestamp": start_ts,
                    "markdown": "\n".join(lines),
                }
            )
            continue

        if event_type == "error":
            error_msg = entry.get("error", "Unknown error")
            trace = entry.get("trace", "")
            lines = [
                f"### {ts_to_time(ts)} - Error\n",
                "",
                f"> **Error:** {error_msg}",
                "",
            ]
            if trace:
                lines.extend(
                    [
                        "**Trace:**",
                        "```",
                        trace[:1000] if len(trace) > 1000 else trace,
                        "```",
                        "",
                    ]
                )
            chunks.append(
                {
                    "timestamp": ts,
                    "markdown": "\n".join(lines),
                }
            )
            continue

        if event_type == "info":
            message = entry.get("message", "")
            if message:
                chunks.append(
                    {
                        "timestamp": ts,
                        "markdown": f"*{ts_to_time(ts)} - Info:* {message}\n",
                    }
                )
            continue

        if event_type == "finish":
            result = entry.get("result", "")
            lines = [
                f"### {ts_to_time(ts)} - Result\n",
                "",
                result,
                "",
            ]
            chunks.append(
                {
                    "timestamp": ts,
                    "markdown": "\n".join(lines),
                }
            )
            continue

        if event_type == "continue":
            to_agent = entry.get("to", "unknown")
            chunks.append(
                {
                    "timestamp": ts,
                    "markdown": f"*Continued in agent: {to_agent}*\n",
                }
            )
            continue

        # Unknown event type - skip
        skipped_count += 1

    # Handle any unpaired tool_start events (agent crashed mid-tool)
    for call_id, start_entry in pending_tools.items():
        tool_name = start_entry.get("tool", "unknown")
        args = start_entry.get("args", {})
        start_ts = start_entry.get("ts", 0)
        lines = [
            f"### {ts_to_time(start_ts)} - Tool: {tool_name}\n",
            "",
        ]
        if args:
            lines.extend(
                [
                    "**Args:**",
                    "```json",
                    json.dumps(args, indent=2),
                    "```",
                    "",
                ]
            )
        lines.append("*Tool call did not complete*\n")
        chunks.append(
            {
                "timestamp": start_ts,
                "markdown": "\n".join(lines),
            }
        )

    # Build header from request/start events
    header_lines = []
    if agent_id:
        header_lines.append(f"# Agent Run: {agent_id}\n")
    else:
        header_lines.append("# Agent Run\n")

    if request_event:
        prompt = request_event.get("prompt", "")
        if prompt:
            # Truncate long prompts in header
            if len(prompt) > 200:
                prompt = prompt[:200] + "..."
            header_lines.append(f"**Prompt:** {prompt}\n")

        persona = request_event.get("persona", "default")
        provider = request_event.get("provider", "")
        model = start_event.get("model", "") if start_event else ""

        meta_parts = [f"**Persona:** {persona}"]
        if provider:
            meta_parts.append(f"**Provider:** {provider}")
        if model:
            meta_parts.append(f"**Model:** {model}")
        header_lines.append(" | ".join(meta_parts) + "\n")

        # Continuation/handoff info
        continue_from = request_event.get("continue_from")
        handoff_from = request_event.get("handoff_from")
        if continue_from:
            header_lines.append(f"*Continued from:* {continue_from}\n")
        if handoff_from:
            header_lines.append(f"*Handoff from:* {handoff_from}\n")

    if start_event:
        start_ts = start_event.get("ts", 0)
        if start_ts:
            header_lines.append(f"**Started:** {ts_to_time(start_ts)}\n")

    meta["header"] = "\n".join(header_lines)

    # Report skipped entries
    if skipped_count > 0:
        meta["error"] = f"Skipped {skipped_count} entries missing 'event' field"

    # Indexer metadata - agents aren't indexed but include for consistency
    meta["indexer"] = {"topic": "agent"}

    return chunks, meta


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
