"""File-based agent process manager for Sunstone.

Cortex monitors the journal's agents/ directory for new request files and manages
agent process lifecycle through file state transitions:
- <timestamp>_pending.jsonl: Request awaiting processing
- <timestamp>_active.jsonl: Agent currently executing
- <timestamp>.jsonl: Agent completed

This service uses watchdog to detect new active files and spawns agent processes,
capturing their stdout events and appending them to the JSONL files.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time

# Removed datetime - using time.time() instead
from pathlib import Path
from typing import Any, Dict, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class AgentProcess:
    """Manages a running agent subprocess."""

    def __init__(self, agent_id: str, process: subprocess.Popen, log_path: Path):
        self.agent_id = agent_id
        self.process = process
        self.log_path = log_path
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


class AgentFileHandler(FileSystemEventHandler):
    """Handles file system events for agent request files."""

    def __init__(self, cortex_service: "CortexService"):
        self.cortex_service = cortex_service
        self.logger = logging.getLogger(__name__)

    def on_moved(self, event):
        """Handle file move events (pending -> active)."""
        if event.is_directory:
            return

        # Check if file was renamed to _active.jsonl
        dest_path = Path(event.dest_path)
        if dest_path.name.endswith("_active.jsonl"):
            agent_id = dest_path.stem.replace("_active", "")
            self.logger.info(f"Detected new active file: {agent_id}")
            self.cortex_service._handle_active_file(agent_id, dest_path)

    def on_created(self, event):
        """Handle file creation events (for direct _active.jsonl creation)."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.name.endswith("_active.jsonl"):
            agent_id = file_path.stem.replace("_active", "")
            self.logger.info(f"Detected created active file: {agent_id}")
            # Small delay to ensure file is fully written
            time.sleep(0.1)
            self.cortex_service._handle_active_file(agent_id, file_path)


class CortexService:
    """File-based agent process manager."""

    def __init__(self, journal_path: Optional[str] = None):
        self.journal_path = Path(journal_path or os.getenv("JOURNAL_PATH", "."))
        self.agents_dir = self.journal_path / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.running_agents: Dict[str, AgentProcess] = {}
        self.agent_requests: Dict[str, Dict[str, Any]] = {}  # Store agent configs
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.observer = None

    def start(self) -> None:
        """Start monitoring for new agent requests."""
        # Process any existing active files on startup
        self._process_existing_active_files()

        # Set up file system observer
        event_handler = AgentFileHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.agents_dir), recursive=False)
        self.observer.start()

        self.logger.info(f"Monitoring {self.agents_dir} for agent requests")

        try:
            while not self.stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def _process_existing_active_files(self) -> None:
        """Clean up any stale active files that exist on startup."""
        for file_path in self.agents_dir.glob("*_active.jsonl"):
            agent_id = file_path.stem.replace("_active", "")
            self.logger.warning(
                f"Found stale active file from previous session: {agent_id}"
            )

            # Write an error event indicating the agent crashed
            error_message = "Agent failed due to unexpected Cortex service shutdown"
            self._write_error_and_complete(file_path, error_message)

    def _handle_active_file(self, agent_id: str, file_path: Path) -> None:
        """Handle a newly activated agent request file."""
        try:
            # Read the request from the first line
            with open(file_path, "r") as f:
                first_line = f.readline()
                if not first_line:
                    self._write_error_and_complete(file_path, "Empty request file")
                    self.logger.error(f"Empty request file: {file_path}")
                    return

                request = json.loads(first_line)

            # Validate request format
            if request.get("event") != "request":
                self._write_error_and_complete(file_path, "Invalid request format")
                self.logger.error(
                    f"Invalid request format in {file_path}: missing 'request' event"
                )
                return

            # Validate prompt early
            prompt = request.get("prompt")
            if not prompt:
                self.logger.error(f"Empty prompt in request {agent_id}")
                self._write_error_and_complete(file_path, "Empty prompt in request")
                return

            # Load persona and merge with request
            from think.utils import get_agent
            from think.mcp_tools import get_tools

            persona = request.get("persona", "default")
            config = get_agent(persona)

            # Merge request into config (request values override persona defaults)
            config.update(request)

            # Expand tools if it's a string (tool pack name)
            tools_config = config.get("tools")
            if isinstance(tools_config, str):
                try:
                    config["tools"] = get_tools(tools_config)
                except KeyError as e:
                    self.logger.warning(f"Invalid tool pack '{tools_config}': {e}, using default")
                    config["tools"] = get_tools("default")
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
            # Store the config for later use (e.g., for save field)
            self.agent_requests[agent_id] = config

            # Pass the full config through to the agent as NDJSON
            ndjson_input = json.dumps(config)

            # Prepare environment
            env = os.environ.copy()
            env["JOURNAL_PATH"] = str(self.journal_path)

            # Spawn the agent process
            cmd = ["think-agents"]
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

                    # Handle finish or error event
                    if event.get("event") in ["finish", "error"]:
                        # Check for save and handoff (only on finish)
                        if event.get("event") == "finish":
                            result = event.get("result", "")

                            # Save result if requested
                            original_request = getattr(self, "agent_requests", {}).get(
                                agent.agent_id
                            )
                            if original_request and original_request.get("save"):
                                self._save_agent_result(
                                    agent.agent_id,
                                    result,
                                    original_request["save"],
                                    original_request.get(
                                        "day"
                                    ),  # Pass optional day parameter
                                )

                            # Handle handoff
                            handoff = event.get("handoff")
                            if handoff:
                                self._spawn_handoff(agent.agent_id, result, handoff)
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
                # Write error event if no finish
                error_event = {
                    "event": "error",
                    "ts": int(time.time() * 1000),
                    "error": f"Agent exited with code {exit_code} without finish event",
                    "exit_code": exit_code,
                    "agent_id": agent.agent_id,
                }
                with open(agent.log_path, "a") as f:
                    f.write(json.dumps(error_event) + "\n")

            # Complete the file (rename from _active.jsonl to .jsonl)
            self._complete_agent_file(agent.agent_id, agent.log_path)

            # Remove from running agents and clean up stored request
            with self.lock:
                if agent.agent_id in self.running_agents:
                    del self.running_agents[agent.agent_id]
                # Clean up stored request
                if (
                    hasattr(self, "agent_requests")
                    and agent.agent_id in self.agent_requests
                ):
                    del self.agent_requests[agent.agent_id]

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
                    error_event = {
                        "event": "error",
                        "ts": int(time.time() * 1000),
                        "error": "Process failed with stderr output",
                        "trace": "\n".join(stderr_lines),
                        "exit_code": exit_code,
                        "agent_id": agent.agent_id,
                    }
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
            error_event = {
                "event": "error",
                "ts": int(time.time() * 1000),
                "error": error_message,
                "agent_id": agent_id,
            }
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
            from think.cortex_client import cortex_request

            # Get parent's config to inherit from
            parent_config = self.agent_requests.get(parent_id, {})

            # Start with parent config and overlay handoff values
            handoff_config = parent_config.copy()
            handoff_config.update(handoff)

            # Use explicit prompt or parent's result
            prompt = handoff_config.get("prompt", result)

            # Use cortex_request to create the handoff agent
            active_path = cortex_request(
                prompt=prompt,
                persona=handoff_config.get("persona", "default"),
                backend=handoff_config.get("backend", "openai"),
                handoff_from=parent_id,
                config={
                    k: v
                    for k, v in handoff_config.items()
                    if k not in ["prompt", "persona", "backend", "handoff_from"]
                },
            )

            self.logger.info(f"Spawned handoff agent {active_path} from {parent_id}")

        except Exception as e:
            self.logger.error(f"Failed to spawn handoff agent: {e}")

    def stop(self) -> None:
        """Stop the Cortex service."""
        self.stop_event.set()

        # Stop the file observer
        if self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join(timeout=5)

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
