# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Callosum-based talent process manager for solstone.

Cortex listens for talent requests via the Callosum message bus and manages
talent process lifecycle:
- Receives requests via Callosum (tract="cortex", event="request")
- Creates <talent>/<timestamp>_active.jsonl files to track active uses
- Spawns talent processes and captures their stdout events
- Broadcasts all talent events back to Callosum
- Renames to <talent>/<timestamp>.jsonl when complete

Talent files provide persistence and historical record, while Callosum provides
real-time event distribution to all interested services.
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
from typing import Any, Dict, Optional

from think.callosum import CallosumConnection
from think.runner import _atomic_symlink
from think.talents import TALENT_EXECUTION_MODULE
from think.utils import get_journal, get_project_root, get_rev, now_ms


class TalentProcess:
    """Manages a running talent subprocess."""

    def __init__(self, use_id: str, process: subprocess.Popen, log_path: Path):
        self.use_id = use_id
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
                    f"Talent {self.use_id} didn't stop gracefully, killing"
                )
                self.process.kill()
                self.process.wait()  # Ensure zombie is reaped


class CortexService:
    """Callosum-based talent process manager."""

    def __init__(self, journal_path: Optional[str] = None):
        self.journal_path = Path(journal_path or get_journal())
        self.talents_dir = self.journal_path / "talents"
        self.talents_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.running_uses: Dict[str, TalentProcess] = {}
        self.use_requests: Dict[str, Dict[str, Any]] = {}  # Store use requests
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.shutdown_requested = threading.Event()

        # Callosum connection for receiving requests and broadcasting events
        self.callosum = CallosumConnection(defaults={"rev": get_rev()})

    def _create_error_event(
        self,
        use_id: str,
        error: str,
        trace: Optional[str] = None,
        exit_code: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create standardized error event."""
        event = {
            "event": "error",
            "ts": now_ms(),
            "use_id": use_id,
            "error": error,
        }
        if trace:
            event["trace"] = trace
        if exit_code is not None:
            event["exit_code"] = exit_code
        return event

    def _recover_orphaned_uses(self, active_files: list) -> None:
        """Recover orphaned active talent files from a previous crash.

        Appends an error event to each file and renames to completed.
        """
        for file_path in active_files:
            use_id = file_path.stem.replace("_active", "")
            try:
                error_event = self._create_error_event(
                    use_id, "Recovered: Cortex restarted while talent was running"
                )
                with open(file_path, "a") as f:
                    f.write(json.dumps(error_event) + "\n")

                completed_path = file_path.parent / f"{use_id}.jsonl"
                file_path.rename(completed_path)
                self.logger.warning(f"Recovered orphaned talent: {use_id}")
            except Exception as e:
                self.logger.error(f"Failed to recover talent {use_id}: {e}")

    def start(self) -> None:
        """Start listening for talent requests via Callosum."""
        # Recover any orphaned active files from previous crash
        active_files = list(self.talents_dir.glob("*/*_active.jsonl"))
        if active_files:
            self.logger.warning(
                f"Found {len(active_files)} orphaned talent use(s), recovering..."
            )
            self._recover_orphaned_uses(active_files)

        # Connect to Callosum to receive requests
        try:
            self.callosum.start(callback=self._handle_callosum_message)
            self.logger.info("Connected to Callosum message bus")
            self.callosum.emit(
                "supervisor", "request", cmd=["sol", "providers", "check"]
            )
            self.logger.info("Requested providers health check via supervisor")
        except Exception as e:
            self.logger.error(f"Failed to connect to Callosum: {e}")
            sys.exit(1)

        # Start status emission thread
        threading.Thread(
            target=self._emit_periodic_status,
            name="cortex-status",
            daemon=True,
        ).start()

        self.logger.info("Cortex service started, listening for talent requests")

        while True:
            try:
                while not self.stop_event.is_set():
                    time.sleep(1)
                    # Exit when idle during shutdown
                    if self.shutdown_requested.is_set():
                        with self.lock:
                            if len(self.running_uses) == 0:
                                self.logger.info(
                                    "No talent uses running, exiting gracefully"
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
        """Handle a new talent request from Callosum.

        Cortex is a minimal process manager - it only handles:
        - File lifecycle (<talent>/<id>_active.jsonl -> <talent>/<id>.jsonl)
        - Process spawning and monitoring
        - Event relay to Callosum

        All config loading, validation, and hydration is done by think.talents.
        Cortex only resolves talent cwd early so the child process starts in
        the correct working directory.
        """
        use_id = request.get("use_id")
        if not use_id:
            self.logger.error("Received request without use_id")
            return

        # Skip if this use is already being processed
        with self.lock:
            if use_id in self.running_uses:
                self.logger.debug(f"Talent use {use_id} already running, skipping")
                return

        # Create _active.jsonl file (exclusive creation to prevent race conditions)
        name = request["name"]
        safe_name = name.replace(":", "--")
        talent_subdir = self.talents_dir / safe_name
        talent_subdir.mkdir(parents=True, exist_ok=True)
        file_path = talent_subdir / f"{use_id}_active.jsonl"
        if file_path.exists():
            self.logger.debug(f"Talent use {use_id} already claimed by another process")
            return

        try:
            with open(file_path, "x") as f:  # 'x' mode fails if file exists
                f.write(json.dumps(request) + "\n")
        except FileExistsError:
            return

        self.logger.info(f"Processing talent request: {use_id}")

        # Store request for later use (output writing)
        with self.lock:
            self.use_requests[use_id] = request

        # Spawn talent process - it handles all validation/hydration
        try:
            self._spawn_subprocess(
                use_id,
                file_path,
                request,
                [sys.executable, "-m", TALENT_EXECUTION_MODULE],
                "talent",
            )
        except Exception as e:
            self.logger.exception(f"Failed to spawn talent {use_id}: {e}")
            self._write_error_and_complete(file_path, f"Failed to spawn talent: {e}")

    def _spawn_subprocess(
        self,
        use_id: str,
        file_path: Path,
        config: Dict[str, Any],
        cmd: list[str],
        process_type: str,
    ) -> None:
        """Spawn a subprocess and monitor its output.

        Args:
            use_id: Unique identifier for this process
            file_path: Path to the JSONL log file
            config: Configuration dict to pass via NDJSON stdin
            cmd: Command to run (e.g., [sys.executable, "-m", TALENT_EXECUTION_MODULE])
            process_type: Label for logging ("talent")
        """
        try:
            # Store the config for later use - thread safe
            with self.lock:
                self.use_requests[use_id] = config

            # Pass the full config through as NDJSON
            ndjson_input = json.dumps(config)

            # Prepare environment
            env = os.environ.copy()

            # Promote top-level config keys to environment so tools can read
            # them as defaults (e.g., sol call todos add uses SOL_FACET).
            # Explicit env overrides below take precedence.
            if config.get("facet"):
                env["SOL_FACET"] = str(config["facet"])
            if config.get("day"):
                env["SOL_DAY"] = str(config["day"])

            # Apply explicit env overrides (from thinking.py etc.) — these win
            env_overrides = config.get("env")
            if env_overrides and isinstance(env_overrides, dict):
                env.update({k: str(v) for k, v in env_overrides.items()})

            # Spawn the subprocess
            self.logger.info(f"Spawning {process_type} {use_id}: {cmd}")
            self.logger.debug(f"NDJSON input: {ndjson_input}")
            subprocess_cwd = None
            if process_type == "talent":
                from think.talent import get_talent

                talent_key = str(config["name"])
                talent_config = get_talent(talent_key)
                if talent_config.get("type") == "cogitate":
                    # Resolve here because prepare_config() runs inside think.talents.
                    cwd_value = talent_config.get("cwd")
                    if cwd_value == "journal":
                        try:
                            subprocess_cwd = str(Path(get_journal()))
                        except Exception as exc:
                            raise RuntimeError(
                                f"Cannot resolve cwd for talent '{talent_key}'"
                            ) from exc
                    elif cwd_value == "repo":
                        subprocess_cwd = get_project_root()
                    else:
                        raise RuntimeError(
                            f"Cannot resolve cwd for talent '{talent_key}'"
                        )

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,
                cwd=subprocess_cwd,
            )

            # Send input and close stdin
            process.stdin.write(ndjson_input + "\n")
            process.stdin.close()

            # Track the running process
            agent = TalentProcess(use_id, process, file_path)
            with self.lock:
                self.running_uses[use_id] = agent

            # Set up timeout (default to 10 minutes if not specified)
            timeout_seconds = config.get("timeout_seconds", 600)
            agent.timeout_timer = threading.Timer(
                timeout_seconds,
                lambda: self._timeout_talent(use_id, agent, timeout_seconds),
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
                f"{process_type.capitalize()} {use_id} spawned successfully "
                f"(PID: {process.pid})"
            )

        except Exception as e:
            self.logger.exception(f"Failed to spawn {process_type} {use_id}: {e}")
            self._write_error_and_complete(
                file_path, f"Failed to spawn {process_type}: {e}"
            )

    def _timeout_talent(
        self, use_id: str, agent: TalentProcess, timeout_seconds: int
    ) -> None:
        """Handle talent timeout."""
        if agent.is_running():
            self.logger.warning(
                f"Talent {use_id} timed out after {timeout_seconds} seconds"
            )
            error_event = self._create_error_event(
                use_id, f"Talent timed out after {timeout_seconds} seconds"
            )
            try:
                with open(agent.log_path, "a") as f:
                    f.write(json.dumps(error_event) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to write timeout event: {e}")

            # Broadcast to callosum so wait_for_uses detects immediately
            try:
                event_copy = error_event.copy()
                event_type = event_copy.pop("event", "error")
                self.callosum.emit("cortex", event_type, **event_copy)
            except Exception:
                pass

            agent.stop()

    def _monitor_stdout(self, agent: TalentProcess) -> None:
        """Monitor talent stdout and append events to the JSONL file."""
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

                        # Ensure event has timestamp and use_id
                        if "ts" not in event:
                            event["ts"] = now_ms()
                        if "use_id" not in event:
                            event["use_id"] = agent.use_id

                        # Inject agent name for WebSocket consumers
                        with self.lock:
                            _req = self.use_requests.get(agent.use_id)
                        if _req and "name" not in event:
                            event["name"] = _req.get("name", "")

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
                            # Capture model and provider for status reporting
                            with self.lock:
                                if agent.use_id in self.use_requests:
                                    model = event.get("model")
                                    if model:
                                        self.use_requests[agent.use_id]["model"] = model
                                    provider = event.get("provider")
                                    if provider:
                                        self.use_requests[agent.use_id]["provider"] = (
                                            provider
                                        )

                        # Handle finish or error event
                        if event.get("event") in ["finish", "error"]:
                            # Check for output (only on finish)
                            if event.get("event") == "finish":
                                result = event.get("result", "")

                                # Get original request (thread-safe access)
                                with self.lock:
                                    original_request = self.use_requests.get(
                                        agent.use_id
                                    )

                                # Log token usage if available
                                usage_data = event.get("usage")
                                if usage_data and original_request:
                                    try:
                                        from think.models import log_token_usage
                                        from think.talent import key_to_context

                                        model = original_request.get("model", "unknown")
                                        name = original_request.get("name", "unknown")
                                        context = key_to_context(name)

                                        # Extract segment from env if set (flat merge puts env at top level)
                                        env_config = original_request.get("env", {})
                                        segment = (
                                            env_config.get("SOL_SEGMENT")
                                            if env_config
                                            else None
                                        )

                                        log_token_usage(
                                            model=model,
                                            usage=usage_data,
                                            context=context,
                                            segment=segment,
                                            type="cogitate",
                                        )
                                    except Exception as e:
                                        self.logger.warning(
                                            f"Failed to log token usage for talent {agent.use_id}: {e}"
                                        )

                                # Write output if requested
                                if original_request and original_request.get("output"):
                                    self._write_output(
                                        agent.use_id,
                                        result,
                                        original_request,
                                    )

                            # Break to trigger cleanup
                            break

                    except json.JSONDecodeError:
                        # Non-JSON output becomes info event
                        info_event = {
                            "event": "info",
                            "ts": now_ms(),
                            "message": line,
                            "use_id": agent.use_id,
                        }
                        with open(agent.log_path, "a") as f:
                            f.write(json.dumps(info_event) + "\n")

        except Exception as e:
            self.logger.error(f"Error monitoring stdout for agent {agent.use_id}: {e}")
        finally:
            # Wait for process to fully exit (reaps zombie)
            exit_code = agent.process.wait()
            self.logger.info(f"Talent {agent.use_id} exited with code {exit_code}")

            # Check if finish event was emitted
            has_finish = self._has_finish_event(agent.log_path)

            if not has_finish:
                # Write error event if no finish using standardized format
                error_event = self._create_error_event(
                    agent.use_id,
                    f"Talent exited with code {exit_code} without finish event",
                    exit_code=exit_code,
                )
                with open(agent.log_path, "a") as f:
                    f.write(json.dumps(error_event) + "\n")

            # Complete the file (rename from _active.jsonl to .jsonl)
            self._complete_use_file(agent.use_id, agent.log_path)

            # Remove from running agents and clean up stored request (thread-safe)
            with self.lock:
                if agent.use_id in self.running_uses:
                    del self.running_uses[agent.use_id]
                # Clean up stored request
                if agent.use_id in self.use_requests:
                    del self.use_requests[agent.use_id]

    def _monitor_stderr(self, agent: TalentProcess) -> None:
        """Monitor talent stderr for errors."""
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
                        # Pass through to cortex stderr with talent prefix for traceability
                        print(
                            f"[talent:{agent.use_id}:stderr] {stripped}",
                            file=sys.stderr,
                            flush=True,
                        )

        except Exception as e:
            self.logger.error(f"Error monitoring stderr for agent {agent.use_id}: {e}")
        finally:
            # If process failed with stderr output, write error event
            if stderr_lines:
                exit_code = agent.process.poll()
                if exit_code is not None and exit_code != 0:
                    error_event = self._create_error_event(
                        agent.use_id,
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
                    except json.JSONDecodeError as exc:
                        self.logger.warning(
                            "Malformed event in %s while scanning for finish: %s",
                            file_path,
                            exc,
                        )
                        continue
        except FileNotFoundError:
            self.logger.debug("Use log disappeared before finish scan: %s", file_path)
        except OSError as exc:
            self.logger.warning(
                "Failed to scan %s for finish events: %s", file_path, exc
            )
        return False

    def _complete_use_file(self, use_id: str, file_path: Path) -> None:
        """Complete a talent use by renaming the file from _active.jsonl to .jsonl."""
        try:
            completed_path = file_path.parent / f"{use_id}.jsonl"
            file_path.rename(completed_path)
            self.logger.info(f"Completed talent use {use_id}: {completed_path}")

            # Create convenience symlink: {name}.log -> {name}/{use_id}.jsonl
            request = self.use_requests.get(use_id)
            if request:
                name = request.get("name")
                if name:
                    safe_name = name.replace(":", "--")
                    link_path = self.talents_dir / f"{safe_name}.log"
                    _atomic_symlink(link_path, f"{safe_name}/{use_id}.jsonl")
                    self.logger.debug(
                        f"Symlinked {safe_name}.log -> {safe_name}/{use_id}.jsonl"
                    )

                    # Append summary to day index
                    self._append_day_index(use_id, request, completed_path)
                else:
                    self.logger.debug(
                        f"No name in request for {use_id}, skipping symlink"
                    )
        except Exception as e:
            self.logger.error(f"Failed to complete talent file {use_id}: {e}")

    def _append_day_index(
        self, use_id: str, request: Dict[str, Any], completed_path: Path
    ) -> None:
        """Append talent-use summary to the day index file."""
        try:
            # Determine day from request or use_id timestamp
            day = request.get("day")
            if not day:
                from datetime import datetime

                ts_seconds = int(use_id) / 1000
                day = datetime.fromtimestamp(ts_seconds).strftime("%Y%m%d")

            start_ts = request.get("ts", 0)

            # Read last few lines to find finish/error event for runtime
            runtime_seconds = None
            status = "completed"
            try:
                with open(completed_path, "r") as f:
                    lines = f.readlines()
                for line in reversed(lines[-10:]):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        event_type = event.get("event")
                        if event_type == "finish":
                            end_ts = event.get("ts", 0)
                            if end_ts and start_ts:
                                runtime_seconds = round((end_ts - start_ts) / 1000.0, 1)
                            break
                        if event_type == "error":
                            status = "error"
                            end_ts = event.get("ts", 0)
                            if end_ts and start_ts:
                                runtime_seconds = round((end_ts - start_ts) / 1000.0, 1)
                            break
                    except json.JSONDecodeError:
                        continue
            except Exception:
                pass

            summary = {
                "use_id": use_id,
                "name": request["name"],
                "day": day,
                "facet": request.get("facet"),
                "ts": start_ts,
                "status": status,
                "runtime_seconds": runtime_seconds,
                "provider": request.get("provider"),
                "model": request.get("model"),
                "schedule": request.get("schedule"),
            }

            day_index_path = self.talents_dir / f"{day}.jsonl"
            with open(day_index_path, "a") as f:
                f.write(json.dumps(summary) + "\n")
                f.flush()

        except Exception as e:
            self.logger.error(f"Failed to append day index for {use_id}: {e}")

    def _write_error_and_complete(self, file_path: Path, error_message: str) -> None:
        """Write an error event to the file and mark it as complete."""
        try:
            use_id = file_path.stem.replace("_active", "")
            error_event = self._create_error_event(use_id, error_message)
            with open(file_path, "a") as f:
                f.write(json.dumps(error_event) + "\n")

            # Complete the file
            self._complete_use_file(use_id, file_path)
        except Exception as e:
            self.logger.error(f"Failed to write error and complete: {e}")

    def _write_output(self, use_id: str, result: str, config: Dict[str, Any]) -> None:
        """Write talent output to config["output_path"].

        The output path is set by the caller — either derived by
        prepare_config in think.talents (day/segment talents) or computed
        by thinking.py via get_activity_output_path (activity talents).
        Cortex does not derive paths itself.
        """
        output_path_str = config.get("output_path")
        if not output_path_str:
            return

        try:
            output_path = Path(output_path_str)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)

            self.logger.info(f"Wrote talent {use_id} output to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to write talent {use_id} output: {e}")

    def stop(self) -> None:
        """Stop the Cortex service."""
        self.stop_event.set()

        # Close Callosum connection
        if self.callosum:
            self.callosum.stop()

        # Stop all running talent uses
        with self.lock:
            for agent in self.running_uses.values():
                agent.stop()

    def _emit_periodic_status(self) -> None:
        """Emit status events every 5 seconds (runs in background thread)."""
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    uses = []
                    for use_id, agent_proc in self.running_uses.items():
                        config = self.use_requests.get(use_id, {})
                        uses.append(
                            {
                                "use_id": use_id,
                                "name": config.get("name", "unknown"),
                                "provider": config.get("provider", "unknown"),
                                "elapsed_seconds": int(
                                    time.time() - agent_proc.start_time
                                ),
                            }
                        )

                # Only emit status when there are active talent uses
                if uses:
                    self.callosum.emit(
                        "cortex",
                        "status",
                        running_uses=len(uses),
                        uses=uses,
                    )
            except Exception as e:
                self.logger.debug(f"Status emission failed: {e}")

            time.sleep(5)

    def get_status(self) -> Dict[str, Any]:
        """Get service status information."""
        with self.lock:
            return {
                "running_uses": len(self.running_uses),
                "use_ids": list(self.running_uses.keys()),
            }


def main() -> None:
    """CLI entry point for the Cortex service."""
    import argparse

    from think.utils import require_solstone, setup_cli

    parser = argparse.ArgumentParser(description="solstone Cortex Talent Manager")
    args = setup_cli(parser)
    require_solstone()

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
