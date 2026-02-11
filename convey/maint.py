# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""App maintenance task discovery and execution.

Maint tasks are one-time scripts that live in apps/{app}/maint/*.py.
Each task is a standalone CLI with a main() function.

State tracking:
- Completed tasks create <journal>/maint/{app}/{task}.jsonl
- The state file contains execution events (exec, line, exit)
- If file exists with exit_code: 0, task is considered complete

Discovery:
- Scans apps/*/maint/*.py for task scripts
- Skips files starting with underscore
- Tasks run in sorted order by task name (app as tiebreaker)

Execution:
- Tasks run as subprocesses
- stdout/stderr captured to state file
- Exit code determines success (0) or failure (non-zero)
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from think.utils import now_ms

logger = logging.getLogger(__name__)


@dataclass
class MaintTask:
    """A discovered maintenance task."""

    app: str
    name: str
    script_path: Path
    description: str = ""

    @property
    def qualified_name(self) -> str:
        """Return app:task qualified name."""
        return f"{self.app}:{self.name}"


def discover_tasks() -> list[MaintTask]:
    """Discover all maint tasks from apps/*/maint/*.py.

    Returns:
        List of MaintTask sorted by (app, name)
    """
    tasks = []
    apps_dir = Path(__file__).parent.parent / "apps"

    if not apps_dir.exists():
        return tasks

    for app_dir in sorted(apps_dir.iterdir()):
        if not app_dir.is_dir() or app_dir.name.startswith("_"):
            continue

        maint_dir = app_dir / "maint"
        if not maint_dir.is_dir():
            continue

        for script in sorted(maint_dir.glob("*.py")):
            if script.name.startswith("_"):
                continue

            # Extract description from module docstring
            description = ""
            try:
                content = script.read_text()
                # Find first docstring (handles files with license headers)
                for quote in ['"""', "'''"]:
                    start = content.find(quote)
                    if start >= 0:
                        end = content.find(quote, start + 3)
                        if end > start:
                            description = (
                                content[start + 3 : end].strip().split("\n")[0]
                            )
                            break
            except Exception:
                pass

            tasks.append(
                MaintTask(
                    app=app_dir.name,
                    name=script.stem,
                    script_path=script,
                    description=description,
                )
            )

    tasks.sort(key=lambda t: (t.name, t.app))
    return tasks


def get_state_file(journal: Path, app: str, task: str) -> Path:
    """Get path to task state file."""
    return journal / "maint" / app / f"{task}.jsonl"


def get_task_status(
    journal: Path, app: str, task: str
) -> tuple[str, Optional[int], Optional[int]]:
    """Check task status from state file.

    Returns:
        Tuple of (status, exit_code, ran_ts) where ran_ts is the exit event
        timestamp in epoch milliseconds, and status is:
        - "pending": No state file exists
        - "success": Completed with exit code 0
        - "failed": Completed with non-zero exit code
    """
    state_file = get_state_file(journal, app, task)

    if not state_file.exists():
        return "pending", None, None

    # Read last line for exit event
    try:
        last_line = ""
        with open(state_file, "r") as f:
            for line in f:
                if line.strip():
                    last_line = line

        if last_line:
            last_event = json.loads(last_line)
            if last_event.get("event") == "exit":
                ts = last_event.get("ts")
                exit_code = last_event.get("exit_code", -1)
                if exit_code == 0:
                    return "success", 0, ts
                return "failed", exit_code, ts
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Error reading state file {state_file}: {e}")

    # File exists but no valid exit event - treat as failed
    return "failed", None, None


def _write_event(f, event: dict) -> None:
    """Write a JSONL event to file."""
    f.write(json.dumps(event) + "\n")
    f.flush()


def run_task(
    journal: Path,
    task: MaintTask,
    emit_fn=None,
) -> tuple[bool, int]:
    """Run a maintenance task.

    Args:
        journal: Path to journal root
        task: MaintTask to run
        emit_fn: Optional function to emit Callosum events

    Returns:
        Tuple of (success, exit_code)
    """
    maint_dir = journal / "maint" / task.app
    maint_dir.mkdir(parents=True, exist_ok=True)

    state_file = get_state_file(journal, task.app, task.name)
    start_time = time.time()
    start_ts = int(start_time * 1000)

    # Build command to run the task
    cmd = [sys.executable, "-m", f"apps.{task.app}.maint.{task.name}"]

    logger.info(f"Running maint task: {task.qualified_name}")

    # Emit start event
    if emit_fn:
        emit_fn(
            "convey",
            "maint_start",
            app=task.app,
            task=task.name,
            description=task.description,
        )

    try:
        with open(state_file, "w") as f:
            # Write exec event
            _write_event(
                f,
                {
                    "event": "exec",
                    "ts": start_ts,
                    "app": task.app,
                    "task": task.name,
                    "cmd": cmd,
                },
            )

            # Run the task
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output
            if proc.stdout:
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    _write_event(
                        f,
                        {
                            "event": "line",
                            "ts": now_ms(),
                            "line": line,
                        },
                    )
                    print(f"  {line}")

            # Wait for completion
            exit_code = proc.wait()
            duration_ms = int((time.time() - start_time) * 1000)

            # Write exit event
            _write_event(
                f,
                {
                    "event": "exit",
                    "ts": now_ms(),
                    "exit_code": exit_code,
                    "duration_ms": duration_ms,
                },
            )

        success = exit_code == 0

        # Emit completion event
        if emit_fn:
            emit_fn(
                "convey",
                "maint_complete",
                app=task.app,
                task=task.name,
                exit_code=exit_code,
                duration_ms=duration_ms,
                success=success,
            )

        if success:
            logger.info(
                f"Completed maint task: {task.qualified_name} ({duration_ms}ms)"
            )
        else:
            logger.warning(
                f"Maint task failed: {task.qualified_name} (exit code {exit_code})"
            )

        return success, exit_code

    except Exception as e:
        logger.error(f"Error running maint task {task.qualified_name}: {e}")

        # Try to write error to state file
        try:
            with open(state_file, "a") as f:
                _write_event(
                    f,
                    {
                        "event": "exit",
                        "ts": now_ms(),
                        "exit_code": -1,
                        "error": str(e),
                    },
                )
        except Exception:
            pass

        if emit_fn:
            emit_fn(
                "convey",
                "maint_complete",
                app=task.app,
                task=task.name,
                exit_code=-1,
                success=False,
                error=str(e),
            )

        return False, -1


def run_pending_tasks(journal: Path, emit_fn=None) -> tuple[int, int]:
    """Run all pending maintenance tasks.

    Args:
        journal: Path to journal root
        emit_fn: Optional function to emit Callosum events

    Returns:
        Tuple of (tasks_run, tasks_succeeded)
    """
    tasks = discover_tasks()
    pending = []

    for task in tasks:
        status, _, _ = get_task_status(journal, task.app, task.name)
        if status == "pending":
            pending.append(task)

    if not pending:
        return 0, 0

    logger.info(f"Found {len(pending)} pending maintenance task(s)")

    ran = 0
    succeeded = 0

    for task in pending:
        ran += 1
        success, _ = run_task(journal, task, emit_fn)
        if success:
            succeeded += 1

    return ran, succeeded


def list_tasks(journal: Path) -> list[dict]:
    """List all tasks with their status.

    Returns:
        List of dicts with task info and status
    """
    tasks = discover_tasks()
    result = []

    for task in tasks:
        status, exit_code, ran_ts = get_task_status(journal, task.app, task.name)
        result.append(
            {
                "app": task.app,
                "name": task.name,
                "qualified_name": task.qualified_name,
                "description": task.description,
                "status": status,
                "exit_code": exit_code,
                "ran_ts": ran_ts,
                "state_file": (
                    str(get_state_file(journal, task.app, task.name))
                    if status != "pending"
                    else None
                ),
            }
        )

    return result


def get_task_by_name(name: str) -> Optional[MaintTask]:
    """Get a task by qualified name (app:task) or just task name.

    Args:
        name: Task name, either "app:task" or just "task"

    Returns:
        MaintTask if found, None otherwise
    """
    tasks = discover_tasks()

    # Try qualified name first
    if ":" in name:
        for task in tasks:
            if task.qualified_name == name:
                return task
    else:
        # Try just task name (must be unique)
        matches = [t for t in tasks if t.name == name]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            logger.warning(
                f"Ambiguous task name '{name}', found in: "
                f"{', '.join(t.app for t in matches)}"
            )

    return None
