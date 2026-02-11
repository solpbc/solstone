# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the maint (maintenance task) system."""

import json
from pathlib import Path

import pytest

from convey.maint import (
    MaintTask,
    get_state_file,
    get_task_status,
    list_tasks,
    run_task,
)


@pytest.fixture
def temp_journal(tmp_path):
    """Create a temporary journal directory."""
    journal = tmp_path / "journal"
    journal.mkdir()
    return journal


class TestMaintTask:
    """Tests for MaintTask dataclass."""

    def test_qualified_name(self):
        task = MaintTask(app="chat", name="fix_metadata", script_path=Path("/dummy"))
        assert task.qualified_name == "chat:fix_metadata"


class TestStatusTracking:
    """Tests for task status tracking."""

    def test_get_state_file_path(self, temp_journal):
        """Test state file path generation."""
        path = get_state_file(temp_journal, "chat", "fix_metadata")
        assert path == temp_journal / "maint" / "chat" / "fix_metadata.jsonl"

    def test_pending_status_no_file(self, temp_journal):
        """Test that missing state file means pending."""
        status, exit_code, ran_ts = get_task_status(
            temp_journal, "chat", "fix_metadata"
        )
        assert status == "pending"
        assert exit_code is None
        assert ran_ts is None

    def test_success_status(self, temp_journal):
        """Test that exit code 0 means success."""
        state_dir = temp_journal / "maint" / "chat"
        state_dir.mkdir(parents=True)
        state_file = state_dir / "fix_metadata.jsonl"
        state_file.write_text(
            '{"event": "exec", "ts": 1000}\n'
            '{"event": "line", "line": "done"}\n'
            '{"event": "exit", "ts": 3000, "exit_code": 0}\n'
        )

        status, exit_code, ran_ts = get_task_status(
            temp_journal, "chat", "fix_metadata"
        )
        assert status == "success"
        assert exit_code == 0
        assert ran_ts is not None

    def test_failed_status(self, temp_journal):
        """Test that non-zero exit code means failed."""
        state_dir = temp_journal / "maint" / "chat"
        state_dir.mkdir(parents=True)
        state_file = state_dir / "fix_metadata.jsonl"
        state_file.write_text(
            '{"event": "exec", "ts": 1000}\n'
            '{"event": "exit", "ts": 2000, "exit_code": 1}\n'
        )

        status, exit_code, ran_ts = get_task_status(
            temp_journal, "chat", "fix_metadata"
        )
        assert status == "failed"
        assert exit_code == 1
        assert ran_ts == 2000

    def test_in_progress_status_no_exit_event(self, temp_journal):
        """Test that file without exit event is treated as in-progress."""
        state_dir = temp_journal / "maint" / "chat"
        state_dir.mkdir(parents=True)
        state_file = state_dir / "fix_metadata.jsonl"
        state_file.write_text('{"event": "exec", "ts": 1000}\n')

        status, exit_code, ran_ts = get_task_status(
            temp_journal, "chat", "fix_metadata"
        )
        assert status == "in_progress"
        assert exit_code is None
        assert ran_ts == 1000


class TestListTasks:
    """Tests for listing tasks with status metadata."""

    def test_list_tasks_includes_ran_ts_and_state_file(self, temp_journal, monkeypatch):
        tasks = [
            MaintTask(app="chat", name="done", script_path=Path("/dummy/done.py")),
            MaintTask(app="chat", name="failed", script_path=Path("/dummy/failed.py")),
            MaintTask(
                app="chat", name="pending", script_path=Path("/dummy/pending.py")
            ),
        ]

        def mock_discover_tasks():
            return tasks

        monkeypatch.setattr("convey.maint.discover_tasks", mock_discover_tasks)

        # Success task with timestamp
        success_file = get_state_file(temp_journal, "chat", "done")
        success_file.parent.mkdir(parents=True, exist_ok=True)
        success_file.write_text(
            '{"event": "exec", "ts": 1000}\n'
            '{"event": "exit", "ts": 5000, "exit_code": 0}\n'
        )

        # Failed task with timestamp
        failed_file = get_state_file(temp_journal, "chat", "failed")
        failed_file.write_text(
            '{"event": "exec", "ts": 2000}\n'
            '{"event": "exit", "ts": 6000, "exit_code": 2}\n'
        )

        listed = list_tasks(temp_journal)
        by_name = {task["name"]: task for task in listed}

        done_task = by_name["done"]
        assert done_task["status"] == "success"
        assert done_task["ran_ts"] == 5000
        assert done_task["state_file"] == str(success_file)

        failed_task = by_name["failed"]
        assert failed_task["status"] == "failed"
        assert failed_task["ran_ts"] == 6000
        assert failed_task["state_file"] == str(failed_file)

        pending_task = by_name["pending"]
        assert pending_task["status"] == "pending"
        assert pending_task["ran_ts"] is None
        assert pending_task["state_file"] is None

    def test_list_tasks_includes_duration_and_line_count(
        self, temp_journal, monkeypatch
    ):
        """Test that list_tasks returns duration_ms and line_count."""
        tasks = [
            MaintTask(app="chat", name="done", script_path=Path("/dummy/done.py")),
            MaintTask(
                app="chat", name="pending", script_path=Path("/dummy/pending.py")
            ),
        ]

        def mock_discover_tasks():
            return tasks

        monkeypatch.setattr("convey.maint.discover_tasks", mock_discover_tasks)

        # Success task with line events and duration
        success_file = get_state_file(temp_journal, "chat", "done")
        success_file.parent.mkdir(parents=True, exist_ok=True)
        success_file.write_text(
            '{"event": "exec", "ts": 1000}\n'
            '{"event": "line", "ts": 1500, "line": "step 1"}\n'
            '{"event": "line", "ts": 2000, "line": "step 2"}\n'
            '{"event": "line", "ts": 2500, "line": "step 3"}\n'
            '{"event": "exit", "ts": 3000, "exit_code": 0, "duration_ms": 2000}\n'
        )

        listed = list_tasks(temp_journal)
        by_name = {task["name"]: task for task in listed}

        done = by_name["done"]
        assert done["duration_ms"] == 2000
        assert done["line_count"] == 3

        pending = by_name["pending"]
        assert pending["duration_ms"] is None
        assert pending["line_count"] == 0

    def test_list_tasks_in_progress_status(self, temp_journal, monkeypatch):
        """Test that list_tasks returns in_progress for tasks without exit event."""
        tasks = [
            MaintTask(
                app="chat", name="running", script_path=Path("/dummy/running.py")
            ),
        ]

        def mock_discover_tasks():
            return tasks

        monkeypatch.setattr("convey.maint.discover_tasks", mock_discover_tasks)

        # In-progress task: exec event but no exit event
        state_file = get_state_file(temp_journal, "chat", "running")
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(
            '{"event": "exec", "ts": 1000}\n'
            '{"event": "line", "ts": 1500, "line": "working..."}\n'
        )

        listed = list_tasks(temp_journal)
        assert len(listed) == 1
        t = listed[0]
        assert t["status"] == "in_progress"
        assert t["ran_ts"] == 1000
        assert t["exit_code"] is None
        assert t["duration_ms"] is None
        assert t["line_count"] == 1


class TestFormatDuration:
    """Tests for duration formatting."""

    def test_milliseconds(self):
        from convey.maint_cli import _format_duration

        assert _format_duration(0) == "0ms"
        assert _format_duration(500) == "500ms"
        assert _format_duration(999) == "999ms"

    def test_seconds(self):
        from convey.maint_cli import _format_duration

        assert _format_duration(1000) == "1s"
        assert _format_duration(2500) == "2s"
        assert _format_duration(59999) == "59s"

    def test_minutes(self):
        from convey.maint_cli import _format_duration

        assert _format_duration(60000) == "1m 0s"
        assert _format_duration(143000) == "2m 23s"


class TestRunTask:
    """Tests for running individual tasks."""

    def test_run_successful_task(self, temp_journal, monkeypatch):
        """Test running a successful task creates correct state file."""
        import subprocess

        task = MaintTask(
            app="test_app",
            name="success_task",
            script_path=Path("/dummy/success.py"),
        )

        # Mock subprocess to simulate success
        def mock_popen(*args, **kwargs):
            class MockProc:
                stdout = iter(["Processing...\n", "Done!\n"])

                def wait(self):
                    return 0

            return MockProc()

        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        success, exit_code = run_task(temp_journal, task)

        assert success is True
        assert exit_code == 0

        # Check state file was created
        state_file = get_state_file(temp_journal, "test_app", "success_task")
        assert state_file.exists()

        # Verify contents
        lines = state_file.read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]

        assert events[0]["event"] == "exec"
        assert events[0]["app"] == "test_app"
        assert events[0]["task"] == "success_task"

        # Should have line events
        line_events = [e for e in events if e["event"] == "line"]
        assert len(line_events) == 2

        # Last event should be exit with code 0
        assert events[-1]["event"] == "exit"
        assert events[-1]["exit_code"] == 0

    def test_run_failing_task(self, temp_journal, monkeypatch):
        """Test running a failing task records failure."""
        import subprocess

        task = MaintTask(
            app="test_app",
            name="fail_task",
            script_path=Path("/dummy/fail.py"),
        )

        # Mock subprocess to simulate failure
        def mock_popen(*args, **kwargs):
            class MockProc:
                stdout = iter(["About to fail\n"])

                def wait(self):
                    return 1

            return MockProc()

        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        success, exit_code = run_task(temp_journal, task)

        assert success is False
        assert exit_code == 1

        # Check state file was created with failure
        state_file = get_state_file(temp_journal, "test_app", "fail_task")
        assert state_file.exists()

        lines = state_file.read_text().strip().split("\n")
        last_event = json.loads(lines[-1])
        assert last_event["exit_code"] == 1

    def test_run_task_emits_events(self, temp_journal, monkeypatch):
        """Test that run_task calls emit_fn with correct events."""
        import subprocess

        task = MaintTask(
            app="test_app",
            name="emit_task",
            script_path=Path("/dummy/emit.py"),
            description="Test task",
        )

        def mock_popen(*args, **kwargs):
            class MockProc:
                stdout = iter(["Working\n"])

                def wait(self):
                    return 0

            return MockProc()

        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        emitted = []

        def capture_emit(tract, event, **kwargs):
            emitted.append((tract, event, kwargs))

        success, _ = run_task(temp_journal, task, emit_fn=capture_emit)

        assert success is True
        assert len(emitted) == 2

        # Check start event
        assert emitted[0][0] == "convey"
        assert emitted[0][1] == "maint_start"
        assert emitted[0][2]["app"] == "test_app"
        assert emitted[0][2]["task"] == "emit_task"

        # Check complete event
        assert emitted[1][0] == "convey"
        assert emitted[1][1] == "maint_complete"
        assert emitted[1][2]["success"] is True
