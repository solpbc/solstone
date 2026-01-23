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
        status, exit_code = get_task_status(temp_journal, "chat", "fix_metadata")
        assert status == "pending"
        assert exit_code is None

    def test_success_status(self, temp_journal):
        """Test that exit code 0 means success."""
        state_dir = temp_journal / "maint" / "chat"
        state_dir.mkdir(parents=True)
        state_file = state_dir / "fix_metadata.jsonl"
        state_file.write_text(
            '{"event": "exec", "ts": 1000}\n'
            '{"event": "line", "line": "done"}\n'
            '{"event": "exit", "exit_code": 0}\n'
        )

        status, exit_code = get_task_status(temp_journal, "chat", "fix_metadata")
        assert status == "success"
        assert exit_code == 0

    def test_failed_status(self, temp_journal):
        """Test that non-zero exit code means failed."""
        state_dir = temp_journal / "maint" / "chat"
        state_dir.mkdir(parents=True)
        state_file = state_dir / "fix_metadata.jsonl"
        state_file.write_text(
            '{"event": "exec", "ts": 1000}\n' '{"event": "exit", "exit_code": 1}\n'
        )

        status, exit_code = get_task_status(temp_journal, "chat", "fix_metadata")
        assert status == "failed"
        assert exit_code == 1

    def test_failed_status_no_exit_event(self, temp_journal):
        """Test that file without exit event is treated as failed."""
        state_dir = temp_journal / "maint" / "chat"
        state_dir.mkdir(parents=True)
        state_file = state_dir / "fix_metadata.jsonl"
        state_file.write_text('{"event": "exec", "ts": 1000}\n')

        status, exit_code = get_task_status(temp_journal, "chat", "fix_metadata")
        assert status == "failed"
        assert exit_code is None


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
