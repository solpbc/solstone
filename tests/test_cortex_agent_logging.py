#!/usr/bin/env python3
"""Test that agents only write to stdout and cortex captures everything."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def test_json_event_writer():
    """Test that JSONEventWriter outputs to stdout and optionally to file."""
    from think.agents import JSONEventWriter

    # Test with file output (for -o option compatibility)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        test_file = f.name

    try:
        writer = JSONEventWriter(test_file)

        # Test that emit works
        test_event = {"event": "test", "ts": int(time.time() * 1000), "message": "test"}
        writer.emit(test_event)  # This will print to stdout AND write to file

        writer.close()

        # Verify file was created and has content
        assert Path(
            test_file
        ).exists(), "JSONEventWriter should create file when path provided"
        content = Path(test_file).read_text()
        assert '"event": "test"' in content
    finally:
        Path(test_file).unlink(missing_ok=True)


def test_agent_no_file_creation():
    """Test that agents don't create log files in journal directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up a mock agent run that would previously create files
        agents_dir = Path(tmpdir) / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)

        # Import the deprecated JournalEventWriter
        from think.agents import JournalEventWriter

        # Create a journal writer (should no longer create files)
        with tempfile.TemporaryDirectory() as journal_tmp:
            env_backup = os.environ.get("JOURNAL_PATH")
            try:
                os.environ["JOURNAL_PATH"] = journal_tmp
                writer = JournalEventWriter()

                # Emit some events
                writer.emit({"event": "start", "ts": int(time.time() * 1000)})
                writer.emit({"event": "finish", "ts": int(time.time() * 1000)})
                writer.close()

                # Verify no files were created
                journal_agents = Path(journal_tmp) / "agents"
                if journal_agents.exists():
                    files = list(journal_agents.glob("*.jsonl"))
                    assert (
                        len(files) == 0
                    ), f"JournalEventWriter should not create files, found: {files}"
            finally:
                if env_backup:
                    os.environ["JOURNAL_PATH"] = env_backup
                else:
                    os.environ.pop("JOURNAL_PATH", None)


def test_cortex_style_capture():
    """Test simulating how cortex captures stdout to log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = {"JOURNAL_PATH": tmpdir}

        # Create agents directory like cortex does
        agents_dir = Path(tmpdir) / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)

        # Generate agent ID like cortex does
        agent_id = str(int(time.time() * 1000))
        log_path = agents_dir / f"{agent_id}.jsonl"

        # Write start event like cortex does
        start_event = {
            "event": "start",
            "ts": int(time.time() * 1000),
            "prompt": "Test prompt",
            "persona": "default",
            "model": "",
            "backend": "openai",
        }

        with open(log_path, "w") as f:
            f.write(json.dumps(start_event) + "\n")

        # Run agent and capture stdout
        proc = subprocess.Popen(
            [sys.executable, "-m", "think.agents", "-q", "Say hello"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,  # Line buffering like cortex
        )

        # Read stdout and write to log file (like cortex does)
        captured_events = []
        if proc.stdout:
            for line in proc.stdout:
                line = line.strip()
                if line:
                    with open(log_path, "a") as f:
                        f.write(line + "\n")
                    try:
                        event = json.loads(line)
                        captured_events.append(event)
                    except json.JSONDecodeError:
                        # Write non-JSON as info event like cortex does
                        info_event = {
                            "event": "info",
                            "ts": int(time.time() * 1000),
                            "message": line,
                        }
                        with open(log_path, "a") as f:
                            f.write(json.dumps(info_event) + "\n")

        proc.wait()

        # Write finish event like cortex does
        exit_code = proc.poll()
        if exit_code is not None:
            finish_event = {
                "event": "finish" if exit_code == 0 else "error",
                "ts": int(time.time() * 1000),
                "exit_code": exit_code,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(finish_event) + "\n")

        # Check log file has content
        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) > 1, "Log file should have multiple events"

        # Verify all lines are valid JSON
        for i, line in enumerate(lines):
            try:
                json.loads(line)
            except json.JSONDecodeError:
                raise AssertionError(f"Invalid JSON at line {i+1}: {line[:50]}...")

        # Check stderr handling
        if proc.stderr:
            stderr_output = proc.stderr.read()
            if stderr_output:
                # Would be logged as error events by cortex
                for line in stderr_output.strip().split("\n"):
                    if line:
                        error_event = {
                            "event": "error",
                            "ts": int(time.time() * 1000),
                            "message": line,
                            "source": "stderr",
                        }
                        # In real cortex, this would be written to log and broadcast


if __name__ == "__main__":
    # Redirect stdout to suppress event output during tests
    import io

    original_stdout = sys.stdout

    sys.stdout = io.StringIO()
    test_json_event_writer()
    sys.stdout = original_stdout
    print("✓ test_json_event_writer passed")

    test_agent_no_file_creation()
    print("✓ test_agent_no_file_creation passed")

    test_cortex_style_capture()
    print("✓ test_cortex_style_capture passed")

    print("\n✓ All tests passed!")
