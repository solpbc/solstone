"""Tests for observe.sense module."""

import subprocess
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from observe.sense import (
    FileSensor,
    HandlerProcess,
    ProcessLogWriter,
    _format_log_line,
)


def test_format_log_line():
    """Test log line formatting."""
    line = _format_log_line("transcribe", "test.flac", "stdout", "Processing...\n")
    assert "[transcribe:test.flac:stdout]" in line
    assert "Processing..." in line
    assert line.endswith("\n")


def test_process_log_writer(tmp_path):
    """Test ProcessLogWriter creates and writes to log file."""
    log_path = tmp_path / "test.log"
    writer = ProcessLogWriter(log_path)

    writer.write("line 1\n")
    writer.write("line 2\n")
    writer.close()

    assert log_path.exists()
    content = log_path.read_text()
    assert "line 1\n" in content
    assert "line 2\n" in content


def test_process_log_writer_thread_safe(tmp_path):
    """Test ProcessLogWriter is thread-safe."""
    log_path = tmp_path / "test.log"
    writer = ProcessLogWriter(log_path)

    def write_lines(prefix):
        for i in range(10):
            writer.write(f"{prefix}-{i}\n")

    threads = [
        threading.Thread(target=write_lines, args=(f"thread{i}",)) for i in range(5)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    writer.close()

    lines = log_path.read_text().split("\n")
    # Should have 50 lines (5 threads * 10 lines each)
    assert len([line for line in lines if line]) == 50


def test_handler_process_cleanup():
    """Test HandlerProcess cleanup joins threads and closes logger."""
    mock_proc = MagicMock()
    mock_logger = MagicMock()
    mock_thread = MagicMock()

    handler = HandlerProcess(
        Path("/tmp/test.flac"), mock_proc, "transcribe", mock_logger
    )
    handler.threads = [mock_thread]

    handler.cleanup()

    mock_thread.join.assert_called_once_with(timeout=1)
    mock_logger.close.assert_called_once()


def test_file_sensor_register():
    """Test registering handlers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sensor = FileSensor(Path(tmpdir))

        sensor.register("*.webm", "describe", ["echo", "{file}"])
        sensor.register("*.flac", "transcribe", ["cat", "{file}"])

        assert "*.webm" in sensor.handlers
        assert "*.flac" in sensor.handlers
        assert sensor.handlers["*.webm"][0] == "describe"
        assert sensor.handlers["*.flac"][0] == "transcribe"


def test_file_sensor_match_pattern():
    """Test pattern matching logic."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create journal/day structure
        journal_dir = Path(tmpdir)
        day_dir = journal_dir / "20250101"
        day_dir.mkdir()

        sensor = FileSensor(journal_dir)
        sensor.register("*.webm", "describe", ["echo", "{file}"])
        sensor.register("*_raw.flac", "transcribe", ["cat", "{file}"])

        # Should match - files in day directory
        webm_file = day_dir / "test.webm"
        assert sensor._match_pattern(webm_file) is not None
        assert sensor._match_pattern(webm_file)[0] == "describe"

        flac_file = day_dir / "123456_raw.flac"
        assert sensor._match_pattern(flac_file) is not None
        assert sensor._match_pattern(flac_file)[0] == "transcribe"

        # Should not match - wrong extension
        txt_file = day_dir / "test.txt"
        assert sensor._match_pattern(txt_file) is None

        # Should not match - in subdirectory
        heard_dir = day_dir / "heard"
        heard_dir.mkdir()
        heard_file = heard_dir / "test.flac"
        assert sensor._match_pattern(heard_file) is None


@patch("observe.sense.subprocess.Popen")
def test_file_sensor_spawn_handler(mock_popen, tmp_path):
    """Test spawning handler process."""
    # Setup mock process
    mock_proc = MagicMock()
    mock_proc.stdout = None
    mock_proc.stderr = None
    mock_popen.return_value = mock_proc

    sensor = FileSensor(tmp_path)
    sensor.register("*.txt", "test", ["echo", "{file}"])

    test_file = tmp_path / "test.txt"
    test_file.write_text("content")

    sensor._spawn_handler(test_file, "test", ["echo", "{file}"])

    # Verify subprocess was spawned with correct command
    mock_popen.assert_called_once()
    args = mock_popen.call_args[0][0]
    assert args == ["echo", str(test_file)]

    # Verify log file was created
    log_file = tmp_path / "health" / "sense_test.log"
    assert log_file.exists()


def test_file_sensor_spawn_handler_duplicate(tmp_path):
    """Test that duplicate file processing is prevented."""
    # Create journal/day structure
    day_dir = tmp_path / "20250101"
    day_dir.mkdir()

    sensor = FileSensor(tmp_path)
    sensor.register("*.txt", "test", ["echo", "hello"])

    test_file = day_dir / "test.txt"
    test_file.write_text("content")

    # Spawn first time (real process)
    sensor._spawn_handler(test_file, "test", ["echo", "hello"])

    # File should now be in running dict
    assert test_file in sensor.running

    # Try to spawn again - should be skipped (file still in running dict)
    # We can check this by verifying the lock prevents it
    with patch("observe.sense.subprocess.Popen") as mock_popen:
        sensor._spawn_handler(test_file, "test", ["echo", "hello"])
        # Should not have called Popen because file already in running
        mock_popen.assert_not_called()


def test_file_sensor_spawn_handler_real_process(tmp_path):
    """Test spawning a real process and monitoring completion."""
    sensor = FileSensor(tmp_path)

    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    # Spawn a simple echo command
    sensor._spawn_handler(test_file, "echo", ["echo", "hello"])

    # Wait for process to complete
    time.sleep(0.5)

    # Process should have completed and been removed from running dict
    assert test_file not in sensor.running

    # Check log file contains output
    log_file = tmp_path / "health" / "sense_echo.log"
    assert log_file.exists()
    log_content = log_file.read_text()
    assert "hello" in log_content
    assert "[echo:test.txt:stdout]" in log_content


def test_file_sensor_spawn_handler_failing_process(tmp_path):
    """Test handling of failing process."""
    sensor = FileSensor(tmp_path)

    test_file = tmp_path / "test.txt"
    test_file.write_text("content")

    # Spawn a command that will fail
    sensor._spawn_handler(test_file, "fail", ["false"])

    # Wait for process to complete
    time.sleep(0.5)

    # Process should have completed and been removed
    assert test_file not in sensor.running


def test_file_sensor_handle_file(tmp_path):
    """Test file handling dispatches to correct handler."""
    with patch.object(FileSensor, "_spawn_handler") as mock_spawn:
        # Create journal/day structure
        day_dir = tmp_path / "20250101"
        day_dir.mkdir()

        sensor = FileSensor(tmp_path)
        sensor.register("*.webm", "describe", ["echo", "{file}"])

        test_file = day_dir / "test.webm"
        test_file.write_text("content")

        sensor._handle_file(test_file)

        # Should have called spawn with correct handler
        mock_spawn.assert_called_once()
        call_args = mock_spawn.call_args[0]
        assert call_args[0] == test_file
        assert call_args[1] == "describe"


def test_file_sensor_handle_nonexistent_file(tmp_path):
    """Test handling of nonexistent file is graceful."""
    with patch.object(FileSensor, "_spawn_handler") as mock_spawn:
        sensor = FileSensor(tmp_path)
        sensor.register("*.txt", "test", ["echo", "{file}"])

        nonexistent = tmp_path / "nonexistent.txt"
        sensor._handle_file(nonexistent)

        # Should not spawn handler for nonexistent file
        mock_spawn.assert_not_called()


def test_file_sensor_stop():
    """Test stopping the sensor."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sensor = FileSensor(Path(tmpdir))

        # Mock observer
        sensor.observer = MagicMock()

        sensor.stop()

        assert sensor.running_flag is False
        sensor.observer.stop.assert_called_once()
        sensor.observer.join.assert_called_once()
