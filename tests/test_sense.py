# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe.sense module."""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from observe.sense import FileSensor, HandlerProcess, HandlerQueue, QueuedItem
from think.runner import DailyLogWriter as ProcessLogWriter
from think.runner import _format_log_line

# --- QueuedItem Tests ---


def test_queued_item_basic():
    """Test QueuedItem stores file_path and queued_at."""
    path = Path("/tmp/test.flac")
    item = QueuedItem(path)

    assert item.file_path == path
    assert item.queued_at > 0
    assert item.remote is None


def test_queued_item_with_remote():
    """Test QueuedItem stores remote context."""
    path = Path("/tmp/test.flac")
    item = QueuedItem(path, remote="my-remote")

    assert item.file_path == path
    assert item.remote == "my-remote"


# --- HandlerQueue Tests ---


def test_handler_queue_can_start_empty():
    """Test can_start returns True when no process running."""
    queue = HandlerQueue("test")
    assert queue.can_start() is True


def test_handler_queue_can_start_with_current():
    """Test can_start returns False when process is running."""
    queue = HandlerQueue("test")
    queue.current_process = MagicMock()  # Simulate running process
    assert queue.can_start() is False


def test_handler_queue_enqueue():
    """Test enqueue adds items to queue."""
    queue = HandlerQueue("test")
    path1 = Path("/tmp/test1.flac")
    path2 = Path("/tmp/test2.flac")

    assert queue.enqueue(path1) is True
    assert queue.enqueue(path2) is True
    assert queue.queue_size() == 2


def test_handler_queue_enqueue_duplicate():
    """Test enqueue rejects duplicate paths."""
    queue = HandlerQueue("test")
    path = Path("/tmp/test.flac")

    assert queue.enqueue(path) is True
    assert queue.enqueue(path) is False  # Duplicate
    assert queue.queue_size() == 1


def test_handler_queue_enqueue_with_remote():
    """Test enqueue preserves remote context."""
    queue = HandlerQueue("test")
    path = Path("/tmp/test.flac")

    queue.enqueue(path, remote="my-remote")

    assert queue.queue_size() == 1
    item = queue.pop_next()
    assert item.remote == "my-remote"


def test_handler_queue_pop_next():
    """Test pop_next returns items in FIFO order."""
    queue = HandlerQueue("test")
    path1 = Path("/tmp/test1.flac")
    path2 = Path("/tmp/test2.flac")

    queue.enqueue(path1)
    queue.enqueue(path2)

    item1 = queue.pop_next()
    assert item1.file_path == path1

    item2 = queue.pop_next()
    assert item2.file_path == path2

    assert queue.pop_next() is None  # Empty


def test_handler_queue_pop_next_empty():
    """Test pop_next returns None on empty queue."""
    queue = HandlerQueue("test")
    assert queue.pop_next() is None


def test_handler_queue_set_clear_current():
    """Test set_current and clear_current."""
    queue = HandlerQueue("test")
    mock_proc = MagicMock()

    queue.set_current(mock_proc)
    assert queue.current_process is mock_proc
    assert queue.can_start() is False

    queue.clear_current()
    assert queue.current_process is None
    assert queue.can_start() is True


def test_handler_queue_queue_size():
    """Test queue_size reports correct count."""
    queue = HandlerQueue("test")
    assert queue.queue_size() == 0

    queue.enqueue(Path("/tmp/test1.flac"))
    assert queue.queue_size() == 1

    queue.enqueue(Path("/tmp/test2.flac"))
    assert queue.queue_size() == 2

    queue.pop_next()
    assert queue.queue_size() == 1


# --- Existing Tests ---


def test_format_log_line():
    """Test log line formatting."""
    line = _format_log_line("transcribe:test.flac", "stdout", "Processing...\n")
    assert "[transcribe:test.flac:stdout]" in line
    assert "Processing..." in line
    assert line.endswith("\n")


def test_process_log_writer(tmp_path, monkeypatch):
    """Test ProcessLogWriter creates and writes to log file."""
    from think import runner

    # Mock journal path and current day to use tmp_path
    monkeypatch.setattr(runner, "_get_journal_path", lambda: tmp_path)
    monkeypatch.setattr(runner, "_current_day", lambda: "20241101")

    ref = "1730476800000"
    writer = ProcessLogWriter(ref, "test")

    writer.write("line 1\n")
    writer.write("line 2\n")
    writer.close()

    # Log file uses {ref}_{name}.log format
    log_path = tmp_path / "20241101" / "health" / f"{ref}_test.log"
    assert log_path.exists()
    content = log_path.read_text()
    assert "line 1\n" in content
    assert "line 2\n" in content

    # Verify symlinks exist
    day_symlink = tmp_path / "20241101" / "health" / "test.log"
    assert day_symlink.is_symlink()
    journal_symlink = tmp_path / "health" / "test.log"
    assert journal_symlink.is_symlink()


def test_process_log_writer_thread_safe(tmp_path, monkeypatch):
    """Test ProcessLogWriter is thread-safe."""
    from think import runner

    # Mock journal path and current day to use tmp_path
    monkeypatch.setattr(runner, "_get_journal_path", lambda: tmp_path)
    monkeypatch.setattr(runner, "_current_day", lambda: "20241101")

    ref = "1730476800000"
    writer = ProcessLogWriter(ref, "test")

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

    # Log file uses {ref}_{name}.log format
    log_path = tmp_path / "20241101" / "health" / f"{ref}_test.log"
    lines = log_path.read_text().split("\n")
    # Should have 50 lines (5 threads * 10 lines each)
    assert len([line for line in lines if line]) == 50


def test_handler_process_cleanup():
    """Test HandlerProcess cleanup joins threads and closes logger."""
    mock_managed = MagicMock()
    mock_managed.name = "transcribe"
    mock_managed.process = MagicMock()

    handler = HandlerProcess(Path("/tmp/test.flac"), mock_managed, "transcribe")

    handler.cleanup()

    mock_managed.cleanup.assert_called_once()


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
    """Test pattern matching logic.

    Files are expected to be in segment directories: journal/YYYYMMDD/HHMMSS_LEN/file.ext
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create journal/day/segment structure
        journal_dir = Path(tmpdir)
        day_dir = journal_dir / "20250101"
        segment_dir = day_dir / "123456_300"
        segment_dir.mkdir(parents=True)

        sensor = FileSensor(journal_dir)
        sensor.register("*.webm", "describe", ["echo", "{file}"])
        sensor.register("*.flac", "transcribe", ["cat", "{file}"])

        # Should match - files in segment directory
        webm_file = segment_dir / "center_DP-3_screen.webm"
        assert sensor._match_pattern(webm_file) is not None
        assert sensor._match_pattern(webm_file)[0] == "describe"

        flac_file = segment_dir / "audio.flac"
        assert sensor._match_pattern(flac_file) is not None
        assert sensor._match_pattern(flac_file)[0] == "transcribe"

        # Should not match - wrong extension
        txt_file = segment_dir / "test.txt"
        assert sensor._match_pattern(txt_file) is None

        # Should not match - file in day root (not in segment dir)
        day_root_file = day_dir / "orphan.webm"
        assert sensor._match_pattern(day_root_file) is None

        # Should not match - jsonl output file
        jsonl_file = segment_dir / "audio.jsonl"
        assert sensor._match_pattern(jsonl_file) is None


@patch("think.runner._get_journal_path")
@patch("think.runner._current_day")
@patch("think.runner.subprocess.Popen")
def test_file_sensor_spawn_handler(mock_popen, mock_day, mock_journal, tmp_path):
    """Test spawning handler process."""
    # Mock runner functions to use tmp_path
    mock_journal.return_value = tmp_path
    mock_day.return_value = "20241101"

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

    # Verify log file was created with {ref}_echo.log format
    health_dir = tmp_path / "20241101" / "health"
    log_files = list(health_dir.glob("*_echo.log"))
    assert len(log_files) == 1, f"Expected 1 echo log file, found {len(log_files)}"


def test_file_sensor_spawn_handler_duplicate(tmp_path, mock_callosum):
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


@patch("think.runner._get_journal_path")
@patch("think.runner._current_day")
def test_file_sensor_spawn_handler_real_process(
    mock_day, mock_journal, tmp_path, mock_callosum
):
    """Test spawning a real process and monitoring completion."""
    # Mock runner functions to use tmp_path
    mock_journal.return_value = tmp_path
    mock_day.return_value = "20241101"

    sensor = FileSensor(tmp_path)

    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    # Spawn a simple echo command
    sensor._spawn_handler(test_file, "echo", ["echo", "hello"])

    # Wait for process to complete
    time.sleep(0.5)

    # Process should have completed and been removed from running dict
    assert test_file not in sensor.running

    # Check log file contains output with {ref}_echo.log format
    health_dir = tmp_path / "20241101" / "health"
    log_files = list(health_dir.glob("*_echo.log"))
    assert len(log_files) == 1, f"Expected 1 echo log file, found {len(log_files)}"

    log_content = log_files[0].read_text()
    assert "hello" in log_content
    # New format is [command_name:stream]
    assert "[echo:stdout]" in log_content


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
        # Create journal/day/segment structure
        day_dir = tmp_path / "20250101"
        segment_dir = day_dir / "143022_300"
        segment_dir.mkdir(parents=True)

        sensor = FileSensor(tmp_path)
        sensor.register("*.webm", "describe", ["echo", "{file}"])

        test_file = segment_dir / "center_DP-3_screen.webm"
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

        # Mock callosum
        sensor.callosum = MagicMock()

        sensor.stop()

        assert sensor.running_flag is False
        sensor.callosum.stop.assert_called_once()


def test_file_sensor_handle_callosum_message(tmp_path):
    """Test handling of observe.observing Callosum events."""
    with patch.object(FileSensor, "_handle_file") as mock_handle:
        # Create journal/day/segment structure
        day_dir = tmp_path / "20250101"
        segment_dir = day_dir / "143022_300"
        segment_dir.mkdir(parents=True)

        sensor = FileSensor(tmp_path)
        sensor.register("*.flac", "transcribe", ["echo", "{file}"])
        sensor.register("*.webm", "describe", ["echo", "{file}"])

        # Create test files with simple names in segment directory
        audio_file = segment_dir / "audio.flac"
        audio_file.write_text("audio content")
        video_file = segment_dir / "center_DP-3_screen.webm"
        video_file.write_text("video content")

        # Simulate observing event with simple filenames
        message = {
            "tract": "observe",
            "event": "observing",
            "day": "20250101",
            "segment": "143022_300",
            "files": ["audio.flac", "center_DP-3_screen.webm"],
        }

        sensor._handle_callosum_message(message)

        # Should have called _handle_file for each file
        assert mock_handle.call_count == 2
        called_paths = [call[0][0] for call in mock_handle.call_args_list]
        assert audio_file in called_paths
        assert video_file in called_paths

        # Should have pre-registered segment tracking
        assert "143022_300" in sensor.segment_files
        assert audio_file in sensor.segment_files["143022_300"]
        assert video_file in sensor.segment_files["143022_300"]
        assert "143022_300" in sensor.segment_start_time
        assert sensor.segment_day["143022_300"] == "20250101"


def test_file_sensor_handle_callosum_message_ignores_other_events(tmp_path):
    """Test that non-observing events are ignored."""
    with patch.object(FileSensor, "_handle_file") as mock_handle:
        sensor = FileSensor(tmp_path)

        # Simulate a different event type
        message = {
            "tract": "observe",
            "event": "status",
            "some_data": "value",
        }

        sensor._handle_callosum_message(message)

        # Should not call _handle_file
        mock_handle.assert_not_called()


def test_file_sensor_handle_callosum_message_invalid_event(tmp_path):
    """Test that invalid observing events are handled gracefully."""
    with patch.object(FileSensor, "_handle_file") as mock_handle:
        sensor = FileSensor(tmp_path)

        # Simulate event missing required fields
        message = {
            "tract": "observe",
            "event": "observing",
            "segment": "143022_300",
            # missing 'day' and 'files'
        }

        sensor._handle_callosum_message(message)

        # Should not call _handle_file
        mock_handle.assert_not_called()


def test_file_sensor_segment_observed_includes_day(tmp_path, mock_callosum):
    """Test that observe.observed event includes day field."""
    from think.callosum import CallosumConnection

    # Create journal/day/segment structure
    day_dir = tmp_path / "20250101"
    segment_dir = day_dir / "143022_300"
    segment_dir.mkdir(parents=True)

    sensor = FileSensor(tmp_path)
    sensor.register("*.flac", "transcribe", ["echo", "{file}"])

    # Set up callosum on sensor to capture emitted events
    emitted_events = []
    sensor.callosum = CallosumConnection()
    sensor.callosum.start(callback=lambda msg: emitted_events.append(msg))

    # Create test file with simple name in segment directory
    audio_file = segment_dir / "audio.flac"
    audio_file.write_text("audio content")

    # Simulate observing event to set up segment tracking (simple filenames)
    message = {
        "tract": "observe",
        "event": "observing",
        "day": "20250101",
        "segment": "143022_300",
        "files": ["audio.flac"],
    }
    sensor._handle_callosum_message(message)

    # Wait for handler to complete
    time.sleep(0.5)

    # Check that segment_day was cleaned up (handler completed)
    assert "143022_300" not in sensor.segment_day

    # Check observe.observed event was emitted with day field
    observed_events = [
        e
        for e in emitted_events
        if e.get("tract") == "observe" and e.get("event") == "observed"
    ]
    assert len(observed_events) == 1
    assert observed_events[0].get("day") == "20250101"
    assert observed_events[0].get("segment") == "143022_300"


def test_file_sensor_segment_observed_no_handlers(tmp_path, mock_callosum):
    """Test that observe.observed is emitted immediately for segments with no matching handlers.

    This covers the case of tmux-only segments where files like .jsonl don't match
    any registered patterns (*.flac, *.webm, etc.).
    """
    from think.callosum import CallosumConnection

    # Create journal/day/segment structure
    day_dir = tmp_path / "20250101"
    segment_dir = day_dir / "143022_300"
    segment_dir.mkdir(parents=True)

    sensor = FileSensor(tmp_path)
    # Only register handlers for audio/video (not .jsonl)
    sensor.register("*.flac", "transcribe", ["echo", "{file}"])
    sensor.register("*.webm", "describe", ["echo", "{file}"])

    # Set up callosum on sensor to capture emitted events
    emitted_events = []
    sensor.callosum = CallosumConnection()
    sensor.callosum.start(callback=lambda msg: emitted_events.append(msg))

    # Create test file that doesn't match any pattern (like tmux captures)
    jsonl_file = segment_dir / "tmux_0_screen.jsonl"
    jsonl_file.write_text('{"content": "terminal output"}')

    # Simulate observing event with only .jsonl file
    message = {
        "tract": "observe",
        "event": "observing",
        "day": "20250101",
        "segment": "143022_300",
        "files": ["tmux_0_screen.jsonl"],
    }
    sensor._handle_callosum_message(message)

    # Segment tracking should be cleaned up immediately (no handlers to wait for)
    assert "143022_300" not in sensor.segment_files
    assert "143022_300" not in sensor.segment_day

    # Check observe.observed event was emitted immediately
    observed_events = [
        e
        for e in emitted_events
        if e.get("tract") == "observe" and e.get("event") == "observed"
    ]
    assert len(observed_events) == 1
    assert observed_events[0].get("day") == "20250101"
    assert observed_events[0].get("segment") == "143022_300"


def test_delete_outputs_screen(tmp_path):
    """Test delete_outputs with screen type."""
    from observe.sense import delete_outputs

    # Create journal/day/segment structure
    day_dir = tmp_path / "20250101"
    segment_dir = day_dir / "143022_300"
    segment_dir.mkdir(parents=True)

    # Create source files and outputs
    (segment_dir / "center_DP-3_screen.webm").write_text("video")
    (segment_dir / "center_DP-3_screen.jsonl").write_text('{"raw": "test"}')
    (segment_dir / "audio.flac").write_text("audio")
    (segment_dir / "audio.jsonl").write_text('{"raw": "test"}')

    # Delete screen outputs
    deleted = delete_outputs(day_dir, "screen")

    assert len(deleted) == 1
    assert deleted[0].name == "center_DP-3_screen.jsonl"
    assert not (segment_dir / "center_DP-3_screen.jsonl").exists()
    assert (segment_dir / "audio.jsonl").exists()  # Audio untouched


def test_delete_outputs_audio(tmp_path):
    """Test delete_outputs with audio type."""
    from observe.sense import delete_outputs

    # Create journal/day/segment structure
    day_dir = tmp_path / "20250101"
    segment_dir = day_dir / "143022_300"
    segment_dir.mkdir(parents=True)

    # Create source files and outputs
    (segment_dir / "center_DP-3_screen.webm").write_text("video")
    (segment_dir / "center_DP-3_screen.jsonl").write_text('{"raw": "test"}')
    (segment_dir / "audio.flac").write_text("audio")
    (segment_dir / "audio.jsonl").write_text('{"raw": "test"}')

    # Delete audio outputs
    deleted = delete_outputs(day_dir, "audio")

    assert len(deleted) == 1
    assert deleted[0].name == "audio.jsonl"
    assert not (segment_dir / "audio.jsonl").exists()
    assert (segment_dir / "center_DP-3_screen.jsonl").exists()  # Screen untouched


def test_delete_outputs_dry_run(tmp_path):
    """Test delete_outputs with dry_run=True."""
    from observe.sense import delete_outputs

    # Create journal/day/segment structure
    day_dir = tmp_path / "20250101"
    segment_dir = day_dir / "143022_300"
    segment_dir.mkdir(parents=True)

    # Create source files and outputs
    (segment_dir / "screen.webm").write_text("video")
    (segment_dir / "screen.jsonl").write_text('{"raw": "test"}')

    # Dry run should return files but not delete
    deleted = delete_outputs(day_dir, "screen", dry_run=True)

    assert len(deleted) == 1
    assert (segment_dir / "screen.jsonl").exists()  # Still exists


def test_delete_outputs_segment_filter(tmp_path):
    """Test delete_outputs with segment filter."""
    from observe.sense import delete_outputs

    # Create journal/day/segments structure
    day_dir = tmp_path / "20250101"
    segment1 = day_dir / "143022_300"
    segment2 = day_dir / "150022_300"
    segment1.mkdir(parents=True)
    segment2.mkdir(parents=True)

    # Create outputs in both segments
    (segment1 / "screen.webm").write_text("video")
    (segment1 / "screen.jsonl").write_text('{"raw": "test"}')
    (segment2 / "screen.webm").write_text("video")
    (segment2 / "screen.jsonl").write_text('{"raw": "test"}')

    # Delete only from segment1
    deleted = delete_outputs(day_dir, "screen", segment_filter="143022_300")

    assert len(deleted) == 1
    assert not (segment1 / "screen.jsonl").exists()
    assert (segment2 / "screen.jsonl").exists()  # Other segment untouched
