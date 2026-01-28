# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe/macos/screencapture.py."""

from unittest.mock import MagicMock

from observe.macos.screencapture import (
    AudioInfo,
    DisplayInfo,
    ScreenCaptureKitManager,
)


class TestAudioInfo:
    """Test AudioInfo dataclass."""

    def test_create_with_full_tracks(self):
        """AudioInfo stores full track dicts."""
        tracks = [
            {
                "name": "system",
                "deviceName": "MacBook Pro Speakers",
                "deviceUID": "BuiltInSpeakerDevice",
                "manufacturer": "Apple Inc.",
                "transportType": "built-in",
            },
            {
                "name": "microphone",
                "deviceName": "MacBook Pro Microphone",
                "deviceUID": "BuiltInMicrophoneDevice",
                "manufacturer": "Apple Inc.",
                "transportType": "built-in",
            },
        ]
        audio = AudioInfo(
            file_path="/tmp/test.m4a",
            sample_rate=48000,
            channels=1,
            tracks=tracks,
        )

        assert audio.file_path == "/tmp/test.m4a"
        assert audio.sample_rate == 48000
        assert audio.channels == 1
        assert len(audio.tracks) == 2
        assert audio.tracks[0]["deviceName"] == "MacBook Pro Speakers"
        assert audio.tracks[1]["name"] == "microphone"


class TestDisplayInfo:
    """Test DisplayInfo dataclass."""

    def test_create(self):
        """DisplayInfo stores display metadata."""
        display = DisplayInfo(
            display_id=1,
            position="center",
            x=0,
            y=0,
            width=1920,
            height=1080,
            file_path="/tmp/test.mov",
        )

        assert display.display_id == 1
        assert display.position == "center"
        assert display.width == 1920
        assert display.height == 1080


class TestScreenCaptureKitManagerStopEvent:
    """Test stop event capture in ScreenCaptureKitManager."""

    def test_stop_event_initial_state(self):
        """Manager starts with no stop event."""
        manager = ScreenCaptureKitManager()
        assert manager._stop_event is None
        assert not manager._stop_received.is_set()

    def test_get_stop_event_timeout(self):
        """get_stop_event returns None on timeout."""
        manager = ScreenCaptureKitManager()
        result = manager.get_stop_event(timeout=0.01)
        assert result is None

    def test_get_stop_event_returns_captured(self):
        """get_stop_event returns captured stop event."""
        manager = ScreenCaptureKitManager()

        # Simulate stop event being captured
        stop_data = {"type": "stop", "reason": "completed"}
        manager._stop_event = stop_data
        manager._stop_received.set()

        result = manager.get_stop_event(timeout=0.1)
        assert result == stop_data
        assert result["reason"] == "completed"

    def test_stop_event_with_error(self):
        """Stop event with error details is captured."""
        manager = ScreenCaptureKitManager()

        stop_data = {
            "type": "stop",
            "reason": "error",
            "errorCode": -1234,
            "errorDomain": "com.apple.ScreenCaptureKit",
        }
        manager._stop_event = stop_data
        manager._stop_received.set()

        result = manager.get_stop_event(timeout=0.1)
        assert result["reason"] == "error"
        assert result["errorCode"] == -1234
        assert result["errorDomain"] == "com.apple.ScreenCaptureKit"

    def test_stop_event_device_change(self):
        """Stop event with device change flags is captured."""
        manager = ScreenCaptureKitManager()

        stop_data = {
            "type": "stop",
            "reason": "device-change",
            "inputDeviceChanged": True,
            "outputDeviceChanged": False,
        }
        manager._stop_event = stop_data
        manager._stop_received.set()

        result = manager.get_stop_event(timeout=0.1)
        assert result["reason"] == "device-change"
        assert result["inputDeviceChanged"] is True
        assert result["outputDeviceChanged"] is False


class TestStreamStdoutParsing:
    """Test _stream_stdout JSON parsing for stop events."""

    def test_parse_stop_event_from_stdout(self):
        """_stream_stdout parses stop events from JSONL output."""
        manager = ScreenCaptureKitManager()

        # Create mock process with stdout that yields stop event
        mock_stdout = MagicMock()
        mock_stdout.__iter__ = MagicMock(
            return_value=iter(['{"type": "stop", "reason": "completed"}\n'])
        )

        manager.process = MagicMock()
        manager.process.stdout = mock_stdout

        # Run the stream function directly
        manager._stream_stdout()

        # Verify stop event was captured
        assert manager._stop_event is not None
        assert manager._stop_event["type"] == "stop"
        assert manager._stop_event["reason"] == "completed"
        assert manager._stop_received.is_set()

    def test_parse_non_json_lines(self):
        """_stream_stdout handles non-JSON log lines gracefully."""
        manager = ScreenCaptureKitManager()

        mock_stdout = MagicMock()
        mock_stdout.__iter__ = MagicMock(
            return_value=iter(
                [
                    "Some log message\n",
                    "Another message\n",
                    '{"type": "stop", "reason": "signal"}\n',
                ]
            )
        )

        manager.process = MagicMock()
        manager.process.stdout = mock_stdout

        manager._stream_stdout()

        # Only stop event should be captured
        assert manager._stop_event is not None
        assert manager._stop_event["reason"] == "signal"

    def test_parse_empty_lines(self):
        """_stream_stdout handles empty lines."""
        manager = ScreenCaptureKitManager()

        mock_stdout = MagicMock()
        mock_stdout.__iter__ = MagicMock(
            return_value=iter(
                [
                    "\n",
                    "   \n",
                    '{"type": "stop", "reason": "completed"}\n',
                ]
            )
        )

        manager.process = MagicMock()
        manager.process.stdout = mock_stdout

        manager._stream_stdout()

        assert manager._stop_event is not None
        assert manager._stop_event["reason"] == "completed"
