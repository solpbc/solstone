#!/usr/bin/env python3
"""
Tmux terminal observer for journaling.

Polls tmux for active sessions and captures terminal content from the active
window's panes. Creates 5-minute segments with JSONL output.
"""

import argparse
import hashlib
import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass

from think.callosum import CallosumConnection
from think.utils import day_path, setup_cli

logger = logging.getLogger(__name__)

# Constants
DEFAULT_INTERVAL = 300  # 5 minutes
DEFAULT_POLL_INTERVAL = 5  # seconds


@dataclass
class PaneInfo:
    """Information about a tmux pane."""

    id: str
    index: int
    left: int
    top: int
    width: int
    height: int
    active: bool
    content: str = ""


@dataclass
class WindowInfo:
    """Information about a tmux window."""

    id: str
    index: int
    name: str
    active: bool


@dataclass
class CaptureResult:
    """Result of capturing a session's active window."""

    session: str
    window: WindowInfo
    windows: list[WindowInfo]
    panes: list[PaneInfo]


def run_tmux_command(args: list[str]) -> str | None:
    """Run a tmux command and return stdout, or None on error."""
    try:
        result = subprocess.run(
            ["tmux"] + args,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug(f"tmux command failed: {e}")
        return None


class TmuxObserver:
    """Observer for tmux terminal activity."""

    def __init__(self, interval: int = DEFAULT_INTERVAL, poll_interval: int = DEFAULT_POLL_INTERVAL):
        self.interval = interval
        self.poll_interval = poll_interval
        self.running = True

        # Segment state
        self.start_at = time.time()
        self.captures: list[dict] = []
        self.capture_id = 0
        self.sessions_seen: set[str] = set()

        # Deduplication: session -> hash of last capture
        self.last_hash: dict[str, str] = {}

        # Callosum for status events
        self.callosum: CallosumConnection | None = None

    def get_active_sessions(self) -> list[dict]:
        """Get sessions with recent client activity.

        Returns list of dicts with 'session' and 'activity' keys.
        """
        output = run_tmux_command([
            "list-clients",
            "-F", "#{client_session} #{client_activity}",
        ])
        if not output:
            return []

        now = time.time()
        active = []
        seen_sessions = set()

        for line in output.strip().split("\n"):
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue

            session, activity_str = parts
            try:
                activity = int(activity_str)
            except ValueError:
                continue

            # Check if active within poll interval
            if now - activity <= self.poll_interval and session not in seen_sessions:
                active.append({"session": session, "activity": activity})
                seen_sessions.add(session)

        return active

    def get_windows(self, session: str) -> list[WindowInfo]:
        """Get all windows for a session."""
        output = run_tmux_command([
            "list-windows",
            "-t", session,
            "-F", "#{window_active} #{window_id} #{window_index} #{window_name}",
        ])
        if not output:
            return []

        windows = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            parts = line.split(" ", 3)
            if len(parts) < 4:
                continue

            active_str, window_id, index_str, name = parts
            try:
                windows.append(WindowInfo(
                    id=window_id,
                    index=int(index_str),
                    name=name,
                    active=(active_str == "1"),
                ))
            except ValueError:
                continue

        return windows

    def get_panes(self, window_id: str) -> list[PaneInfo]:
        """Get all panes for a window with layout info."""
        output = run_tmux_command([
            "list-panes",
            "-t", window_id,
            "-F", "#{pane_id} #{pane_index} #{pane_left} #{pane_top} #{pane_width} #{pane_height} #{pane_active}",
        ])
        if not output:
            return []

        panes = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) != 7:
                continue

            try:
                panes.append(PaneInfo(
                    id=parts[0],
                    index=int(parts[1]),
                    left=int(parts[2]),
                    top=int(parts[3]),
                    width=int(parts[4]),
                    height=int(parts[5]),
                    active=(parts[6] == "1"),
                ))
            except ValueError:
                continue

        return panes

    def capture_pane(self, pane_id: str) -> str:
        """Capture visible pane content with ANSI escape codes."""
        output = run_tmux_command([
            "capture-pane",
            "-p",  # Print to stdout
            "-e",  # Include escape sequences (ANSI codes)
            "-t", pane_id,
        ])
        return output if output else ""

    def capture_session(self, session: str) -> CaptureResult | None:
        """Capture the active window of a session with all its panes.

        Returns None if session doesn't exist or has no active window.
        """
        windows = self.get_windows(session)
        if not windows:
            return None

        # Find active window
        active_window = next((w for w in windows if w.active), None)
        if not active_window:
            return None

        # Get panes for active window
        panes = self.get_panes(active_window.id)
        if not panes:
            return None

        # Capture content for each pane
        for pane in panes:
            pane.content = self.capture_pane(pane.id)

        return CaptureResult(
            session=session,
            window=active_window,
            windows=windows,
            panes=panes,
        )

    def compute_hash(self, result: CaptureResult) -> str:
        """Compute hash of capture for deduplication."""
        # Hash window id + all pane contents
        parts = [result.window.id]
        for pane in sorted(result.panes, key=lambda p: p.id):
            parts.append(pane.content)
        content = "\n".join(parts)
        return hashlib.md5(content.encode()).hexdigest()

    def result_to_dict(self, result: CaptureResult, ts: float) -> dict:
        """Convert CaptureResult to JSON-serializable dict."""
        self.capture_id += 1
        return {
            "id": self.capture_id,
            "ts": ts,
            "session": result.session,
            "window": {
                "id": result.window.id,
                "index": result.window.index,
                "name": result.window.name,
            },
            "windows": [
                {"id": w.id, "index": w.index, "name": w.name, "active": w.active}
                for w in result.windows
            ],
            "panes": [
                {
                    "id": p.id,
                    "index": p.index,
                    "left": p.left,
                    "top": p.top,
                    "width": p.width,
                    "height": p.height,
                    "active": p.active,
                    "content": p.content,
                }
                for p in result.panes
            ],
        }

    def poll_and_capture(self) -> list[dict]:
        """Poll for active sessions and capture changed content.

        Returns list of capture dicts for sessions that changed.
        """
        active_sessions = self.get_active_sessions()
        if not active_sessions:
            return []

        ts = time.time()
        new_captures = []

        for session_info in active_sessions:
            session = session_info["session"]
            self.sessions_seen.add(session)

            result = self.capture_session(session)
            if not result:
                continue

            # Check if content changed
            content_hash = self.compute_hash(result)
            if self.last_hash.get(session) == content_hash:
                logger.debug(f"Session {session} unchanged, skipping")
                continue

            self.last_hash[session] = content_hash
            capture_dict = self.result_to_dict(result, ts)
            new_captures.append(capture_dict)
            logger.debug(f"Captured session {session}: {len(result.panes)} panes")

        return new_captures

    def handle_boundary(self):
        """Write accumulated captures to segment JSONL and reset."""
        if not self.captures:
            logger.info("No captures in segment, skipping write")
            self.reset_segment()
            return

        # Build segment path
        start_dt = time.localtime(self.start_at)
        date_part = time.strftime("%Y%m%d", start_dt)
        time_part = time.strftime("%H%M%S", start_dt)
        duration = int(time.time() - self.start_at)
        segment_key = f"{time_part}_{duration}"

        segment_dir = day_path(date_part) / segment_key
        segment_dir.mkdir(parents=True, exist_ok=True)

        output_path = segment_dir / "tmux.jsonl"

        # Write JSONL: metadata header + captures
        with open(output_path, "w") as f:
            # Header with metadata
            header = {
                "captures": len(self.captures),
                "sessions": sorted(self.sessions_seen),
            }
            f.write(json.dumps(header) + "\n")

            # Write each capture
            for capture in self.captures:
                f.write(json.dumps(capture) + "\n")

        logger.info(
            f"Wrote {len(self.captures)} captures to {output_path}"
        )

        # Emit event
        if self.callosum:
            self.callosum.emit(
                "observe",
                "tmux_captured",
                segment=segment_key,
                captures=len(self.captures),
                sessions=sorted(self.sessions_seen),
            )

        self.reset_segment()

    def reset_segment(self):
        """Reset state for new segment."""
        self.start_at = time.time()
        self.captures = []
        self.capture_id = 0
        self.sessions_seen = set()
        # Keep last_hash for cross-segment deduplication

    def emit_status(self):
        """Emit observe.status event for health monitoring."""
        if not self.callosum:
            return

        elapsed = int(time.time() - self.start_at)
        self.callosum.emit(
            "observe",
            "status",
            tmux={
                "captures": len(self.captures),
                "sessions": sorted(self.sessions_seen),
                "window_elapsed_seconds": elapsed,
            },
        )

    def main_loop(self):
        """Run the main observer loop."""
        logger.info(
            f"Starting tmux observer (interval={self.interval}s, poll={self.poll_interval}s)"
        )

        # Start Callosum connection
        self.callosum = CallosumConnection()
        self.callosum.start()

        last_status_emit = 0.0

        while self.running:
            # Poll and capture
            new_captures = self.poll_and_capture()
            self.captures.extend(new_captures)

            # Check for segment boundary
            elapsed = time.time() - self.start_at
            if elapsed >= self.interval:
                logger.info(f"Segment boundary at {elapsed:.1f}s")
                self.handle_boundary()

            # Emit status every 5 seconds
            now = time.time()
            if now - last_status_emit >= 5:
                self.emit_status()
                last_status_emit = now

            # Sleep until next poll
            time.sleep(self.poll_interval)

        # Cleanup
        self.shutdown()

    def shutdown(self):
        """Clean shutdown - write any pending captures."""
        logger.info("Shutting down tmux observer...")

        if self.captures:
            logger.info(f"Writing {len(self.captures)} pending captures")
            self.handle_boundary()

        if self.callosum:
            self.callosum.stop()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Tmux terminal observer for journaling."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Segment duration in seconds (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--poll",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Poll interval in seconds (default: {DEFAULT_POLL_INTERVAL})",
    )
    args = setup_cli(parser)

    # Verify journal path exists
    journal = os.getenv("JOURNAL_PATH")
    if not journal or not os.path.exists(journal):
        logger.error(f"JOURNAL_PATH not set or does not exist: {journal}")
        return 1

    # Check tmux is available
    if run_tmux_command(["list-sessions"]) is None:
        logger.warning("tmux not available or no sessions - will poll for server")

    observer = TmuxObserver(interval=args.interval, poll_interval=args.poll)

    # Signal handlers
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        observer.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        observer.main_loop()
    except Exception as e:
        logger.error(f"Observer error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
