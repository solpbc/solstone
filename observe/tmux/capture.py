# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tmux terminal capture library.

Provides functions for capturing tmux session content, designed for use
by the Linux observer for fallback capture when screen is idle.
"""

import hashlib
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


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


class TmuxCapture:
    """Tmux terminal capture with deduplication."""

    def __init__(self):
        # Deduplication: session -> hash of last capture
        self.last_hash: dict[str, str] = {}

    def reset_hashes(self):
        """Reset deduplication hashes (call at segment boundary)."""
        self.last_hash.clear()

    def is_available(self) -> bool:
        """Check if tmux is available on this system."""
        return run_tmux_command(["list-sessions"]) is not None

    def is_active(self, poll_interval: float = 5.0) -> bool:
        """Check if any tmux sessions have recent activity.

        Args:
            poll_interval: Consider sessions active if client activity
                within this many seconds.

        Returns:
            True if any active sessions found.
        """
        return len(self.get_active_sessions(poll_interval)) > 0

    def get_active_sessions(self, poll_interval: float = 5.0) -> list[dict]:
        """Get sessions with recent client activity.

        Args:
            poll_interval: Consider sessions active if client activity
                within this many seconds.

        Returns:
            List of dicts with 'session' and 'activity' keys.
        """
        output = run_tmux_command(
            [
                "list-clients",
                "-F",
                "#{client_session} #{client_activity}",
            ]
        )
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
            if now - activity <= poll_interval and session not in seen_sessions:
                active.append({"session": session, "activity": activity})
                seen_sessions.add(session)

        return active

    def get_windows(self, session: str) -> list[WindowInfo]:
        """Get all windows for a session."""
        output = run_tmux_command(
            [
                "list-windows",
                "-t",
                session,
                "-F",
                "#{window_active} #{window_id} #{window_index} #{window_name}",
            ]
        )
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
                windows.append(
                    WindowInfo(
                        id=window_id,
                        index=int(index_str),
                        name=name,
                        active=(active_str == "1"),
                    )
                )
            except ValueError:
                continue

        return windows

    def get_panes(self, window_id: str) -> list[PaneInfo]:
        """Get all panes for a window with layout info."""
        output = run_tmux_command(
            [
                "list-panes",
                "-t",
                window_id,
                "-F",
                "#{pane_id} #{pane_index} #{pane_left} #{pane_top} #{pane_width} #{pane_height} #{pane_active}",
            ]
        )
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
                panes.append(
                    PaneInfo(
                        id=parts[0],
                        index=int(parts[1]),
                        left=int(parts[2]),
                        top=int(parts[3]),
                        width=int(parts[4]),
                        height=int(parts[5]),
                        active=(parts[6] == "1"),
                    )
                )
            except ValueError:
                continue

        return panes

    def capture_pane(self, pane_id: str) -> str:
        """Capture visible pane content with ANSI escape codes."""
        output = run_tmux_command(
            [
                "capture-pane",
                "-p",  # Print to stdout
                "-e",  # Include escape sequences (ANSI codes)
                "-t",
                pane_id,
            ]
        )
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

    def capture_changed(self, session: str) -> CaptureResult | None:
        """Capture session if content changed since last capture.

        Uses internal hash tracking for deduplication.

        Returns:
            CaptureResult if content changed, None if unchanged or error.
        """
        result = self.capture_session(session)
        if not result:
            return None

        content_hash = self.compute_hash(result)
        if self.last_hash.get(session) == content_hash:
            logger.debug(f"Session {session} unchanged, skipping")
            return None

        self.last_hash[session] = content_hash
        return result

    def result_to_dict(
        self, result: CaptureResult, capture_id: int, relative_ts: float
    ) -> dict:
        """Convert CaptureResult to JSON-serializable dict.

        Output format matches screen.jsonl structure from sol describe:
        - frame_id: Capture sequence number
        - timestamp: Seconds since segment start
        - requests: Empty list (no AI processing)
        - analysis: Category info with templated visual_description
        - tmux: All terminal-specific data

        Args:
            result: Capture result to convert.
            capture_id: Sequential frame ID for this capture.
            relative_ts: Timestamp relative to segment start.

        Returns:
            Dict ready for JSON serialization.
        """
        # Build visual description from tmux info
        pane_count = len(result.panes)
        pane_word = "pane" if pane_count == 1 else "panes"
        visual_description = (
            f"Terminal session '{result.session}' with {pane_count} {pane_word} "
            f"in window '{result.window.name}'"
        )

        return {
            "frame_id": capture_id,
            "timestamp": relative_ts,
            "requests": [],
            "analysis": {
                "visual_description": visual_description,
                "primary": "tmux",
                "secondary": "none",
                "overlap": False,
            },
            "content": {
                "tmux": {
                    "session": result.session,
                    "window": {
                        "id": result.window.id,
                        "index": result.window.index,
                        "name": result.window.name,
                    },
                    "windows": [
                        {
                            "id": w.id,
                            "index": w.index,
                            "name": w.name,
                            "active": w.active,
                        }
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
                },
            },
        }


def write_captures_jsonl(captures: list[dict], segment_dir: Path) -> list[str]:
    """Write tmux captures to JSONL files, grouped by session.

    Creates one file per session: tmux_{session}_screen.jsonl
    Frame entries match screen.jsonl format for unified formatting/indexing.
    No header line since tmux captures have no raw media file.

    Args:
        captures: List of capture dicts from result_to_dict()
        segment_dir: Directory to write files to (created if needed)

    Returns:
        List of filenames written (empty if no captures)
    """
    if not captures:
        return []

    segment_dir.mkdir(parents=True, exist_ok=True)

    # Group captures by session
    by_session: dict[str, list[dict]] = {}
    for capture in captures:
        session = capture.get("content", {}).get("tmux", {}).get("session", "unknown")
        if session not in by_session:
            by_session[session] = []
        by_session[session].append(capture)

    # Write one file per session
    files_written = []
    for session, session_captures in by_session.items():
        # Sanitize session name for filename
        safe_session = session.replace("/", "_").replace(" ", "_")
        filename = f"tmux_{safe_session}_screen.jsonl"
        output_path = segment_dir / filename

        with open(output_path, "w") as f:
            # No header - tmux captures have no raw media file
            for capture in session_captures:
                f.write(json.dumps(capture) + "\n")

        files_written.append(filename)
        logger.info(f"Wrote {len(session_captures)} tmux captures to {output_path}")

    return files_written
