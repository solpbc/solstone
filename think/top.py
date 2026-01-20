# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""solstone activity manager - interactive TUI for service monitoring.

Connects to the Callosum message bus to display real-time service status
and provides keyboard controls for restarting services.
"""

import argparse
import asyncio
import logging
import queue
import time
from datetime import datetime, timedelta

import psutil
from blessed import Terminal
from desktop_notifier import DesktopNotifier, Urgency

from think.callosum import CallosumConnection
from think.utils import setup_cli

# Desktop notification system
_notifier: DesktopNotifier | None = None


def _get_notifier() -> DesktopNotifier:
    """Get or create the global desktop notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = DesktopNotifier(app_name="solstone activity manager")
    return _notifier


class ServiceManager:
    """Interactive TUI for managing solstone services."""

    def __init__(self):
        self.services = []  # From supervisor/status events
        self.crashed = []  # From supervisor/status crashed field
        self.tasks = []  # From supervisor/status tasks field
        self.selected = 0
        self.callosum = CallosumConnection()
        self.running = True
        self.term = Terminal()
        self.service_status = {}  # Maps service_name -> (status_type, timestamp)
        self.STATUS_TIMEOUT = 5  # Seconds before auto-clearing service status
        # Fixed column width for log truncation (icon + name + pid + time + mem + cpu + age + spacing)
        self.LOG_FIXED_WIDTH = 63
        self.last_log_lines = (
            {}
        )  # Maps ref -> (timestamp, stream, line) for most recent log
        self.cpu_cache = {}  # Maps pid -> last cpu_percent value
        self.cpu_procs = {}  # Maps pid -> Process object for cpu tracking
        self.running_tasks = {}  # Maps ref -> task info from logs tract
        self.command_queues = (
            {}
        )  # Maps command_name -> queued count from supervisor.queue
        self.event_queue: queue.Queue = queue.Queue()  # Callosum events for main loop
        self.active_notifications = {}  # Maps service_name -> notification_id
        self.crash_history = {}  # Maps service_name -> [crash_timestamps]

        # Observe status tracking (merged from observer and sense events)
        self.observe_status = {}  # Latest observe/status event fields (merged)
        self.observe_last_ts = 0.0  # Timestamp when last observe/status event received
        self.recent_segments = []  # Last 3 completed segments (day, segment, duration)

        # Mode hysteresis: don't show IDLE until 10s of continuous idle
        self.displayed_mode = "idle"  # What we show in the UI
        self.last_active_ts = 0.0  # When we last saw an active mode
        self.MODE_IDLE_DELAY = 10  # Seconds before showing IDLE after going idle

    def count_recent_crashes(self, service: str, window_minutes: int = 5) -> int:
        """Count recent crashes for a service within the time window.

        Args:
            service: Service name
            window_minutes: Time window in minutes to count crashes

        Returns:
            Number of crashes within the time window
        """
        if service not in self.crash_history:
            return 0

        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        # Filter to only recent crashes
        recent = [ts for ts in self.crash_history[service] if ts >= cutoff]
        # Update history to remove old crashes
        self.crash_history[service] = recent
        return len(recent)

    def set_service_status(self, service: str, status_type: str) -> None:
        """Set per-service status with timestamp for auto-clear.

        Args:
            service: Service name
            status_type: One of "requested", "restarting", "started", "stopped"
        """
        self.service_status[service] = (status_type, time.time())

    def get_service_icon(self, service: str) -> tuple[str, str]:
        """Get status icon and color for a service.

        Args:
            service: Service name

        Returns:
            Tuple of (icon, terminal_color_attr) where color_attr is
            a blessed Terminal attribute name like "green" or "normal"
        """
        if service not in self.service_status:
            return (" ", "normal")

        status_type, timestamp = self.service_status[service]

        # Check if status has expired
        if time.time() - timestamp > self.STATUS_TIMEOUT:
            return (" ", "normal")

        # Return icon based on status type
        if status_type == "requested":
            return ("↻", "yellow")
        elif status_type == "restarting":
            return ("◐", "yellow")
        elif status_type == "started":
            return ("✓", "green")
        elif status_type == "stopped":
            return ("✗", "red")
        else:
            return (" ", "normal")

    async def clear_notification(self, service: str) -> None:
        """Clear active notification for a service.

        Args:
            service: Service name
        """
        if service in self.active_notifications:
            notif_id = self.active_notifications[service]
            try:
                notifier = _get_notifier()
                await notifier.clear(notif_id)
            except Exception as exc:
                logging.debug("Failed to clear notification %s: %s", notif_id, exc)
            finally:
                # Remove from tracking even if clear failed
                del self.active_notifications[service]

    async def send_notification(self, service: str, message: str) -> None:
        """Send a desktop notification for a service crash.

        Tracks the notification ID and sets up dismissal callback.

        Args:
            service: Service name
            message: The notification message to display
        """
        try:
            notifier = _get_notifier()

            # Create dismissal callback that captures service name
            def on_dismissed() -> None:
                """Clean up tracking when user dismisses notification."""
                self.active_notifications.pop(service, None)

            notif_id = await notifier.send(
                title="solstone activity manager",
                message=message,
                urgency=Urgency.Critical,
                on_dismissed=on_dismissed,
            )

            # Track this notification
            self.active_notifications[service] = notif_id

        except Exception as exc:
            logging.error("Failed to send notification for %s: %s", service, exc)

    def _queue_event(self, message: dict) -> None:
        """Queue Callosum event for processing in main loop (thread-safe).

        Called from Callosum's background thread. All state mutations happen
        in _process_event which runs in the main async loop.

        Args:
            message: Callosum message dict with tract/event fields
        """
        self.event_queue.put_nowait(message)

    async def _process_event(self, message: dict) -> None:
        """Process a Callosum event (runs in main async loop).

        Args:
            message: Callosum message dict with tract/event fields
        """
        tract = message.get("tract")
        event = message.get("event")

        if tract == "supervisor":
            if event == "status":
                self.services = message.get("services", [])
                self.crashed = message.get("crashed", [])
                self.tasks = message.get("tasks", [])

                # Poll CPU for current services and tasks
                all_pids = [svc["pid"] for svc in self.services]
                all_pids.extend([task["pid"] for task in self.running_tasks.values()])

                for pid in all_pids:
                    try:
                        if pid not in self.cpu_procs:
                            # First time seeing this PID - initialize tracking
                            self.cpu_procs[pid] = psutil.Process(pid)
                            self.cpu_procs[pid].cpu_percent(interval=None)  # Start
                        else:
                            # Get CPU % since last status update
                            self.cpu_cache[pid] = self.cpu_procs[pid].cpu_percent(
                                interval=None
                            )
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # Clean up dead processes
                        self.cpu_procs.pop(pid, None)
                        self.cpu_cache.pop(pid, None)

                # Keep selection in bounds
                if self.selected >= len(self.services):
                    self.selected = max(0, len(self.services) - 1)

            elif event == "restarting":
                service = message.get("service")
                self.set_service_status(service, "restarting")

            elif event == "started":
                service = message.get("service")
                self.set_service_status(service, "started")

                # Clear any active crash notification (service restarted successfully)
                if service in self.active_notifications:
                    await self.clear_notification(service)

                # Reset crash history when service starts successfully
                self.crash_history.pop(service, None)

            elif event == "queue":
                # Track per-command queue depths
                command = message.get("command")
                queued = message.get("queued", 0)
                if command:
                    if queued > 0:
                        self.command_queues[command] = queued
                    else:
                        self.command_queues.pop(command, None)

            elif event == "stopped":
                service = message.get("service")
                exit_code = message.get("exit_code", "?")
                ref = message.get("ref")
                self.set_service_status(service, "stopped")

                # Send notification for non-zero exits
                if exit_code != 0 and exit_code != "?":
                    # Track crash timestamp
                    if service not in self.crash_history:
                        self.crash_history[service] = []
                    self.crash_history[service].append(datetime.now())

                    # Count recent crashes
                    crash_count = self.count_recent_crashes(service)

                    # Get last log line if available
                    log_line = ""
                    if ref and ref in self.last_log_lines:
                        _, _, log_line = self.last_log_lines[ref]
                        # Truncate to 100 chars to avoid giant notifications
                        log_line = log_line[:100]

                    # Format notification message with crash count
                    if crash_count > 1:
                        msg = f"{service} crashed ({crash_count}x in 5 min)\nExit code: {exit_code}"
                    else:
                        msg = f"{service} exited with code {exit_code}"

                    if log_line:
                        msg += f"\nLast log: {log_line}"

                    # Clear existing notification first (deduplication)
                    if service in self.active_notifications:
                        await self.clear_notification(service)

                    # Send notification directly (we're in the async loop)
                    await self.send_notification(service, msg)

        elif tract == "logs":
            if event == "exec":
                # New process started via runner
                ref = message.get("ref")
                name = message.get("name")
                pid = message.get("pid")
                cmd = message.get("cmd", [])
                if ref and name and pid:
                    self.running_tasks[ref] = {
                        "ref": ref,
                        "name": name,
                        "pid": pid,
                        "cmd": cmd,
                        "start_time": datetime.now(),
                    }
                    # Initialize CPU tracking for this task
                    try:
                        self.cpu_procs[pid] = psutil.Process(pid)
                        self.cpu_procs[pid].cpu_percent(interval=None)  # Start tracking
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            elif event == "line":
                ref = message.get("ref")
                name = message.get("name")
                pid = message.get("pid")
                line = message.get("line", "")
                stream = message.get("stream", "stdout")

                if ref:
                    # Create task entry if we haven't seen it yet (missed exec event)
                    if ref not in self.running_tasks and name and pid:
                        self.running_tasks[ref] = {
                            "ref": ref,
                            "name": name,
                            "pid": pid,
                            "cmd": [],  # Unknown since we missed exec
                            "start_time": datetime.now(),
                        }
                        # Initialize CPU tracking
                        try:
                            self.cpu_procs[pid] = psutil.Process(pid)
                            self.cpu_procs[pid].cpu_percent(interval=None)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    # Store log line
                    self.last_log_lines[ref] = (datetime.now(), stream, line)

            elif event == "exit":
                # Process exited - clean up
                ref = message.get("ref")
                if ref:
                    # Remove from running tasks
                    task = self.running_tasks.pop(ref, None)
                    if task:
                        # Clean up CPU tracking for this task's PID
                        pid = task["pid"]
                        self.cpu_procs.pop(pid, None)
                        self.cpu_cache.pop(pid, None)

                    # Clean up log lines
                    self.last_log_lines.pop(ref, None)

        elif tract == "observe":
            if event == "status":
                # Merge observe status (observer and sense emit different fields)
                # Observer: mode, screencast, tmux, audio, activity
                # Sense: describe, transcribe
                for key, value in message.items():
                    if key not in ("tract", "event", "ts"):
                        self.observe_status[key] = value
                self.observe_last_ts = time.time()

                # Track mode hysteresis for stable display
                mode = message.get("mode")
                if mode and mode != "idle":
                    # Active mode - update timestamp and display immediately
                    self.last_active_ts = time.time()
                    self.displayed_mode = mode

            elif event == "observed":
                # Segment completed - track recent completions
                day = message.get("day")
                segment = message.get("segment")
                duration = message.get("duration", 0)
                if day and segment:
                    self.recent_segments.insert(0, (day, segment, duration))
                    # Keep only last 3
                    self.recent_segments = self.recent_segments[:3]

    def format_uptime(self, seconds: int) -> str:
        """Format uptime in human-readable format.

        Args:
            seconds: Uptime in seconds

        Returns:
            Formatted string like "2d 3h 15m" or "45s"
        """
        if seconds < 60:
            return f"{seconds}s"
        delta = timedelta(seconds=seconds)
        parts = []
        if delta.days:
            parts.append(f"{delta.days}d")
        hours = delta.seconds // 3600
        if hours:
            parts.append(f"{hours}h")
        mins = (delta.seconds % 3600) // 60
        if mins:
            parts.append(f"{mins}m")
        return " ".join(parts)

    def format_log_age(self, timestamp: datetime) -> str:
        """Format log timestamp age in human-readable format.

        Args:
            timestamp: When the log line was received

        Returns:
            Formatted string like "0m", "35m", "2h", "3d" (never seconds)
        """
        delta = datetime.now() - timestamp
        total_seconds = int(delta.total_seconds())

        if total_seconds < 60:
            return "0m"
        elif total_seconds < 3600:  # Less than 1 hour
            return f"{total_seconds // 60}m"
        elif total_seconds < 86400:  # Less than 1 day
            return f"{total_seconds // 3600}h"
        else:
            return f"{total_seconds // 86400}d"

    def format_runtime(self, start_time: datetime) -> str:
        """Format task runtime in human-readable format.

        Args:
            start_time: When the task started

        Returns:
            Formatted string like "45s", "2m 15s", "1h 5m"
        """
        delta = datetime.now() - start_time
        total_seconds = int(delta.total_seconds())

        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:  # Less than 1 hour
            mins = total_seconds // 60
            secs = total_seconds % 60
            return f"{mins}m {secs}s"
        else:  # 1 hour or more
            hours = total_seconds // 3600
            mins = (total_seconds % 3600) // 60
            return f"{hours}h {mins}m"

    def get_memory_mb(self, pid: int) -> str:
        """Get process memory in MB, or '-' if unavailable.

        Args:
            pid: Process ID

        Returns:
            Memory usage in MB as integer, or "-" if unavailable
        """
        try:
            process = psutil.Process(pid)
            mem_bytes = process.memory_info().rss  # Resident Set Size
            mem_mb = mem_bytes / (1024 * 1024)
            return str(int(round(mem_mb)))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return "-"

    def get_cpu_percent(self, pid: int) -> str:
        """Get cached CPU percentage, or '-' if unavailable.

        Args:
            pid: Process ID

        Returns:
            CPU percentage as integer, or "-" if unavailable
        """
        if pid in self.cpu_cache:
            return f"{self.cpu_cache[pid]:.0f}"
        return "-"

    def get_log_display(self, ref: str) -> tuple[str, str, str]:
        """Get formatted log line, color, and age for a ref.

        Args:
            ref: Process reference ID

        Returns:
            Tuple of (log_display, log_color, log_age) where:
            - log_display: Truncated log line for available width
            - log_color: Terminal color attribute for the log line
            - log_age: Formatted age string like "0m", "5m", "2h"
        """
        if ref not in self.last_log_lines:
            return ("", "", "")

        t = self.term
        timestamp, stream, log_line = self.last_log_lines[ref]
        log_age = self.format_log_age(timestamp)

        # Calculate available width for log text
        available = max(0, t.width - self.LOG_FIXED_WIDTH)

        if available <= 0:
            return ("", "", log_age)

        # Truncate log line if needed
        if len(log_line) > available:
            log_display = log_line[: available - 3] + "..."
        else:
            log_display = log_line

        # Color code based on stream
        log_color = t.red if stream == "stderr" else t.normal

        return (log_display, log_color, log_age)

    def cleanup_dead_tasks(self) -> None:
        """Remove tasks whose PIDs are no longer valid.

        This handles cases where:
        - Process died without sending exit event (crash, kill -9, etc.)
        - Exit event was lost or not received
        - Process became a zombie
        """
        refs_to_remove = []

        for ref, task in self.running_tasks.items():
            pid = task["pid"]
            try:
                # Try to get process info - this will fail if PID is invalid
                process = psutil.Process(pid)
                # Check if process is a zombie
                if process.status() == psutil.STATUS_ZOMBIE:
                    refs_to_remove.append(ref)
            except psutil.NoSuchProcess:
                # Process no longer exists
                refs_to_remove.append(ref)
            except psutil.AccessDenied:
                # We don't have permission - assume it's still running
                pass

        # Clean up dead tasks
        for ref in refs_to_remove:
            task = self.running_tasks.pop(ref, None)
            if task:
                pid = task["pid"]
                self.cpu_procs.pop(pid, None)
                self.cpu_cache.pop(pid, None)
            self.last_log_lines.pop(ref, None)

    def render_tasks_table(self) -> list[str]:
        """Render the running tasks table.

        Returns:
            List of output lines for the tasks table
        """
        t = self.term
        output = []

        # Get PIDs of supervised services to exclude from tasks
        service_pids = {svc["pid"] for svc in self.services}

        # Filter out tasks that are actually supervised services
        tasks_only = [
            task
            for task in self.running_tasks.values()
            if task["pid"] not in service_pids
        ]

        # Section separator and table header
        output.append("─" * t.width)
        header = f"  {'Task':<15} {'PID':<8} {'Runtime':<12} {'MB':>7}  {'%':>5} {'Last':>5} Log"
        output.append(t.bold + header + t.normal)

        if not tasks_only:
            output.append(t.dim + "  -" + t.normal)
            return output

        # Task rows (sorted by start time, oldest first)
        tasks_sorted = sorted(tasks_only, key=lambda x: x["start_time"])

        for task in tasks_sorted:
            name = task["name"]
            # Append queue count if this command has queued requests
            queued = self.command_queues.get(name, 0)
            if queued > 0:
                name = f"{name} ({queued})"
            name = name[:14]
            pid = str(task["pid"])
            runtime = self.format_runtime(task["start_time"])
            memory = self.get_memory_mb(task["pid"])
            cpu = self.get_cpu_percent(task["pid"])
            log_display, log_color, log_age = self.get_log_display(task["ref"])

            line = f"  {name:<15} {pid:<8} {runtime:<12} {memory:>7}  {cpu:>5} {log_age:>5} "
            output.append(line + log_color + log_display + t.normal)

        return output

    def format_queue_status(self, handler_dict: dict) -> str:
        """Format processing queue status for a handler with fixed width.

        Args:
            handler_dict: Dict with optional 'running' and 'queued' keys

        Returns:
            Fixed-width (8 char) string like "▸1 +2   " or "─       " when empty
        """
        width = 8  # Handles up to +999 queued items
        if not handler_dict:
            return "─".ljust(width)

        parts = []
        if handler_dict.get("running"):
            parts.append("▸1")
        queued = handler_dict.get("queued", [])
        if queued:
            parts.append(f"+{len(queued)}")

        result = " ".join(parts) if parts else "─"
        return result.ljust(width)

    def get_displayed_mode(self) -> str:
        """Get the mode to display, with hysteresis to avoid IDLE flicker.

        State updates happen in _process_event() when events arrive.
        This method only reads state to determine what to display.

        Returns:
            Mode string: "screencast", "tmux", or "idle"
        """
        raw_mode = self.observe_status.get("mode", "idle")

        # Active modes - return what handle_event() already set
        if raw_mode in ("screencast", "tmux"):
            return self.displayed_mode

        # For idle: only show after MODE_IDLE_DELAY seconds of continuous idle
        time_since_active = time.time() - self.last_active_ts
        if time_since_active >= self.MODE_IDLE_DELAY:
            return "idle"

        # Still within grace period - show last active mode
        return self.displayed_mode

    def render_observe_section(self) -> list[str]:
        """Render the observe status section with stable layout.

        Returns:
            List of output lines for the observe section
        """
        t = self.term
        output = []

        # Health indicator (dot based on heartbeat freshness)
        if self.observe_last_ts > 0:
            age_seconds = int(time.time() - self.observe_last_ts)
            if age_seconds < 30:
                health_dot = t.green + "●" + t.normal
            elif age_seconds < 60:
                health_dot = t.yellow + "●" + t.normal
            else:
                health_dot = t.red + "●" + t.normal
        else:
            health_dot = t.dim + "○" + t.normal

        # Section header with health dot
        output.append("─" * t.width)
        output.append(f"  {t.bold}Observe{t.normal} {health_dot}")

        if not self.observe_status:
            # No status received yet
            output.append(t.dim + "  (waiting for status)" + t.normal)
        else:
            # Build fixed-width status line
            status_parts = []

            # Mode with hysteresis (avoids IDLE flicker)
            mode = self.get_displayed_mode()
            if mode == "screencast":
                screencast = self.observe_status.get("screencast", {})
                elapsed = screencast.get("window_elapsed_seconds", 0)
                mode_str = (
                    t.red
                    + "[LIVE]"
                    + t.normal
                    + f" screencast {self.format_uptime(elapsed)}"
                )
            elif mode == "tmux":
                tmux = self.observe_status.get("tmux", {})
                captures = tmux.get("captures", 0)
                mode_str = t.magenta + "[TMUX]" + t.normal + f" {captures} captures"
            else:
                activity = self.observe_status.get("activity", {})
                if activity.get("screen_locked"):
                    mode_str = t.dim + "[IDLE] locked" + t.normal
                else:
                    mode_str = t.dim + "[IDLE]" + t.normal
            status_parts.append(mode_str)

            # Voice activity (only show when there are hits)
            audio = self.observe_status.get("audio", {})
            hits = audio.get("threshold_hits", 0)
            if hits > 0:
                will_save = audio.get("will_save", False)
                if will_save:
                    status_parts.append(t.green + f"voice {hits}" + t.normal)
                else:
                    status_parts.append(f"voice {hits}")

            # Processing queues (always shown with fixed width)
            describe = self.observe_status.get("describe", {})
            transcribe = self.observe_status.get("transcribe", {})

            describe_status = self.format_queue_status(describe)
            transcribe_status = self.format_queue_status(transcribe)

            status_parts.append(f"describe {describe_status}")
            status_parts.append(f"transcribe {transcribe_status}")

            # Join with separator
            output.append("  " + " │ ".join(status_parts))

        # Recent segments
        if self.recent_segments:
            recent_strs = []
            for _day, segment, duration in self.recent_segments:
                # Just show segment key and duration in minutes
                duration_min = max(1, duration // 60)
                recent_strs.append(f"{segment} ({duration_min}m)")
            output.append(t.dim + "  Recent: " + " ".join(recent_strs) + t.normal)

        return output

    def render(self) -> str:
        """Render the entire UI.

        Returns:
            Complete terminal output string
        """
        t = self.term
        output = []

        # Clear and move to top
        output.append(t.home + t.clear)

        # Title
        title = "solstone activity manager"
        output.append(t.bold + t.cyan + title.center(t.width) + t.normal)
        output.append("")

        # Table header
        header = f"  {'Service':<15} {'PID':<8} {'Uptime':<12} {'MB':>7}  {'%':>5} {'Last':>5} Log"
        output.append(t.bold + header + t.normal)
        output.append("─" * t.width)

        # Service rows
        if self.services:
            for i, svc in enumerate(self.services):
                icon, icon_color = self.get_service_icon(svc["name"])
                name = svc["name"][:14]
                pid = str(svc["pid"])
                uptime = self.format_uptime(svc["uptime_seconds"])
                memory = self.get_memory_mb(svc["pid"])
                cpu = self.get_cpu_percent(svc["pid"])
                log_display, log_color, log_age = self.get_log_display(svc["ref"])

                line_content = f" {name:<15} {pid:<8} {uptime:<12} {memory:>7}  {cpu:>5} {log_age:>5} "

                if i == self.selected:
                    # Selected row: highlight entire row
                    output.append(t.black_on_white(icon + line_content + log_display))
                else:
                    # Non-selected: colored icon + normal content + log color
                    icon_str = getattr(t, icon_color) + icon + t.normal
                    output.append(
                        icon_str + line_content + log_color + log_display + t.normal
                    )
        else:
            output.append(t.dim + "  (waiting for services)" + t.normal)

        # Observe status section
        observe_output = self.render_observe_section()
        output.extend(observe_output)

        # Running tasks table (from logs tract)
        tasks_output = self.render_tasks_table()
        output.extend(tasks_output)

        # Crashed services (if any)
        if self.crashed:
            output.append(t.bold + t.red + "Crashed:" + t.normal)
            for c in self.crashed:
                output.append(f"  {c['name']} (attempts: {c['restart_attempts']})")
            output.append("")

        # Help footer
        output.append("─" * t.width)
        if len(self.services) > 1:
            output.append(t.dim + "↑/↓: Navigate  r: Restart  q: Quit" + t.normal)
        elif len(self.services) == 1:
            output.append(t.dim + "r: Restart  q: Quit" + t.normal)
        else:
            output.append(t.dim + "q: Quit" + t.normal)

        return "\n".join(output)

    def send_restart(self) -> None:
        """Send restart request for currently selected service."""
        if 0 <= self.selected < len(self.services):
            svc = self.services[self.selected]
            self.callosum.emit("supervisor", "restart", service=svc["name"])
            self.set_service_status(svc["name"], "requested")

    async def run(self) -> None:
        """Main event loop for the TUI."""
        self.callosum.start(callback=self._queue_event)

        # Track iteration count for periodic timeout checks
        iteration = 0

        with self.term.cbreak(), self.term.hidden_cursor():
            # Initial render
            print(self.render(), flush=True)

            while self.running:
                # Non-blocking keyboard check
                key = self.term.inkey(timeout=0.2)

                if key:
                    if key.name == "KEY_UP":
                        self.selected = max(0, self.selected - 1)
                    elif key.name == "KEY_DOWN":
                        self.selected = min(
                            len(self.services) - 1 if self.services else 0,
                            self.selected + 1,
                        )
                    elif key.lower() == "r":
                        self.send_restart()
                    elif key.lower() == "q" or key.code == 3:  # q or Ctrl-C
                        self.running = False
                    elif key.code == 4:  # Ctrl-D
                        self.running = False

                # Process queued Callosum events (thread-safe: queue filled by
                # _queue_event in Callosum thread, drained here in main loop)
                while True:
                    try:
                        msg = self.event_queue.get_nowait()
                        await self._process_event(msg)
                    except queue.Empty:
                        break

                # Check for dead tasks roughly every 5 seconds (~17 iterations at ~0.3s each)
                iteration += 1
                if iteration % 17 == 0:
                    self.cleanup_dead_tasks()

                # Render on every iteration (includes callosum updates)
                print(self.render(), flush=True)

                # Small delay to avoid busy loop
                await asyncio.sleep(0.1)

        self.callosum.stop()
        # Clear screen on exit
        print(self.term.clear())


def main() -> None:
    """CLI entry point for solstone activity manager."""
    parser = argparse.ArgumentParser(
        description="solstone activity manager - real-time service monitoring"
    )
    setup_cli(parser)

    manager = ServiceManager()
    try:
        asyncio.run(manager.run())
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
