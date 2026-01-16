# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Interactive service manager for solstone supervisor.

Connects to the Callosum message bus to display real-time service status
and provides keyboard controls for restarting services.
"""

import argparse
import asyncio
import logging
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
        _notifier = DesktopNotifier(app_name="solstone Manager")
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
        self.status_message = ""
        self.last_log_lines = (
            {}
        )  # Maps ref -> (timestamp, stream, line) for most recent log
        self.cpu_cache = {}  # Maps pid -> last cpu_percent value
        self.cpu_procs = {}  # Maps pid -> Process object for cpu tracking
        self.running_tasks = {}  # Maps ref -> task info from logs tract
        self.pending_notifications = []  # Queue for async notifications (dicts)
        self.active_notifications = {}  # Maps service_name -> notification_id
        self.crash_history = {}  # Maps service_name -> [crash_timestamps]

        # Observe status tracking
        self.observe_status = {}  # Latest observe/status event fields
        self.observe_last_ts = 0.0  # Timestamp when last status received
        self.recent_segments = []  # Last 3 completed segments (day, segment, duration)

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
                title="solstone Manager",
                message=message,
                urgency=Urgency.Critical,
                on_dismissed=on_dismissed,
            )

            # Track this notification
            self.active_notifications[service] = notif_id

        except Exception as exc:
            logging.error("Failed to send notification for %s: %s", service, exc)

    def handle_event(self, message: dict) -> None:
        """Process Callosum events.

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
                        if pid in self.cpu_procs:
                            del self.cpu_procs[pid]
                        if pid in self.cpu_cache:
                            del self.cpu_cache[pid]

                # Keep selection in bounds
                if self.selected >= len(self.services):
                    self.selected = max(0, len(self.services) - 1)

            elif event == "restarting":
                service = message.get("service")
                self.status_message = f"Restarting {service}..."

            elif event == "started":
                service = message.get("service")
                self.status_message = f"Started {service}"

                # Clear any active crash notification (service restarted successfully)
                if service in self.active_notifications:
                    self.pending_notifications.append(
                        {"service": service, "message": None, "clear_only": True}
                    )

                # Reset crash history when service starts successfully
                if service in self.crash_history:
                    del self.crash_history[service]

            elif event == "stopped":
                service = message.get("service")
                exit_code = message.get("exit_code", "?")
                ref = message.get("ref")
                self.status_message = f"Stopped {service} (exit {exit_code})"

                # Queue notification for non-zero exits
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

                    # Queue notification dict with service info
                    self.pending_notifications.append(
                        {"service": service, "message": msg}
                    )

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
                    if ref in self.running_tasks:
                        # Clean up CPU tracking for this task's PID
                        task = self.running_tasks[ref]
                        pid = task["pid"]
                        if pid in self.cpu_procs:
                            del self.cpu_procs[pid]
                        if pid in self.cpu_cache:
                            del self.cpu_cache[pid]
                        del self.running_tasks[ref]

                    # Clean up log lines
                    if ref in self.last_log_lines:
                        del self.last_log_lines[ref]

        elif tract == "observe":
            if event == "status":
                # Update observe status and heartbeat timestamp
                self.observe_status = message
                self.observe_last_ts = time.time()

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
            task = self.running_tasks[ref]
            pid = task["pid"]

            # Clean up CPU tracking
            if pid in self.cpu_procs:
                del self.cpu_procs[pid]
            if pid in self.cpu_cache:
                del self.cpu_cache[pid]

            # Remove task
            del self.running_tasks[ref]

            # Clean up log lines
            if ref in self.last_log_lines:
                del self.last_log_lines[ref]

    def render_tasks_table(self) -> list[str]:
        """Render the running tasks table.

        Returns:
            List of output lines for the tasks table
        """
        if not self.running_tasks:
            return []

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

        if not tasks_only:
            return []

        # Section header
        count = len(tasks_only)
        output.append("")
        output.append(t.bold + f"Running Tasks ({count})" + t.normal)
        output.append("─" * min(80, t.width))

        # Table header
        header = f"  {'Task':<15} {'PID':<8} {'Runtime':<12} {'MB':<8} {'%':<6} {'Last':<6} {'Log'}"
        output.append(t.bold + header + t.normal)

        # Task rows (sorted by start time, oldest first)
        tasks_sorted = sorted(tasks_only, key=lambda x: x["start_time"])

        for task in tasks_sorted:
            name = task["name"][:14]
            pid = str(task["pid"])
            runtime = self.format_runtime(task["start_time"])
            memory = self.get_memory_mb(task["pid"])
            cpu = self.get_cpu_percent(task["pid"])

            # Get log line for this task
            log_display = ""
            log_color = ""
            log_age = ""
            ref = task["ref"]
            if ref in self.last_log_lines:
                timestamp, stream, log_line = self.last_log_lines[ref]
                log_age = self.format_log_age(timestamp)
                # Calculate available width for log text
                # Fixed: "  " (2) + name (15) + pid (8) + runtime (12) + memory (8) + cpu (6) + age (6) + spaces (6)
                fixed_width = 63
                available = max(0, t.width - fixed_width)

                if available > 0:
                    if len(log_line) > available:
                        log_display = log_line[: available - 3] + "..."
                    else:
                        log_display = log_line

                    # Color code based on stream
                    if stream == "stderr":
                        log_color = t.red
                    else:
                        log_color = t.normal

            line = f"  {name:<15} {pid:<8} {runtime:<12} {memory:>7}  {cpu:>5} {log_age:>5} "
            output.append(line + log_color + log_display + t.normal)

        return output

    def format_queue_status(self, handler_dict: dict) -> str:
        """Format processing queue status for a handler.

        Args:
            handler_dict: Dict with optional 'running' and 'queued' keys

        Returns:
            Formatted string like "▸1 +2" (1 running, 2 queued) or ""
        """
        if not handler_dict:
            return ""

        parts = []
        if handler_dict.get("running"):
            parts.append("▸1")
        queued = handler_dict.get("queued", [])
        if queued:
            parts.append(f"+{len(queued)}")

        return " ".join(parts)

    def render_observe_section(self) -> list[str]:
        """Render the observe status section.

        Returns:
            List of output lines for the observe section
        """
        t = self.term
        output = []

        # Calculate heartbeat age and display string
        if self.observe_last_ts > 0:
            age_seconds = int(time.time() - self.observe_last_ts)
            age_str = f"{age_seconds}s ago"
            if age_seconds < 30:
                heartbeat = t.green + age_str + t.normal
            elif age_seconds < 60:
                heartbeat = t.yellow + age_str + t.normal
            else:
                heartbeat = t.red + age_str + t.normal
        else:
            age_str = "waiting..."
            heartbeat = t.dim + age_str + t.normal

        # Header line with heartbeat indicator
        padding = t.width - len("Observe") - len(age_str) - 2
        output.append("")
        output.append("─" * min(80, t.width))
        output.append(f"{t.bold}Observe{t.normal}{' ' * max(1, padding)}{heartbeat}")

        # Build status line
        status_parts = []

        if not self.observe_status:
            # No status received yet
            output.append(t.dim + "  (no status received)" + t.normal)
        else:
            # Mode badge
            mode = self.observe_status.get("mode", "unknown")
            if mode == "screencast":
                badge = t.red + "[LIVE]" + t.normal
                # Get elapsed time from screencast info
                screencast = self.observe_status.get("screencast", {})
                elapsed = screencast.get("window_elapsed_seconds", 0)
                status_parts.append(f"{badge} screencast {self.format_uptime(elapsed)}")
            elif mode == "tmux":
                badge = t.magenta + "[TMUX]" + t.normal
                tmux = self.observe_status.get("tmux", {})
                captures = tmux.get("captures", 0)
                status_parts.append(f"{badge} {captures} captures")
            else:
                badge = t.dim + "[IDLE]" + t.normal
                activity = self.observe_status.get("activity", {})
                if activity.get("screen_locked"):
                    status_parts.append(f"{badge} screen locked")
                else:
                    status_parts.append(badge)

            # Voice activity
            audio = self.observe_status.get("audio", {})
            hits = audio.get("threshold_hits", 0)
            if hits > 0:
                will_save = audio.get("will_save", False)
                if will_save:
                    status_parts.append(t.green + f"voice {hits}" + t.normal)
                else:
                    status_parts.append(f"voice {hits}")

            # Processing queues (from sense.py status)
            describe = self.observe_status.get("describe", {})
            transcribe = self.observe_status.get("transcribe", {})

            describe_status = self.format_queue_status(describe)
            transcribe_status = self.format_queue_status(transcribe)

            if describe_status:
                status_parts.append(f"describe {describe_status}")
            if transcribe_status:
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
        title = "solstone Service Manager"
        output.append(t.bold + t.cyan + title.center(t.width) + t.normal)
        output.append("")

        # Table header
        header = f"  {'Service':<15} {'PID':<8} {'Uptime':<12} {'MB':<8} {'%':<6} {'Last':<6} {'Log'}"
        output.append(t.bold + header + t.normal)
        output.append("─" * min(80, t.width))

        # Service rows
        if self.services:
            for i, svc in enumerate(self.services):
                indicator = "→" if i == self.selected else " "
                name = svc["name"][:14]
                pid = str(svc["pid"])
                uptime = self.format_uptime(svc["uptime_seconds"])
                memory = self.get_memory_mb(svc["pid"])
                cpu = self.get_cpu_percent(svc["pid"])

                # Get log line for this service
                log_display = ""
                log_color = ""
                log_age = ""
                if svc["ref"] in self.last_log_lines:
                    timestamp, stream, log_line = self.last_log_lines[svc["ref"]]
                    log_age = self.format_log_age(timestamp)
                    # Calculate available width: total - (fixed columns)
                    # Fixed: "→ " (2) + name (15) + pid (8) + uptime (12) + memory (8) + cpu (6) + age (6) + spaces (6)
                    fixed_width = 63
                    available = max(0, t.width - fixed_width)

                    # Truncate log line if needed
                    if available > 0:
                        if len(log_line) > available:
                            log_display = log_line[: available - 3] + "..."
                        else:
                            log_display = log_line

                        # Color code based on stream
                        if stream == "stderr":
                            log_color = t.red
                        else:
                            log_color = t.normal

                line = f"{indicator} {name:<15} {pid:<8} {uptime:<12} {memory:>7}  {cpu:>5} {log_age:>5} "

                if i == self.selected:
                    output.append(t.black_on_white(line + log_display))
                else:
                    output.append(line + log_color + log_display + t.normal)
        else:
            output.append(t.dim + "  No services running" + t.normal)

        # Observe status section
        observe_output = self.render_observe_section()
        output.extend(observe_output)

        output.append("")

        # Running tasks table (from logs tract)
        tasks_output = self.render_tasks_table()
        output.extend(tasks_output)

        # Crashed services (if any)
        if self.crashed:
            output.append(t.bold + t.red + "Crashed:" + t.normal)
            for c in self.crashed:
                output.append(f"  {c['name']} (attempts: {c['restart_attempts']})")
            output.append("")

        # Running tasks (if any)
        if self.tasks:
            output.append(t.bold + "Running tasks:" + t.normal)
            for task in self.tasks[:3]:  # Show max 3
                duration = self.format_uptime(task["duration_seconds"])
                output.append(f"  {task['name']} - {duration}")
            if len(self.tasks) > 3:
                output.append(f"  ... and {len(self.tasks) - 3} more")
            output.append("")

        # Status message
        if self.status_message:
            output.append(t.green + self.status_message + t.normal)
            output.append("")

        # Help footer
        output.append("─" * min(80, t.width))
        output.append(t.dim + "↑/↓: Navigate  r: Restart  q/Ctrl-C: Quit" + t.normal)

        return "\n".join(output)

    def send_restart(self) -> None:
        """Send restart request for currently selected service."""
        if 0 <= self.selected < len(self.services):
            svc = self.services[self.selected]
            self.callosum.emit("supervisor", "restart", service=svc["name"])
            self.status_message = f"Restart requested for {svc['name']}"

    async def run(self) -> None:
        """Main event loop for the TUI."""
        self.callosum.start(callback=self.handle_event)

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
                        self.status_message = ""
                    elif key.name == "KEY_DOWN":
                        self.selected = min(
                            len(self.services) - 1 if self.services else 0,
                            self.selected + 1,
                        )
                        self.status_message = ""
                    elif key.lower() == "r":
                        self.send_restart()
                    elif key.lower() == "q" or key.code == 3:  # q or Ctrl-C
                        self.running = False
                    elif key.code == 4:  # Ctrl-D
                        self.running = False

                # Check for dead tasks every 5 seconds (50 iterations * 0.1s = 5s)
                iteration += 1
                if iteration % 50 == 0:
                    self.cleanup_dead_tasks()

                # Process pending notifications
                while self.pending_notifications:
                    notif = self.pending_notifications.pop(0)
                    service = notif["service"]

                    # Check if this is a clear-only request
                    if notif.get("clear_only"):
                        await self.clear_notification(service)
                    else:
                        # Clear any existing notification first (deduplication)
                        if service in self.active_notifications:
                            await self.clear_notification(service)

                        # Send new notification
                        message = notif["message"]
                        await self.send_notification(service, message)

                # Render on every iteration (includes callosum updates)
                print(self.render(), flush=True)

                # Small delay to avoid busy loop
                await asyncio.sleep(0.1)

        self.callosum.stop()
        # Clear screen on exit
        print(self.term.clear())


def main() -> None:
    """CLI entry point for service manager."""
    parser = argparse.ArgumentParser(
        description="Interactive service manager for solstone supervisor"
    )
    setup_cli(parser)

    manager = ServiceManager()
    try:
        asyncio.run(manager.run())
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
