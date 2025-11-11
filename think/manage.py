"""Interactive service manager for Sunstone supervisor.

Connects to the Callosum message bus to display real-time service status
and provides keyboard controls for restarting services.
"""

import argparse
import asyncio
from datetime import datetime, timedelta

import psutil
from blessed import Terminal

from think.callosum import CallosumConnection
from think.utils import setup_cli


class ServiceManager:
    """Interactive TUI for managing Sunstone services."""

    def __init__(self):
        self.services = []  # From supervisor/status events
        self.crashed = []  # From supervisor/status crashed field
        self.tasks = []  # From supervisor/status tasks field
        self.selected = 0
        self.callosum = CallosumConnection()
        self.running = True
        self.term = Terminal()
        self.status_message = ""
        self.last_log_lines = {}  # Maps ref -> (timestamp, stream, line) for most recent log
        self.cpu_cache = {}  # Maps pid -> last cpu_percent value
        self.cpu_procs = {}  # Maps pid -> Process object for cpu tracking
        self.running_tasks = {}  # Maps ref -> task info from logs tract

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

            elif event == "stopped":
                service = message.get("service")
                exit_code = message.get("exit_code", "?")
                self.status_message = f"Stopped {service} (exit {exit_code})"

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
                line = message.get("line", "")
                stream = message.get("stream", "stdout")
                if ref:
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

    def render_tasks_table(self) -> list[str]:
        """Render the running tasks table.

        Returns:
            List of output lines for the tasks table
        """
        if not self.running_tasks:
            return []

        t = self.term
        output = []

        # Section header
        count = len(self.running_tasks)
        output.append("")
        output.append(t.bold + f"Running Tasks ({count})" + t.normal)
        output.append("─" * min(80, t.width))

        # Table header
        header = f"  {'Task':<15} {'PID':<8} {'Runtime':<12} {'MB':<8} {'%':<6} {'Last':<6} {'Log'}"
        output.append(t.bold + header + t.normal)

        # Task rows (sorted by start time, oldest first)
        tasks_sorted = sorted(
            self.running_tasks.values(), key=lambda x: x["start_time"]
        )

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
        title = "Sunstone Service Manager"
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
        description="Interactive service manager for Sunstone supervisor"
    )
    setup_cli(parser)

    manager = ServiceManager()
    try:
        asyncio.run(manager.run())
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
