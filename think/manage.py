"""Interactive service manager for Sunstone supervisor.

Connects to the Callosum message bus to display real-time service status
and provides keyboard controls for restarting services.
"""

import argparse
import asyncio
from datetime import timedelta

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
        self.last_log_lines = {}  # Maps ref -> (stream, line) for most recent log

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
            if event == "line":
                ref = message.get("ref")
                line = message.get("line", "")
                stream = message.get("stream", "stdout")
                if ref:
                    self.last_log_lines[ref] = (stream, line)

            elif event == "exit":
                # Clean up log lines for exited processes
                ref = message.get("ref")
                if ref and ref in self.last_log_lines:
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
        header = f"  {'Service':<15} {'PID':<8} {'Uptime':<12} {'Ref':<10} {'Last Log'}"
        output.append(t.bold + header + t.normal)
        output.append("─" * min(80, t.width))

        # Service rows
        if self.services:
            for i, svc in enumerate(self.services):
                indicator = "→" if i == self.selected else " "
                name = svc["name"][:14]
                pid = str(svc["pid"])
                uptime = self.format_uptime(svc["uptime_seconds"])
                ref = svc["ref"][:8] if len(svc["ref"]) > 8 else svc["ref"]

                # Get log line for this service
                log_display = ""
                log_color = ""
                if svc["ref"] in self.last_log_lines:
                    stream, log_line = self.last_log_lines[svc["ref"]]
                    # Calculate available width: total - (fixed columns)
                    # Fixed: "→ " (2) + name (15) + pid (8) + uptime (12) + ref (10) + spaces (4)
                    fixed_width = 51
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

                line = f"{indicator} {name:<15} {pid:<8} {uptime:<12} {ref:<10} "

                if i == self.selected:
                    output.append(t.black_on_white(line + log_display))
                else:
                    output.append(line + log_color + log_display + t.normal)
        else:
            output.append(t.dim + "  No services running" + t.normal)

        output.append("")

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
        output.append(t.dim + "↑/↓: Navigate  k: Restart  q/Ctrl-C: Quit" + t.normal)

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
                    elif key.lower() == "k":
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
