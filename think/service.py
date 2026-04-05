# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Cross-platform background service management for solstone.

Usage:
    sol service install [--port PORT]  Install solstone as a background service
    sol service uninstall              Remove the background service
    sol service start                  Start the background service
    sol service stop                   Stop the background service
    sol service restart [--if-installed]  Restart the background service
    sol service status                 Show service installation and runtime status
    sol service logs                   View service logs
    sol service logs -f                Follow service logs

    sol up                             Install (if needed), start, and show status
    sol down                           Stop the background service

Default convey port for installed services is 5015.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

from think.utils import get_journal, get_journal_info

SERVICE_LABEL = "org.solpbc.solstone"
SYSTEMD_UNIT = "solstone"
DEFAULT_SERVICE_PORT = 5015


def _platform() -> str:
    """Return 'darwin', 'linux', or raise on unsupported."""
    if sys.platform == "darwin":
        return "darwin"
    elif sys.platform.startswith("linux"):
        return "linux"
    else:
        print(f"Error: unsupported platform '{sys.platform}'", file=sys.stderr)
        sys.exit(1)


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_LABEL}.plist"


def _unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / f"{SYSTEMD_UNIT}.service"


def _sol_bin() -> str:
    """Return absolute path to the sol binary in the current venv."""
    return str(Path(sys.executable).parent / "sol")


def _collect_env() -> dict[str, str]:
    """Collect environment variables for the service file.

    Only captures HOME and PATH (with venv bin). API keys are NOT written
    into service files — the supervisor reads them from journal.json at
    process startup via setup_cli(). Never propagate _SOLSTONE_JOURNAL_OVERRIDE
    into service files — installed services should use default path resolution.
    """
    venv_bin = str(Path(sys.executable).parent)

    return {
        "HOME": str(Path.home()),
        "PATH": f"{venv_bin}:/usr/local/bin:/usr/bin:/bin",
    }


def _generate_plist(env: dict[str, str], port: int = DEFAULT_SERVICE_PORT) -> bytes:
    """Generate a launchd plist for the solstone supervisor."""
    journal_path = str(Path(get_journal()).resolve())
    sol = _sol_bin()

    plist = {
        "Label": SERVICE_LABEL,
        "ProgramArguments": [sol, "supervisor", str(port)],
        "EnvironmentVariables": env,
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": f"{journal_path}/health/launchd-stdout.log",
        "StandardErrorPath": f"{journal_path}/health/launchd-stderr.log",
    }
    return plistlib.dumps(plist)


def _generate_systemd_unit(
    env: dict[str, str], port: int = DEFAULT_SERVICE_PORT
) -> str:
    """Generate a systemd user unit for the solstone supervisor."""
    sol = _sol_bin()
    env_lines = "\n".join(f"Environment={k}={v}" for k, v in sorted(env.items()))

    return (
        f"[Unit]\n"
        f"Description=Solstone Supervisor\n"
        f"After=default.target\n"
        f"\n"
        f"[Service]\n"
        f"Type=simple\n"
        f"ExecStart={sol} supervisor {port}\n"
        f"Restart=on-failure\n"
        f"RestartSec=5\n"
        f"{env_lines}\n"
        f"\n"
        f"[Install]\n"
        f"WantedBy=default.target\n"
    )


def _check_linger() -> None:
    """Warn if systemd linger is not enabled for the current user."""
    try:
        result = subprocess.run(
            ["loginctl", "show-user", os.environ.get("USER", ""), "--property=Linger"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and "Linger=no" in result.stdout:
            print(
                "Warning: systemd linger is not enabled. "
                "The service will stop when you log out.\n"
                "Enable it with: sudo loginctl enable-linger $USER"
            )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def _install(port: int = DEFAULT_SERVICE_PORT) -> int:
    platform = _platform()
    env = _collect_env()

    journal_path, _source = get_journal_info()
    Path(journal_path, "health").mkdir(parents=True, exist_ok=True)

    if platform == "darwin":
        plist_data = _generate_plist(env, port=port)
        path = _plist_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        uid = os.getuid()
        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}", str(path)],
            capture_output=True,
        )

        path.write_bytes(plist_data)
        print(f"Wrote {path}")

        result = subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error loading service: {result.stderr.strip()}", file=sys.stderr)
            return 1
        print("Service loaded into launchd")

    else:
        unit_content = _generate_systemd_unit(env, port=port)
        path = _unit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(unit_content)
        print(f"Wrote {path}")

        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", SYSTEMD_UNIT], check=True)
        print("Service enabled")

        _check_linger()

    return 0


def _uninstall() -> int:
    platform = _platform()

    if platform == "darwin":
        path = _plist_path()
        uid = os.getuid()
        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}", str(path)],
            capture_output=True,
        )
        if path.exists():
            path.unlink()
            print(f"Removed {path}")
        else:
            print("Service was not installed")

    else:
        path = _unit_path()
        subprocess.run(
            ["systemctl", "--user", "stop", SYSTEMD_UNIT],
            capture_output=True,
        )
        subprocess.run(
            ["systemctl", "--user", "disable", SYSTEMD_UNIT],
            capture_output=True,
        )
        if path.exists():
            path.unlink()
            subprocess.run(
                ["systemctl", "--user", "daemon-reload"],
                capture_output=True,
            )
            print(f"Removed {path}")
        else:
            print("Service was not installed")

    return 0


def _start() -> int:
    platform = _platform()
    if platform == "darwin":
        uid = os.getuid()
        path = _plist_path()
        if not path.exists():
            print(
                "Error: service not installed. Run 'sol service install' first.",
                file=sys.stderr,
            )
            return 1
        result = subprocess.run(
            ["launchctl", "kickstart", f"gui/{uid}/{SERVICE_LABEL}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error starting service: {result.stderr.strip()}", file=sys.stderr)
            return 1
    else:
        if not _unit_path().exists():
            print(
                "Error: service not installed. Run 'sol service install' first.",
                file=sys.stderr,
            )
            return 1
        result = subprocess.run(
            ["systemctl", "--user", "start", SYSTEMD_UNIT],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error starting service: {result.stderr.strip()}", file=sys.stderr)
            return 1

    print("Service started")
    return 0


def _stop() -> int:
    platform = _platform()
    if platform == "darwin":
        uid = os.getuid()
        result = subprocess.run(
            ["launchctl", "kill", "SIGTERM", f"gui/{uid}/{SERVICE_LABEL}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error stopping service: {result.stderr.strip()}", file=sys.stderr)
            return 1
    else:
        result = subprocess.run(
            ["systemctl", "--user", "stop", SYSTEMD_UNIT],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error stopping service: {result.stderr.strip()}", file=sys.stderr)
            return 1

    print("Service stopped")
    return 0


def _restart(if_installed: bool = False) -> int:
    platform = _platform()
    if platform == "darwin":
        installed = _plist_path().exists()
    else:
        installed = _unit_path().exists()

    if not installed:
        if if_installed:
            return 0
        print(
            "Error: service not installed. Run 'sol service install' first.",
            file=sys.stderr,
        )
        return 1

    if platform == "darwin":
        uid = os.getuid()
        subprocess.run(
            ["launchctl", "kill", "SIGTERM", f"gui/{uid}/{SERVICE_LABEL}"],
            capture_output=True,
        )
        result = subprocess.run(
            ["launchctl", "kickstart", f"gui/{uid}/{SERVICE_LABEL}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error restarting service: {result.stderr.strip()}", file=sys.stderr)
            return 1
    else:
        result = subprocess.run(
            ["systemctl", "--user", "restart", SYSTEMD_UNIT],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error restarting service: {result.stderr.strip()}", file=sys.stderr)
            return 1

    print("Service restarted")
    return 0


def _status() -> int:
    platform = _platform()

    if platform == "darwin":
        installed = _plist_path().exists()
    else:
        installed = _unit_path().exists()

    if not installed:
        print("Service: not installed")
        print("Run 'sol service install' to install, or 'sol up' to install and start.")
        return 1

    print("Service: installed")

    if platform == "darwin":
        uid = os.getuid()
        result = subprocess.run(
            ["launchctl", "print", f"gui/{uid}/{SERVICE_LABEL}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("State: running (launchd)")
        else:
            print("State: stopped")
            return 0
    else:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", SYSTEMD_UNIT],
            capture_output=True,
            text=True,
        )
        state = result.stdout.strip()
        if state == "active":
            print("State: running (systemd)")
        else:
            print(f"State: {state}")
            return 0

    print()
    from think.health_cli import health_check

    return health_check()


def _logs(follow: bool = False) -> int:
    platform = _platform()

    if platform == "linux":
        cmd = ["journalctl", "--user", "-u", SYSTEMD_UNIT, "--no-pager", "-n", "100"]
        if follow:
            cmd.append("--follow")
        result = subprocess.run(cmd)
        return result.returncode
    else:
        journal_path = Path(get_journal())
        stdout_log = journal_path / "health" / "launchd-stdout.log"
        stderr_log = journal_path / "health" / "launchd-stderr.log"

        if follow:
            logs_to_follow = [str(p) for p in [stdout_log, stderr_log] if p.exists()]
            if not logs_to_follow:
                print("No service log files found", file=sys.stderr)
                return 1
            result = subprocess.run(["/usr/bin/tail", "-f"] + logs_to_follow)
            return result.returncode
        else:
            for log_path in [stdout_log, stderr_log]:
                if log_path.exists():
                    print(f"=== {log_path.name} ===")
                    print(log_path.read_text(errors="replace")[-10000:])
                else:
                    print(f"=== {log_path.name} === (not found)")
            return 0


def _up(port: int = DEFAULT_SERVICE_PORT) -> int:
    """Install if needed, start if not running, show status."""
    platform = _platform()

    if platform == "darwin":
        installed = _plist_path().exists()
    else:
        installed = _unit_path().exists()

    if not installed:
        print("Installing service...")
        rc = _install(port=port)
        if rc != 0:
            return rc

    if platform == "darwin":
        uid = os.getuid()
        result = subprocess.run(
            ["launchctl", "print", f"gui/{uid}/{SERVICE_LABEL}"],
            capture_output=True,
            text=True,
        )
        running = result.returncode == 0
    else:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", SYSTEMD_UNIT],
            capture_output=True,
            text=True,
        )
        running = result.stdout.strip() == "active"

    if not running:
        print("Starting service...")
        rc = _start()
        if rc != 0:
            return rc

    return _status()


def _down() -> int:
    """Stop the service."""
    return _stop()


_SUBCOMMANDS = {
    "uninstall": _uninstall,
    "start": _start,
    "stop": _stop,
    "status": _status,
    "down": lambda **_kw: _down(),
}


def _parse_port(args: list[str]) -> int:
    """Extract --port PORT from args, return DEFAULT_SERVICE_PORT if absent."""
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                print(f"Error: invalid port '{args[i + 1]}'", file=sys.stderr)
                sys.exit(1)
        if arg.startswith("--port="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                print(f"Error: invalid port '{arg}'", file=sys.stderr)
                sys.exit(1)
    return DEFAULT_SERVICE_PORT


def main() -> None:
    """Entry point for ``sol service``."""
    args = sys.argv[1:]

    if args and args[0] == "logs":
        follow = "-f" in args[1:] or "--follow" in args[1:]
        sys.exit(_logs(follow=follow))

    if not args:
        print("Usage: sol service <install|uninstall|start|stop|restart|status|logs>")
        print("       sol service install [--port PORT]  (default: 5015)")
        print(
            "       sol service restart [--if-installed]  "
            "(restart; --if-installed noops if not installed)"
        )
        print("       sol up [--port PORT]               (install + start + status)")
        print("       sol down                           (stop)")
        sys.exit(1)

    subcmd = args[0]
    rest = args[1:]

    if subcmd == "install":
        sys.exit(_install(port=_parse_port(rest)))
    elif subcmd == "up":
        sys.exit(_up(port=_parse_port(rest)))
    elif subcmd == "restart":
        if_installed = "--if-installed" in rest
        sys.exit(_restart(if_installed=if_installed))
    elif subcmd in _SUBCOMMANDS:
        sys.exit(_SUBCOMMANDS[subcmd]())
    else:
        print(f"Unknown subcommand: {subcmd}", file=sys.stderr)
        print("Available: install, uninstall, start, stop, restart, status, logs")
        sys.exit(1)
