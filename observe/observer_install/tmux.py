# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""tmux observer installer."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from apps.observer.utils import list_observers

from .common import (
    InstallError,
    create_or_reuse_registration,
    default_server_url,
    default_stream,
    emit_json,
    marker_path,
    observer_key_prefix_from_config,
    poll_status_until,
    print_summary,
    read_marker,
    run_probe,
    run_step,
    write_marker,
    xdg_install_dir,
)

PLATFORM = "tmux"
INSTALL_NAME = "solstone-tmux"
SOURCE_URL = "https://github.com/solpbc/solstone-tmux.git"
UNIT_NAME = "solstone-tmux.service"
CONFIG_PATH = (
    Path.home() / ".local" / "share" / "solstone-tmux" / "config" / "config.json"
)
DEFAULT_CONFIG = {
    "server_url": "",
    "key": "",
    "stream": "",
    "capture_interval": 5,
    "segment_interval": 300,
    "sync_retry_delays": [5, 30, 120, 300],
    "sync_max_retries": 10,
    "cache_retention_days": 7,
    "status_indicator": True,
}


class TmuxDriver:
    """Install and manage the tmux observer service."""

    def run(self, args) -> int:
        server_url = default_server_url(args.server_url)
        name = default_stream("tmux", args.name)
        clone_dir = xdg_install_dir(INSTALL_NAME)
        marker = read_marker(INSTALL_NAME)
        if marker and marker.get("name") != name and not args.force:
            raise InstallError(
                f"{INSTALL_NAME} is already installed for {marker.get('name')}",
                hint="rerun with --force to replace the existing install marker",
            )

        tool_statuses = _check_tools()
        tmux_present = run_probe(["sh", "-c", "command -v tmux"]).returncode == 0
        if args.dry_run:
            if not args.json_output:
                _print_dry_run(name, server_url, clone_dir, tool_statuses, tmux_present)
            if args.json_output:
                emit_json(
                    _result(name, server_url, clone_dir, "planned", None, None, True)
                )
            return 0

        _raise_for_preflight(tool_statuses)
        if not tmux_present and not args.json_output:
            print(
                "warning: tmux not detected on PATH; observer will start when tmux is launched"
            )

        version, changed = _prepare_source(clone_dir, args)
        active = _active_registration(name)
        service_active = _service_is_active()
        config_prefix = observer_key_prefix_from_config(CONFIG_PATH)

        if (
            marker
            and not args.force
            and not changed
            and active
            and config_prefix == active.get("key", "")[:8]
        ):
            if service_active:
                result = _result(
                    name,
                    server_url,
                    clone_dir,
                    "already_installed",
                    active.get("key", "")[:8],
                    version,
                    False,
                )
                _output_result(result, args.json_output)
                return 0
            run_step(
                f"restart {UNIT_NAME}",
                ["systemctl", "--user", "restart", UNIT_NAME],
                json_output=args.json_output,
            )
            status = poll_status_until(name)
            result = _result(
                name,
                server_url,
                clone_dir,
                status,
                active.get("key", "")[:8],
                version,
                False,
            )
            _output_result(result, args.json_output)
            return 0

        registration = create_or_reuse_registration(name, force=args.force)
        _write_config(server_url, registration.key, name)
        run_step(
            "run make install-service",
            ["make", "install-service"],
            cwd=clone_dir,
            json_output=args.json_output,
        )
        run_step(
            f"restart {UNIT_NAME}",
            ["systemctl", "--user", "restart", UNIT_NAME],
            json_output=args.json_output,
        )
        status = poll_status_until(name)
        version = _git_stdout(clone_dir, ["rev-parse", "HEAD"]) or version
        _write_install_marker(marker, name, version)
        result = _result(
            name,
            server_url,
            clone_dir,
            status,
            registration.prefix,
            version,
            False,
        )
        _output_result(result, args.json_output)
        return 0


def _check_tools() -> list[tuple[str, bool, str | None]]:
    checks = [
        (
            "git",
            ["sh", "-c", "command -v git"],
            "install git with your package manager",
        ),
        ("uv", ["sh", "-c", "command -v uv"], "install uv: https://docs.astral.sh/uv/"),
        (
            "pipx",
            ["sh", "-c", "command -v pipx"],
            "install pipx with your package manager",
        ),
        (
            "make",
            ["sh", "-c", "command -v make"],
            "install make with your package manager",
        ),
        (
            "systemctl --user",
            ["systemctl", "--user", "--version"],
            "systemd user services are required for this observer",
        ),
    ]
    return [
        (label, run_probe(cmd).returncode == 0, hint) for label, cmd, hint in checks
    ]


def _raise_for_preflight(tool_statuses: list[tuple[str, bool, str | None]]) -> None:
    missing_tools = [(label, hint) for label, ok, hint in tool_statuses if not ok]
    if missing_tools:
        label, hint = missing_tools[0]
        raise InstallError(f"missing required tool: {label}", hint=hint)


def _prepare_source(clone_dir: Path, args) -> tuple[str | None, bool]:
    if not clone_dir.exists():
        run_step(
            f"clone {SOURCE_URL} into {clone_dir}",
            ["git", "clone", SOURCE_URL, str(clone_dir)],
            json_output=args.json_output,
        )
        return _git_stdout(clone_dir, ["rev-parse", "HEAD"]), True

    if not (clone_dir / ".git").exists():
        raise InstallError(
            f"{clone_dir} exists but is not a git repository",
            hint="move it aside or choose --force after restoring the observer clone",
        )
    origin = _git_stdout(clone_dir, ["remote", "get-url", "origin"])
    if origin != SOURCE_URL:
        raise InstallError(
            f"{clone_dir} has unexpected origin {origin}",
            hint=f"expected {SOURCE_URL}",
        )
    dirty = run_probe(["git", "status", "--porcelain"], cwd=clone_dir)
    if dirty.stdout.strip():
        raise InstallError(
            f"{clone_dir} has local changes",
            hint="commit, clean, or move the clone before rerunning install",
        )

    run_step(
        "fetch observer updates",
        ["git", "fetch", "origin"],
        cwd=clone_dir,
        json_output=args.json_output,
    )
    local = _git_stdout(clone_dir, ["rev-parse", "HEAD"])
    upstream = _git_stdout(clone_dir, ["rev-parse", "@{u}"]) or _git_stdout(
        clone_dir, ["rev-parse", "origin/HEAD"]
    )
    if not local or not upstream or local == upstream:
        return local, False

    ancestor = run_probe(
        ["git", "merge-base", "--is-ancestor", local, upstream], cwd=clone_dir
    )
    if ancestor.returncode != 0:
        raise InstallError(
            f"{clone_dir} has diverged from upstream",
            hint="resolve the clone manually before rerunning install",
        )
    run_step(
        "pull observer updates",
        ["git", "pull", "--ff-only"],
        cwd=clone_dir,
        json_output=args.json_output,
    )
    return _git_stdout(clone_dir, ["rev-parse", "HEAD"]), True


def _write_config(server_url: str, key: str, name: str) -> None:
    config = dict(DEFAULT_CONFIG)
    config.update({"server_url": server_url, "key": key, "stream": name})
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise InstallError(f"failed to write {CONFIG_PATH}", hint=str(exc)) from exc


def _write_install_marker(marker: dict | None, name: str, version: str | None) -> None:
    now = _now_utc()
    write_marker(
        INSTALL_NAME,
        {
            "name": name,
            "platform": PLATFORM,
            "source": SOURCE_URL,
            "installed_at": (marker.get("installed_at") if marker else None) or now,
            "last_run": now,
            "version": version,
        },
    )


def _active_registration(name: str) -> dict | None:
    for observer in list_observers():
        if observer.get("name") == name and not observer.get("revoked", False):
            return observer
    return None


def _service_is_active() -> bool:
    process = run_probe(["systemctl", "--user", "is-active", UNIT_NAME])
    return process.returncode == 0 and process.stdout.strip() == "active"


def _git_stdout(cwd: Path, args: list[str]) -> str | None:
    process = run_probe(["git", *args], cwd=cwd)
    if process.returncode != 0:
        return None
    return process.stdout.strip() or None


def _now_utc() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _result(
    name: str,
    server_url: str,
    clone_dir: Path,
    status: str,
    key_prefix: str | None,
    version: str | None,
    dry_run: bool,
) -> dict:
    return {
        "platform": PLATFORM,
        "name": name,
        "source_path": str(clone_dir),
        "service_unit": UNIT_NAME,
        "key_prefix": key_prefix,
        "server_url": server_url,
        "config_path": str(CONFIG_PATH),
        "marker_path": str(marker_path(INSTALL_NAME)),
        "status": status,
        "version": version,
        "dry_run": dry_run,
    }


def _output_result(result: dict, json_output: bool) -> None:
    if json_output:
        emit_json(result)
    else:
        print_summary(result)


def _print_dry_run(
    name: str,
    server_url: str,
    clone_dir: Path,
    tool_statuses: list[tuple[str, bool, str | None]],
    tmux_present: bool,
) -> None:
    print("Dry-run: would install solstone-tmux observer")
    print()
    print("Platform: tmux")
    print(f"Stream:   {name}")
    print(f"Server:   {server_url}")
    print(f"Source:   {SOURCE_URL}")
    print(f"Target:   {clone_dir}")
    print(f"Config:   {CONFIG_PATH}")
    print(f"Service:  {UNIT_NAME}")
    print(f"Marker:   {marker_path(INSTALL_NAME)}")
    print()
    print("Preflight:")
    for label, ok, hint in tool_statuses:
        display = (
            "systemctl --user available"
            if label == "systemctl --user"
            else f"{label} found"
        )
        if ok:
            print(f"  ✓ {display}")
        else:
            print(f"  ✗ {label} missing")
            if hint:
                print(f"    {hint}")
    if tmux_present:
        print("  ✓ tmux found")
    else:
        print("  ✗ tmux missing")
        print(
            "    warning: tmux not detected on PATH; observer will start when tmux is launched"
        )
    print()
    print("Plan:")
    print(f"  would clone {SOURCE_URL} into {clone_dir}")
    print(f"  would create observer registration '{name}'")
    print(f"  would write {CONFIG_PATH}")
    print("  would run: make install-service")
    print("  would wait up to 30s for observer status")
    print(f"  would write marker {marker_path(INSTALL_NAME)}")
    print()
    print("Summary:")
    print("  Key prefix: <not generated in dry-run>")
    print(f"  Logs:       journalctl --user -u {UNIT_NAME} -f")
    print(f"  Status:     sol observer status {name}")
    print()
    print("Dry-run complete; no files were written.")
