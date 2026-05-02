# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Linux observer installer."""

from __future__ import annotations

import datetime as dt
import json
import shutil
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

PLATFORM = "linux"
INSTALL_NAME = "solstone-linux"
SOURCE_URL = "https://github.com/solpbc/solstone-linux.git"
UNIT_NAME = "solstone-linux.service"
CONFIG_PATH = (
    Path.home() / ".local" / "share" / "solstone-linux" / "config" / "config.json"
)
OS_RELEASE_PATH = Path("/etc/os-release")
DEFAULT_CONFIG = {
    "server_url": "",
    "key": "",
    "stream": "",
    "segment_interval": 300,
    "sync_retry_delays": [5, 30, 120, 300],
    "sync_max_retries": 10,
    "cache_retention_days": 7,
}

DISTRO_PACKAGES = {
    "fedora": (
        [
            "python3-gobject",
            "gtk4",
            "gstreamer1-plugins-base",
            "gstreamer1-plugin-pipewire",
            "pipewire-gstreamer",
            "alsa-lib-devel",
            "pulseaudio-utils",
            "pipewire-pulseaudio",
            "xdg-desktop-portal",
            "pipx",
        ],
        "sudo dnf install python3-gobject gtk4 gstreamer1-plugins-base gstreamer1-plugin-pipewire pipewire-gstreamer alsa-lib-devel pulseaudio-utils pipewire-pulseaudio xdg-desktop-portal pipx",
        "rpm",
    ),
    "debian-ubuntu": (
        [
            "python3-gi",
            "gir1.2-gdk-4.0",
            "gir1.2-gtk-4.0",
            "gstreamer1.0-pipewire",
            "libasound2-dev",
            "pulseaudio-utils",
            "pipewire-pulse",
            "xdg-desktop-portal",
            "pipx",
        ],
        "sudo apt install python3-gi gir1.2-gdk-4.0 gir1.2-gtk-4.0 gstreamer1.0-pipewire libasound2-dev pulseaudio-utils pipewire-pulse xdg-desktop-portal pipx",
        "dpkg",
    ),
    "arch": (
        [
            "python-gobject",
            "gtk4",
            "gstreamer",
            "gst-plugin-pipewire",
            "libpulse",
            "alsa-lib",
            "xdg-desktop-portal",
            "pipx",
        ],
        "sudo pacman -S python-gobject gtk4 gstreamer gst-plugin-pipewire libpulse alsa-lib xdg-desktop-portal pipx",
        "pacman",
    ),
    "opensuse": (
        [
            "python3-gobject",
            "python3-gobject-Gdk",
            "typelib-1_0-Gtk-4_0",
            "gtk4-tools",
            "gstreamer-plugins-base",
            "gstreamer-plugin-pipewire",
            "pipewire-pulseaudio",
            "pulseaudio-utils",
            "alsa-devel",
            "xdg-desktop-portal",
            "python3-pipx",
        ],
        "sudo zypper install python3-gobject python3-gobject-Gdk typelib-1_0-Gtk-4_0 \\\n  gtk4-tools gstreamer-plugins-base gstreamer-plugin-pipewire \\\n  pipewire-pulseaudio pulseaudio-utils alsa-devel \\\n  xdg-desktop-portal python3-pipx",
        "rpm",
    ),
}


def detect_distro() -> str | None:
    """Detect the Linux package family."""
    values = _read_os_release()
    distro_id = values.get("id", "")
    id_like = values.get("id_like", "").split()
    for candidate in [distro_id]:
        mapped = _map_os_release_id(candidate)
        if mapped:
            return mapped
    for candidate in id_like:
        mapped = _map_os_release_like(candidate)
        if mapped:
            return mapped
    for binary, distro in (
        ("zypper", "opensuse"),
        ("dnf", "fedora"),
        ("dpkg", "debian-ubuntu"),
        ("pacman", "arch"),
        ("rpm", "fedora"),
    ):
        if shutil.which(binary):
            return distro
    return None


class LinuxDriver:
    """Install and manage the Linux observer service."""

    def run(self, args) -> int:
        distro = detect_distro()
        if distro is None:
            raise InstallError(
                "unsupported Linux distribution",
                hint="install the observer dependencies from solstone-linux/INSTALL.md",
            )
        if distro not in DISTRO_PACKAGES:
            raise InstallError("unsupported Linux distribution")

        server_url = default_server_url(args.server_url)
        name = default_stream("linux", args.name)
        clone_dir = xdg_install_dir(INSTALL_NAME)
        marker = read_marker(INSTALL_NAME)
        if marker and marker.get("name") != name and not args.force:
            raise InstallError(
                f"{INSTALL_NAME} is already installed for {marker.get('name')}",
                hint="rerun with --force to replace the existing install marker",
            )

        tool_statuses = _check_tools()
        package_statuses = _check_packages(distro)
        if args.dry_run:
            if not args.json_output:
                _print_dry_run(
                    name,
                    server_url,
                    clone_dir,
                    distro,
                    tool_statuses,
                    package_statuses,
                )
            if args.json_output:
                emit_json(
                    _result(name, server_url, clone_dir, "planned", None, None, True)
                )
            return 0

        _raise_for_preflight(tool_statuses, package_statuses, distro)
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


def _read_os_release() -> dict[str, str]:
    try:
        lines = OS_RELEASE_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in lines:
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.lower()] = value.strip().strip('"').strip("'").lower()
    return values


def _map_os_release_id(value: str) -> str | None:
    if value == "fedora":
        return "fedora"
    if value in {"debian", "ubuntu", "pop", "linuxmint"}:
        return "debian-ubuntu"
    if value in {"arch", "manjaro"}:
        return "arch"
    if value in {"opensuse", "opensuse-leap", "opensuse-tumbleweed", "sles", "suse"}:
        return "opensuse"
    return None


def _map_os_release_like(value: str) -> str | None:
    if value in {"fedora", "rhel", "centos"}:
        return "fedora"
    if value in {"debian", "ubuntu"}:
        return "debian-ubuntu"
    if value == "arch":
        return "arch"
    if value in {"suse", "opensuse"}:
        return "opensuse"
    return None


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


_PIPX_PACKAGE_NAMES = {"pipx", "python3-pipx"}


def _check_packages(distro: str) -> list[tuple[str, bool]]:
    packages, _install_command, query_method = DISTRO_PACKAGES[distro]
    pipx_on_path = run_probe(["sh", "-c", "command -v pipx"]).returncode == 0
    result = []
    for package in packages:
        if package in _PIPX_PACKAGE_NAMES and pipx_on_path:
            result.append((package, True))
            continue
        if query_method == "dpkg":
            cmd = ["dpkg", "-s", package]
        elif query_method == "pacman":
            cmd = ["pacman", "-Q", package]
        else:
            cmd = ["rpm", "-q", package]
        result.append((package, run_probe(cmd).returncode == 0))
    return result


def _raise_for_preflight(
    tool_statuses: list[tuple[str, bool, str | None]],
    package_statuses: list[tuple[str, bool]],
    distro: str,
) -> None:
    missing_tools = [(label, hint) for label, ok, hint in tool_statuses if not ok]
    missing_packages = [package for package, ok in package_statuses if not ok]
    if missing_tools:
        label, hint = missing_tools[0]
        raise InstallError(f"missing required tool: {label}", hint=hint)
    if missing_packages:
        _packages, install_command, _query_method = DISTRO_PACKAGES[distro]
        raise InstallError(
            "missing required system packages: " + ", ".join(missing_packages),
            hint=install_command,
        )


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
    distro: str,
    tool_statuses: list[tuple[str, bool, str | None]],
    package_statuses: list[tuple[str, bool]],
) -> None:
    print("Dry-run: would install solstone-linux observer")
    print()
    print("Platform: linux")
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
    print(f"  ✓ distro detected: {distro}")
    _packages, install_command, _query_method = DISTRO_PACKAGES[distro]
    for package, ok in package_statuses:
        if ok:
            print(f"  ✓ package {package} installed")
        else:
            print(f"  ✗ package {package} missing")
            print(f"    {install_command}")
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
