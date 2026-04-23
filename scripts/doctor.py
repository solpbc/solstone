#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Pre-install diagnostics for solstone.

Runs a fixed battery of blocker and advisory checks using only the Python
standard library so a fresh clone can be diagnosed before `uv sync`. Exit code
`0` means no blocker failed; exit code `1` means at least one blocker-severity
check failed.

Decision log:
- uv floor: 0.7.12 — `uv.lock` revision=3 requires >= 0.7.12 per
  astral-sh/uv#15220.
- disk threshold: 10 GiB — measured `.venv`=7.88 GiB + playwright=0.61 GiB +
  uv-cache first-install growth ~1 GiB + buffer.
- Makefile UV-guard strategy: MAKECMDGOALS filter; prep verified the
  doctor-only matrix on GNU make.
- Ramon triage docs are absent in this worktree; the battery follows the task
  spec directly.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

ROOT = Path(__file__).resolve().parent.parent
MIN_UV = (0, 7, 12)
MIN_FREE_GIB = 10.0

Severity = Literal["blocker", "advisory"]
Status = Literal["ok", "fail", "warn", "skip"]
Platform = Literal["linux", "darwin"]


@dataclass(frozen=True)
class Args:
    verbose: bool
    json: bool
    port: int


@dataclass(frozen=True)
class Check:
    name: str
    severity: Severity
    platforms: tuple[Platform, ...]


@dataclass(frozen=True)
class CheckResult:
    name: str
    severity: Severity
    status: Status
    detail: str
    fix: str | None
    platform: str | None = None


@dataclass(frozen=True)
class ProbeOutput:
    stdout: str
    stderr: str
    returncode: int


def platform_tag() -> Platform:
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def make_result(
    check: Check,
    status: Status,
    detail: str,
    fix: str | None = None,
    *,
    platform: str | None = None,
) -> CheckResult:
    return CheckResult(
        name=check.name,
        severity=check.severity,
        status=status,
        detail=detail,
        fix=fix,
        platform=platform,
    )


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def version_text(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def parse_version(text: str) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def compare_versions(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
    if left < right:
        return -1
    if left > right:
        return 1
    return 0


def unexpected_output_result(
    check: Check,
    output: str,
    *,
    fix: str | None = None,
) -> CheckResult:
    snippet = truncate(output or "<empty>", 80)
    return make_result(
        check,
        "fail",
        f"probe returned unexpected output: {snippet}",
        fix,
    )


def command_text(cmd: Sequence[str]) -> str:
    return " ".join(cmd)


def run_probe(
    check: Check,
    cmd: Sequence[str],
    *,
    timeout: float,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    ok_returncodes: tuple[int, ...] = (0,),
    allow_nonzero: bool = False,
    allow_empty_stdout: bool = False,
    fix: str | None = None,
) -> ProbeOutput | CheckResult:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        completed = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            check=False,
        )
    except FileNotFoundError:
        return make_result(check, "fail", f"probe command not found: {cmd[0]}", fix)
    except subprocess.TimeoutExpired:
        return make_result(
            check,
            "fail",
            f"probe timed out after {timeout:g}s: {command_text(cmd)}",
            fix,
        )
    except OSError as exc:
        return make_result(
            check,
            "fail",
            f"probe failed: {type(exc).__name__}: {exc}",
            fix,
        )

    if completed.returncode not in ok_returncodes and not allow_nonzero:
        detail = completed.stderr.strip() or completed.stdout.strip() or "<empty>"
        return make_result(
            check,
            "fail",
            f"probe exited {completed.returncode}: {truncate(detail, 80)}",
            fix,
        )

    if not allow_empty_stdout and not completed.stdout.strip():
        return unexpected_output_result(
            check,
            completed.stderr.strip() or completed.stdout.strip(),
            fix=fix,
        )

    return ProbeOutput(
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def python_version_check(args: Args) -> CheckResult:
    del args
    check = CHECK_MAP["python_version"]
    pyproject = ROOT / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError as exc:
        return make_result(
            check,
            "fail",
            f"could not read {pyproject.name}: {type(exc).__name__}: {exc}",
            "install Python >=3.10, then retry",
        )
    match = re.search(r'^requires-python\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        return make_result(
            check,
            "fail",
            "could not parse requires-python from pyproject.toml",
            "install Python >=3.10, then retry",
        )
    spec = match.group(1)
    min_match = re.search(r">=\s*(\d+)\.(\d+)(?:\.(\d+))?", spec)
    if not min_match:
        return make_result(
            check,
            "fail",
            f"unsupported requires-python specifier: {spec}",
            "install Python >=3.10, then retry",
        )
    minimum = (
        int(min_match.group(1)),
        int(min_match.group(2)),
        int(min_match.group(3) or 0),
    )
    current = sys.version_info[:3]
    if compare_versions(current, minimum) < 0:
        return make_result(
            check,
            "fail",
            f"python {version_text(current)} does not satisfy {spec}",
            "install Python >=3.10, then `rm -rf .venv .installed && make install`",
        )
    return make_result(
        check,
        "ok",
        f"python {version_text(current)} satisfies {spec}",
    )


def uv_installed_check(args: Args) -> CheckResult:
    del args
    check = CHECK_MAP["uv_installed"]
    fix = "curl -LsSf https://astral.sh/uv/install.sh | sh"
    probe = run_probe(check, ["uv", "--version"], timeout=0.5, fix=fix)
    if isinstance(probe, CheckResult):
        return probe
    version = parse_version(probe.stdout)
    if version is None:
        return unexpected_output_result(check, probe.stdout, fix=fix)
    if compare_versions(version, MIN_UV) < 0:
        return make_result(
            check,
            "fail",
            f"uv {version_text(version)} is older than required {version_text(MIN_UV)}",
            fix,
        )
    return make_result(
        check,
        "ok",
        f"uv {version_text(version)} >= {version_text(MIN_UV)}",
    )


def venv_consistent_check(args: Args) -> CheckResult:
    del args
    check = CHECK_MAP["venv_consistent"]
    python_bin = ROOT / ".venv" / "bin" / "python"
    expected = (ROOT / ".venv").resolve()
    if not python_bin.exists():
        return make_result(
            check,
            "skip",
            ".venv absent; run make install",
        )
    probe = run_probe(
        check,
        [str(python_bin), "-c", "import sys; print(sys.prefix)"],
        timeout=0.5,
        fix="rm -rf .venv .installed && make install",
    )
    if isinstance(probe, CheckResult):
        return probe
    prefix_text = probe.stdout.strip()
    if not prefix_text:
        return unexpected_output_result(
            check,
            probe.stdout,
            fix="rm -rf .venv .installed && make install",
        )
    actual = Path(prefix_text).resolve()
    if actual != expected:
        return make_result(
            check,
            "fail",
            f".venv points at {actual}, expected {expected}",
            "rm -rf .venv .installed && make install",
        )
    return make_result(check, "ok", f".venv points at this repo ({expected})")


def sol_importable_check(args: Args) -> CheckResult:
    del args
    check = CHECK_MAP["sol_importable"]
    python_bin = ROOT / ".venv" / "bin" / "python"
    fix = "rm -rf .venv .installed && make install"
    if not python_bin.exists():
        return make_result(check, "skip", ".venv absent; run make install")
    probe = run_probe(
        check,
        [str(python_bin), "-c", "import sol"],
        cwd=Path("/"),
        timeout=2.0,
        allow_nonzero=True,
        allow_empty_stdout=True,
        fix=fix,
    )
    if isinstance(probe, CheckResult):
        return probe
    if probe.returncode == 0:
        return make_result(check, "ok", "import sol succeeded outside repo cwd")
    stderr = probe.stderr.strip()
    if "ModuleNotFoundError: No module named 'sol'" in stderr:
        return make_result(
            check,
            "fail",
            "ModuleNotFoundError: No module named 'sol'",
            fix,
        )
    first_line = next((line for line in stderr.splitlines() if line.strip()), "")
    detail = truncate(
        first_line or f"import sol failed with exit {probe.returncode}", 120
    )
    return make_result(check, "fail", detail, fix)


def npx_on_path_check(args: Args) -> CheckResult:
    del args
    check = CHECK_MAP["npx_on_path"]
    npx = shutil.which("npx")
    if npx is None:
        return make_result(
            check,
            "fail",
            "npx not found on PATH",
            "install Node/npm so `npx` is on PATH, then rerun doctor",
        )
    return make_result(check, "ok", f"npx on PATH at {npx}")


def npx_non_interactive_check(args: Args) -> CheckResult:
    del args
    check = CHECK_MAP["npx_non_interactive"]
    fix = "repair npm/npx, then rerun `CI=true npx --yes --version`"
    probe = run_probe(
        check,
        ["npx", "--yes", "--version"],
        timeout=2.0,
        env={"CI": "true"},
        fix=fix,
    )
    if isinstance(probe, CheckResult):
        return probe
    if not probe.stdout.strip():
        return unexpected_output_result(check, probe.stdout, fix=fix)
    return make_result(check, "ok", "npx --yes is non-interactive")


def resolve_alias_target() -> Path | None:
    alias = Path.home() / ".local" / "bin" / "sol"
    if not alias.is_symlink():
        return None
    target = Path(os.readlink(alias))
    if not target.is_absolute():
        target = alias.parent / target
    return target.resolve()


def resolve_darwin_exe(
    check: Check,
    pid: int,
    *,
    expected_repo_sol: Path,
    alias_target: Path | None,
) -> Path | CheckResult:
    probe = run_probe(
        check,
        ["lsof", "-p", str(pid), "-Fn"],
        timeout=1.0,
        allow_empty_stdout=False,
        fix=f"kill {pid}  # or run 'sol service stop' if this is your install",
    )
    if isinstance(probe, CheckResult):
        return probe
    paths: list[Path] = []
    for line in probe.stdout.splitlines():
        if not line.startswith("n"):
            continue
        value = line[1:].strip()
        if not value.startswith("/"):
            continue
        paths.append(Path(value).resolve())
    if not paths:
        return make_result(
            check,
            "fail",
            f"could not verify ownership (pid={pid}): no executable path from lsof",
            f"kill {pid}  # or run 'sol service stop' if this is your install",
        )
    for candidate in paths:
        if candidate == expected_repo_sol or candidate == alias_target:
            return candidate
    sol_paths = [candidate for candidate in paths if candidate.name == "sol"]
    if len(sol_paths) == 1:
        return sol_paths[0]
    return make_result(
        check,
        "fail",
        f"could not verify ownership (pid={pid}): ambiguous executable paths",
        f"kill {pid}  # or run 'sol service stop' if this is your install",
    )


def port_5015_free_check(args: Args) -> CheckResult:
    check = CHECK_MAP["port_5015_free"]
    # In a git worktree (hopper lode, personal worktree) the host's port state
    # is not this worktree's concern — the worktree will never run its own
    # service. Skip, matching the pattern in stale_alias_symlink_check.
    try:
        alias_state_cls, check_alias_fn = import_install_guard()
    except Exception:
        pass
    else:
        state, _ = check_alias_fn(ROOT)
        if state is alias_state_cls.WORKTREE:
            return make_result(
                check,
                "skip",
                "git worktree; run doctor from the primary clone",
            )
    port = args.port
    if shutil.which("lsof") is None:
        return make_result(
            check,
            "skip",
            "lsof not available; cannot probe port ownership",
        )
    probe = run_probe(
        check,
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpn"],
        timeout=1.0,
        ok_returncodes=(0, 1),
        allow_empty_stdout=True,
        fix="kill <pid>  # or run 'sol service stop' if this is your install",
    )
    if isinstance(probe, CheckResult):
        return probe
    pids = [
        line[1:].strip() for line in probe.stdout.splitlines() if line.startswith("p")
    ]
    if not pids:
        return make_result(check, "ok", f"port {port} is free")
    pid_text = pids[0]
    try:
        pid = int(pid_text)
    except ValueError:
        return unexpected_output_result(
            check,
            probe.stdout,
            fix="kill <pid>  # or run 'sol service stop' if this is your install",
        )
    expected_repo_sol = (ROOT / ".venv" / "bin" / "sol").resolve()
    alias_target = resolve_alias_target()
    if platform_tag() == "darwin":
        resolved = resolve_darwin_exe(
            check,
            pid,
            expected_repo_sol=expected_repo_sol,
            alias_target=alias_target,
        )
        if isinstance(resolved, CheckResult):
            return resolved
        exe_path = resolved
    else:
        try:
            exe_path = Path(os.readlink(f"/proc/{pid}/exe")).resolve()
        except OSError as exc:
            return make_result(
                check,
                "fail",
                f"could not verify ownership (pid={pid}): {type(exc).__name__}: {exc}",
                f"kill {pid}  # or run 'sol service stop' if this is your install",
            )
    if exe_path == expected_repo_sol:
        return make_result(
            check,
            "ok",
            f"port {port} owned by this repo's solstone ({exe_path})",
        )
    if alias_target is not None and exe_path == alias_target:
        return make_result(
            check,
            "ok",
            f"port {port} owned by installed solstone ({exe_path})",
        )
    return make_result(
        check,
        "fail",
        f"port {port} held by pid {pid} ({exe_path})",
        f"kill {pid}  # or run 'sol service stop' if this is your install",
    )


def disk_space_check(args: Args) -> CheckResult:
    del args
    check = CHECK_MAP["disk_space"]
    usage = shutil.disk_usage(ROOT)
    free_gib = usage.free / (1024**3)
    if free_gib < MIN_FREE_GIB:
        return make_result(
            check,
            "warn",
            f"only {free_gib:.1f} GiB free on the repo filesystem (<{MIN_FREE_GIB:.0f} GiB)",
            "free disk on the repo filesystem before `make install`",
        )
    return make_result(
        check,
        "ok",
        f"{free_gib:.1f} GiB free (>= {MIN_FREE_GIB:.0f} GiB)",
    )


def config_dir_readable_check(args: Args) -> CheckResult:
    del args
    check = CHECK_MAP["config_dir_readable"]
    home = Path.home()
    if not home.exists():
        return make_result(
            check,
            "fail",
            f"home directory does not exist: {home}",
            f"fix ownership/permissions of {home}",
        )
    required_access = os.R_OK | os.W_OK | os.X_OK
    if not os.access(home, required_access):
        return make_result(
            check,
            "fail",
            f"home directory is not readable and writable: {home}",
            f"fix ownership/permissions of {home}",
        )
    current_platform = platform_tag()
    if current_platform == "darwin":
        config_dir = home / "Library" / "LaunchAgents"
    else:
        config_dir = home / ".config"
    if config_dir.exists() and not os.access(config_dir, required_access):
        return make_result(
            check,
            "fail",
            f"service config directory is not writable: {config_dir}",
            f"fix ownership/permissions of {config_dir}",
        )
    if config_dir.exists():
        detail = f"home and service config dir are writable ({config_dir})"
    else:
        detail = f"home is writable; install will create {config_dir}"
    return make_result(check, "ok", detail)


def import_install_guard() -> tuple[object, object]:
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("think.install_guard")
    return module.AliasState, module.check_alias


def stale_alias_symlink_check(args: Args) -> CheckResult:
    del args
    check = CHECK_MAP["stale_alias_symlink"]
    try:
        alias_state_cls, check_alias = import_install_guard()
    except Exception as exc:
        return make_result(
            check,
            "skip",
            f"could not import think.install_guard: {type(exc).__name__}: {exc}",
        )
    state, other = check_alias(ROOT)
    worktree = alias_state_cls.WORKTREE
    absent = alias_state_cls.ABSENT
    owned = alias_state_cls.OWNED
    cross_repo = alias_state_cls.CROSS_REPO
    dangling = alias_state_cls.DANGLING
    not_symlink = alias_state_cls.NOT_SYMLINK
    if state is worktree:
        return make_result(
            check,
            "skip",
            "git worktree; run doctor from the primary clone",
        )
    if state in {absent, owned}:
        return make_result(
            check,
            "ok",
            "sol alias absent or owned by this repo",
        )
    if state is cross_repo:
        detail = f"~/.local/bin/sol points at another repo ({other})"
    elif state is dangling:
        detail = f"~/.local/bin/sol is dangling ({other})"
    elif state is not_symlink:
        detail = "~/.local/bin/sol exists but is not a symlink"
    else:
        detail = f"unexpected alias state: {state}"
    return make_result(
        check,
        "fail",
        detail,
        "run `make uninstall-service` from the installed repo, or remove `~/.local/bin/sol` manually if the repo is gone",
    )


def macos_firewall_localhost_check(args: Args) -> CheckResult:
    del args
    check = CHECK_MAP["macos_firewall_localhost"]
    if platform_tag() != "darwin":
        return make_result(check, "skip", "not supported on linux", platform="linux")
    tool = "/usr/libexec/ApplicationFirewall/socketfilterfw"
    if not Path(tool).exists():
        return make_result(check, "skip", "socketfilterfw not available")
    global_probe = run_probe(
        check,
        [tool, "--getglobalstate"],
        timeout=1.0,
        fix="sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setblockall off",
    )
    if isinstance(global_probe, CheckResult):
        return global_probe
    global_text = global_probe.stdout.lower()
    if "disabled" in global_text or "state = 0" in global_text:
        return make_result(
            check,
            "ok",
            "firewall settings will not block localhost service access",
        )
    if "enabled" not in global_text and "state = 1" not in global_text:
        return unexpected_output_result(
            check,
            global_probe.stdout,
            fix="sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setblockall off",
        )
    block_probe = run_probe(
        check,
        [tool, "--getblockall"],
        timeout=1.0,
        fix="sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setblockall off",
    )
    if isinstance(block_probe, CheckResult):
        return block_probe
    block_text = block_probe.stdout.lower()
    if "enabled" in block_text or "state = 1" in block_text:
        return make_result(
            check,
            "warn",
            "firewall enabled with block-all incoming; localhost service access may fail",
            "sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setblockall off",
        )
    if "disabled" in block_text or "state = 0" in block_text:
        return make_result(
            check,
            "ok",
            "firewall enabled but block-all incoming is off",
        )
    return unexpected_output_result(
        check,
        block_probe.stdout,
        fix="sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setblockall off",
    )


def launchd_stale_plist_check(args: Args) -> CheckResult:
    del args
    check = CHECK_MAP["launchd_stale_plist"]
    if platform_tag() != "darwin":
        return make_result(check, "skip", "not supported on linux", platform="linux")
    plist_path = Path.home() / "Library" / "LaunchAgents" / "org.solpbc.solstone.plist"
    if not plist_path.exists():
        return make_result(check, "skip", "launchd plist absent")
    try:
        with plist_path.open("rb") as handle:
            data = plistlib.load(handle)
    except Exception as exc:
        return make_result(
            check,
            "fail",
            f"could not parse plist: {type(exc).__name__}: {exc}",
            "rm ~/Library/LaunchAgents/org.solpbc.solstone.plist && make install-service",
        )
    program_arguments = data.get("ProgramArguments")
    if not isinstance(program_arguments, list) or not program_arguments:
        return make_result(
            check,
            "fail",
            "plist is missing ProgramArguments[0]",
            "rm ~/Library/LaunchAgents/org.solpbc.solstone.plist && make install-service",
        )
    executable = Path(str(program_arguments[0]))
    if not executable.exists():
        return make_result(
            check,
            "fail",
            f"plist points to missing executable: {executable}",
            "rm ~/Library/LaunchAgents/org.solpbc.solstone.plist && make install-service",
        )
    return make_result(check, "ok", f"launchd plist target exists ({executable})")


def screen_recording_permission_check(args: Args) -> CheckResult:
    del args
    check = CHECK_MAP["screen_recording_permission"]
    if platform_tag() != "darwin":
        return make_result(check, "skip", "not supported on linux", platform="linux")
    return make_result(
        check,
        "skip",
        "no adopted non-prompting probe for CLI-scoped macOS TCC state",
        "System Settings → Privacy & Security → Screen Recording / Screen & System Audio Recording",
    )


def microphone_permission_check(args: Args) -> CheckResult:
    del args
    check = CHECK_MAP["microphone_permission"]
    if platform_tag() != "darwin":
        return make_result(check, "skip", "not supported on linux", platform="linux")
    return make_result(
        check,
        "skip",
        "no adopted non-prompting probe for CLI-scoped macOS TCC state",
        "System Settings → Privacy & Security → Microphone",
    )


CHECKS: list[tuple[Check, Callable[[Args], CheckResult]]] = [
    (Check("python_version", "blocker", ("linux", "darwin")), python_version_check),
    (Check("uv_installed", "blocker", ("linux", "darwin")), uv_installed_check),
    (Check("venv_consistent", "blocker", ("linux", "darwin")), venv_consistent_check),
    (Check("sol_importable", "blocker", ("linux", "darwin")), sol_importable_check),
    (Check("npx_on_path", "blocker", ("linux", "darwin")), npx_on_path_check),
    (
        Check("npx_non_interactive", "advisory", ("linux", "darwin")),
        npx_non_interactive_check,
    ),
    (Check("port_5015_free", "blocker", ("linux", "darwin")), port_5015_free_check),
    (Check("disk_space", "advisory", ("linux", "darwin")), disk_space_check),
    (
        Check("config_dir_readable", "blocker", ("linux", "darwin")),
        config_dir_readable_check,
    ),
    (
        Check("stale_alias_symlink", "blocker", ("linux", "darwin")),
        stale_alias_symlink_check,
    ),
    (
        Check("macos_firewall_localhost", "advisory", ("darwin",)),
        macos_firewall_localhost_check,
    ),
    (
        Check("launchd_stale_plist", "advisory", ("darwin",)),
        launchd_stale_plist_check,
    ),
    (
        Check("screen_recording_permission", "advisory", ("darwin",)),
        screen_recording_permission_check,
    ),
    (
        Check("microphone_permission", "advisory", ("darwin",)),
        microphone_permission_check,
    ),
]

CHECK_MAP = {check.name: check for check, _func in CHECKS}


def parse_args(argv: Sequence[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(
        prog="doctor",
        description="Run pre-install diagnostics for solstone.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print every check result"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--port", type=int, default=5015, help="port to probe (default: 5015)"
    )
    namespace = parser.parse_args(argv)
    return Args(
        verbose=namespace.verbose,
        json=namespace.json,
        port=namespace.port,
    )


def run_checks(args: Args) -> list[CheckResult]:
    current_platform = platform_tag()
    results: list[CheckResult] = []
    for check, func in CHECKS:
        if current_platform not in check.platforms:
            results.append(
                make_result(
                    check,
                    "skip",
                    f"not supported on {current_platform}",
                    platform=current_platform,
                )
            )
            continue
        results.append(func(args))
    return results


def print_result_line(result: CheckResult) -> None:
    label = result.status.upper()
    print(f"  {label} {result.name} — {result.detail}")
    if result.fix:
        print(f"    → {result.fix}")


def summary_counts(results: Sequence[CheckResult]) -> dict[str, int]:
    return {
        "total": len(results),
        "failed": sum(1 for result in results if result.status == "fail"),
        "warnings": sum(1 for result in results if result.status == "warn"),
        "skipped": sum(1 for result in results if result.status == "skip"),
    }


def emit_text(results: Sequence[CheckResult], *, verbose: bool) -> None:
    if verbose:
        for result in results:
            print_result_line(result)
    else:
        for result in results:
            if result.status in {"fail", "warn"}:
                print_result_line(result)
    summary = summary_counts(results)
    print(
        "doctor: "
        f"{summary['total']} checks, "
        f"{summary['failed']} failed, "
        f"{summary['warnings']} warnings, "
        f"{summary['skipped']} skipped"
    )


def emit_json(results: Sequence[CheckResult]) -> None:
    payload = {
        "checks": [
            {
                "name": result.name,
                "severity": result.severity,
                "status": result.status,
                "detail": result.detail,
                "fix": result.fix,
            }
            for result in results
        ],
        "summary": summary_counts(results),
    }
    print(json.dumps(payload))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_checks(args)
    if args.json:
        emit_json(results)
    else:
        emit_text(results, verbose=args.verbose)
    blocker_failed = any(
        result.severity == "blocker" and result.status == "fail" for result in results
    )
    return 1 if blocker_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
