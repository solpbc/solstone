# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""User-runtime setup orchestration for solstone."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from solstone.think.user_config import (
    config_path,
    default_journal,
    read_user_config,
    write_user_config,
)
from solstone.think.utils import get_project_root
from solstone.think.utils import is_source_checkout as source_checkout

TOTAL_STEPS = 6
MANIFEST_SCHEMA_VERSION = 1
HEALTH_ATTEMPTS = 20
HEALTH_SLEEP_SECONDS = 1.0
DOCTOR_TIMEOUT_SECONDS = 30

StepStatus = Literal["ok", "skipped", "failed"]


class SetupMode(Enum):
    INTERACTIVE = "interactive"
    NON_INTERACTIVE = "non_interactive"
    DRY_RUN = "dry_run"
    EXPLAIN = "explain"


@dataclass
class SetupContext:
    mode: SetupMode
    project_root: Path
    is_source_checkout: bool
    journal_path: Path
    journal_source: str
    config_path: Path
    manifest_path: Path
    port: int
    port_source: str
    port_supplied: bool
    step_timeout_seconds: int
    variant: str
    variant_source: str
    yes: bool
    skip_models: bool
    skip_skills: bool
    skip_service: bool
    accept_existing_journal: bool
    force: bool
    stdin_is_tty: bool
    stdout_is_tty: bool
    args_resolved: dict[str, object]
    doctor_advisories: list[dict[str, Any]]


@dataclass(frozen=True)
class StepResult:
    name: str
    status: StepStatus
    paths: list[str]
    started_at: str
    finished_at: str
    error: dict[str, object] | None
    reason: str | None = None


class SetupDeadEnd(Exception):
    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


def _journal_arg(value: str) -> Path:
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("--journal must not be empty")
    path = Path(value).expanduser()
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise argparse.ArgumentTypeError(
            f"--journal could not be resolved: {exc}"
        ) from exc
    if resolved == Path.cwd().resolve():
        raise argparse.ArgumentTypeError("--journal must not be empty")
    return path


def _port_arg(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--port must be in 1024-65535 (got {value})"
        ) from exc
    if not 1024 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"--port must be in 1024-65535 (got {port})")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sol setup",
        description="Set up solstone user-runtime artifacts and start the service.",
    )
    parser.add_argument(
        "--journal",
        metavar="PATH",
        type=_journal_arg,
        default=None,
        help="journal directory to persist in ~/.config/solstone/config.toml",
    )
    parser.add_argument(
        "--port",
        metavar="INT",
        type=_port_arg,
        default=5015,
        help="convey service port (default: 5015)",
    )
    parser.add_argument(
        "--variant",
        choices=("auto", "cpu", "cuda", "coreml"),
        default="auto",
        help="Parakeet model/runtime variant passed to sol install-models (default: auto)",
    )
    parser.add_argument(
        "--step-timeout-seconds",
        metavar="INT",
        type=int,
        default=1800,
        help=(
            "timeout for model, skill, and wrapper steps in seconds "
            "(default: 1800; doctor uses a separate 30s timeout)"
        ),
    )
    parser.add_argument(
        "-y",
        "--yes",
        "--non-interactive",
        dest="yes",
        action="store_true",
        help="run without prompts; fail with retry guidance when input is required",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved plan and commands without changing files or services",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print the setup steps and resolved defaults without running them",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="skip local model installation",
    )
    parser.add_argument(
        "--skip-skills",
        action="store_true",
        help="skip Claude Code skill installation",
    )
    parser.add_argument(
        "--skip-service",
        action="store_true",
        help="skip service installation, start, and health check",
    )
    parser.add_argument(
        "--accept-existing-journal",
        action="store_true",
        help="allow setup to use a non-empty existing journal directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run all steps unconditionally",
    )
    return parser


def resolve_mode(args: argparse.Namespace) -> SetupMode:
    stdin_is_tty = sys.stdin.isatty()
    stdout_is_tty = sys.stdout.isatty()

    if args.explain:
        return SetupMode.EXPLAIN
    if args.dry_run:
        return SetupMode.DRY_RUN
    if args.yes:
        return SetupMode.NON_INTERACTIVE
    if stdin_is_tty and stdout_is_tty:
        return SetupMode.INTERACTIVE
    return SetupMode.NON_INTERACTIVE


def resolve_context(args: argparse.Namespace, raw_argv: list[str]) -> SetupContext:
    mode = resolve_mode(args)
    project_root = Path(get_project_root())
    is_source_checkout = source_checkout()
    journal_path, journal_source = resolve_journal_path(args)
    cfg_path = config_path()
    manifest_path = journal_path / "health" / "setup-state.json"
    port_supplied = arg_supplied(raw_argv, "--port")
    step_timeout_supplied = arg_supplied(raw_argv, "--step-timeout-seconds")
    variant_supplied = arg_supplied(raw_argv, "--variant")

    args_resolved: dict[str, object] = {
        "journal": {
            "value": str(journal_path),
            "source": journal_source,
        },
        "port": {
            "value": args.port,
            "source": "cli" if port_supplied else "default",
        },
        "step_timeout_seconds": {
            "value": args.step_timeout_seconds,
            "source": "cli" if step_timeout_supplied else "default",
        },
        "variant": {
            "value": args.variant,
            "source": "cli" if variant_supplied else "default",
        },
        "yes": {"value": bool(args.yes), "source": "cli" if args.yes else "default"},
        "force": {
            "value": bool(args.force),
            "source": "cli" if args.force else "default",
        },
        "dry_run": {
            "value": bool(args.dry_run),
            "source": "cli" if args.dry_run else "default",
        },
        "explain": {
            "value": bool(args.explain),
            "source": "cli" if args.explain else "default",
        },
        "skip_models": {
            "value": bool(args.skip_models),
            "source": "cli" if args.skip_models else "default",
        },
        "skip_skills": {
            "value": bool(args.skip_skills),
            "source": "cli" if args.skip_skills else "default",
        },
        "skip_service": {
            "value": bool(args.skip_service),
            "source": "cli" if args.skip_service else "default",
        },
        "accept_existing_journal": {
            "value": bool(args.accept_existing_journal),
            "source": "cli" if args.accept_existing_journal else "default",
        },
        "parakeet_onnx_variant_env": {
            "value": os.environ.get("PARAKEET_ONNX_VARIANT"),
            "source": "env",
        },
        "is_source_checkout": {
            "value": is_source_checkout,
            "source": "detected",
        },
    }

    ctx = SetupContext(
        mode=mode,
        project_root=project_root,
        is_source_checkout=is_source_checkout,
        journal_path=journal_path,
        journal_source=journal_source,
        config_path=cfg_path,
        manifest_path=manifest_path,
        port=args.port,
        port_source="cli" if port_supplied else "default",
        port_supplied=port_supplied,
        step_timeout_seconds=args.step_timeout_seconds,
        variant=args.variant,
        variant_source="cli" if variant_supplied else "default",
        yes=bool(args.yes),
        skip_models=bool(args.skip_models),
        skip_skills=bool(args.skip_skills),
        skip_service=bool(args.skip_service),
        accept_existing_journal=bool(args.accept_existing_journal),
        force=bool(args.force),
        stdin_is_tty=sys.stdin.isatty(),
        stdout_is_tty=sys.stdout.isatty(),
        args_resolved=args_resolved,
        doctor_advisories=[],
    )
    if ctx.journal_path.exists() and not ctx.journal_path.is_dir():
        dead_end_journal_is_file(ctx)
    return ctx


def resolve_journal_path(args: argparse.Namespace) -> tuple[Path, str]:
    if args.journal is not None:
        return expand_path(args.journal), "cli"

    env_path = os.environ.get("SOLSTONE_JOURNAL", "").strip()
    if env_path:
        return expand_path(env_path), "env"

    configured = read_user_config().get("journal", "").strip()
    if configured:
        return expand_path(configured), "config"

    return expand_path(default_journal()), "default"


def arg_supplied(raw_argv: list[str], flag: str) -> bool:
    return flag in raw_argv or any(item.startswith(f"{flag}=") for item in raw_argv)


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def absolute_string(path: Path) -> str:
    return str(path.expanduser().resolve())


def non_empty_journal(path: Path) -> bool:
    return path.is_dir() and (
        (path / "config").is_dir()
        or any(path.glob("*.jsonl"))
        or any(
            p.is_dir() and p.name.isdigit() and len(p.name) == 8 for p in path.iterdir()
        )
    )


def read_manifest(ctx: SetupContext) -> dict[str, Any] | None:
    try:
        return json.loads(ctx.manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class PriorRunStatus:
    state: str  # "none" | "clean" | "partial"
    timestamp: str | None
    failed_steps: tuple[str, ...]


def prior_run_status(ctx: SetupContext) -> PriorRunStatus:
    manifest = read_manifest(ctx)
    if manifest is None:
        return PriorRunStatus("none", None, ())
    steps = manifest.get("steps") or []
    failed = tuple(
        s.get("name", "<unknown>")
        for s in steps
        if s.get("status") not in ("ok", "skipped")
    )
    completed_at = manifest.get("completed_at")
    if completed_at and not failed:
        return PriorRunStatus("clean", completed_at, ())
    return PriorRunStatus("partial", manifest.get("started_at"), failed)


def prior_step_lookup(manifest: dict[str, Any]) -> dict[str, dict]:
    lookup = {}
    for step in manifest.get("steps", []):
        lookup[step["name"]] = step
    return lookup


def can_skip(prior_step: dict | None) -> bool:
    if prior_step is None or prior_step.get("status") != "ok":
        return False
    return all(Path(path).exists() for path in prior_step.get("paths", []))


def write_manifest(ctx: SetupContext, manifest: dict[str, Any]) -> None:
    try:
        ctx.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".tmp_setup_state",
            suffix=".json",
            dir=ctx.manifest_path.parent,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
                handle.write("\n")
            os.replace(tmp_path, ctx.manifest_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    except Exception as exc:
        logging.warning("could not write setup manifest: %s", exc)


def initial_manifest(ctx: SetupContext) -> dict[str, Any]:
    previous = read_manifest(ctx)
    if previous is not None:
        logging.debug("previous setup manifest found at %s", ctx.manifest_path)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "started_at": utc_now(),
        "completed_at": None,
        "mode": ctx.mode.value,
        "args_resolved": ctx.args_resolved,
        "steps": [],
    }


def append_step(manifest: dict[str, Any], result: StepResult) -> None:
    steps = manifest.setdefault("steps", [])
    steps.append(asdict(result))


def step_result(
    name: str,
    status: StepStatus,
    paths: list[Path | str],
    started_at: str,
    error: dict[str, object] | None = None,
    reason: str | None = None,
) -> StepResult:
    return StepResult(
        name=name,
        status=status,
        paths=[absolute_string(Path(path)) for path in paths],
        started_at=started_at,
        finished_at=utc_now(),
        error=error,
        reason=reason,
    )


def print_step_header(
    step_index: int, label: str, command: list[str] | None = None
) -> None:
    if command:
        print(
            f"[step {step_index}/{TOTAL_STEPS}] running {label}: {format_command(command)}"
        )
    else:
        print(f"[step {step_index}/{TOTAL_STEPS}] running {label}...")


def print_step_skipped(step_index: int, name: str, reason: str) -> None:
    print(f"[step {step_index}/{TOTAL_STEPS}] skipped {name}: {reason}")


def format_command(command: list[str]) -> str:
    return " ".join(command)


def run_inherited(command: list[str], *, timeout: float | None = None) -> int:
    if timeout is None:
        result = subprocess.run(command, stdout=None, stderr=None, check=False)
        return int(result.returncode)
    proc = subprocess.Popen(command, stdout=None, stderr=None)
    try:
        return int(proc.wait(timeout=timeout))
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise


def doctor_command(ctx: SetupContext) -> list[str]:
    return [
        sys.executable,
        "-m",
        "solstone.think.sol_cli",
        "doctor",
        "--json",
        "--port",
        str(ctx.port),
    ]


def install_models_command(ctx: SetupContext) -> list[str]:
    return [
        sys.executable,
        "-m",
        "solstone.think.sol_cli",
        "install-models",
        "--variant",
        ctx.variant,
    ]


def skills_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "solstone.think.sol_cli",
        "skills",
        "install",
        "--agent",
        "claude",
    ]


def wrapper_command() -> list[str]:
    return [sys.executable, "-m", "solstone.think.install_guard", "install"]


def service_install_command(ctx: SetupContext) -> list[str]:
    return [
        sys.executable,
        "-m",
        "solstone.think.sol_cli",
        "service",
        "install",
        "--port",
        str(ctx.port),
    ]


def step_doctor(ctx: SetupContext, step_index: int) -> StepResult:
    started_at = utc_now()
    command = doctor_command(ctx)
    print_step_header(step_index, "doctor", command)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=DOCTOR_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return step_result(
            "doctor",
            "failed",
            [],
            started_at,
            {
                "message": f"doctor timed out after {DOCTOR_TIMEOUT_SECONDS}s",
                "exit_code": 1,
            },
        )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(
                result.stderr,
                end="" if result.stderr.endswith("\n") else "\n",
                file=sys.stderr,
            )
        return step_result(
            "doctor",
            "failed",
            [],
            started_at,
            {"message": "doctor blocker failed", "exit_code": int(result.returncode)},
        )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return step_result(
            "doctor",
            "failed",
            [],
            started_at,
            {
                "message": f"doctor JSON parse failed: {exc}",
                "exit_code": 1,
            },
        )

    checks = payload.get("checks", [])
    if isinstance(checks, list):
        ctx.doctor_advisories[:] = [
            check
            for check in checks
            if isinstance(check, dict)
            and check.get("severity") == "advisory"
            and check.get("status") in ("warn", "fail")
        ]
    maybe_handle_port_in_use(ctx)
    print(f"[step {step_index}/{TOTAL_STEPS}] doctor passed")
    return step_result("doctor", "ok", [], started_at)


def _port_advisory_present(ctx: SetupContext) -> bool:
    for advisory in ctx.doctor_advisories:
        if advisory.get("name") == "port_5015_free":
            return True
        detail = str(advisory.get("detail", ""))
        if f"port {ctx.port}" in detail:
            return True
    return False


def maybe_handle_port_in_use(ctx: SetupContext) -> None:
    if ctx.skip_service or ctx.port_supplied:
        return
    if not _port_advisory_present(ctx):
        return
    if ctx.mode is SetupMode.NON_INTERACTIVE:
        dead_end_port_in_use(ctx)
        return
    prompt_port_choice(ctx)


def prompt_port_choice(ctx: SetupContext) -> None:
    advisory = next(
        (
            item
            for item in ctx.doctor_advisories
            if item.get("name") == "port_5015_free"
        ),
        None,
    )
    if advisory:
        detail = advisory.get("detail")
        if detail:
            print(f"  {detail}")
        fix = advisory.get("fix")
        if fix:
            print(f"  suggested fix: {fix}")
    print()
    print("  1) enter a different port")
    print("  2) proceed anyway on this port")
    print("  3) abort setup")
    while True:
        choice = input("choice [1/2/3]: ").strip()
        if choice == "1":
            while True:
                raw = input("port: ").strip()
                try:
                    new_port = _port_arg(raw)
                except argparse.ArgumentTypeError as exc:
                    print(f"  {exc}")
                    continue
                ctx.port = new_port
                ctx.port_source = "prompt"
                ctx.args_resolved["port"] = {"value": new_port, "source": "prompt"}
                return
        if choice == "2":
            return
        if choice == "3":
            raise SetupDeadEnd("setup aborted by user", 2)
        print("  invalid choice; enter 1, 2, or 3")


def step_journal(ctx: SetupContext, step_index: int) -> StepResult:
    started_at = utc_now()
    print_step_header(step_index, "journal config")
    persisted = read_user_config().get("journal", "").strip()
    persisted_matches = bool(persisted) and expand_path(persisted) == ctx.journal_path
    if (
        non_empty_journal(ctx.journal_path)
        and not ctx.accept_existing_journal
        and not persisted_matches
    ):
        if ctx.mode is SetupMode.NON_INTERACTIVE:
            dead_end_existing_journal(ctx)
        if not prompt_accept_existing_journal(ctx.journal_path):
            raise SetupDeadEnd("setup aborted by user", 2)

    if not persisted_matches:
        ctx.journal_path.mkdir(parents=True, exist_ok=True)
        write_user_config(journal=str(ctx.journal_path))
        print(f"[step {step_index}/{TOTAL_STEPS}] wrote {ctx.config_path}")
    else:
        print(f"[step {step_index}/{TOTAL_STEPS}] journal config already current")
        ctx.journal_path.mkdir(parents=True, exist_ok=True)
    return step_result(
        "journal",
        "ok",
        [ctx.config_path, ctx.journal_path],
        started_at,
    )


def prompt_accept_existing_journal(path: Path) -> bool:
    answer = input(f"Use existing journal at {path}? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def linux_model_sentinel() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub" / ".solstone-install-complete"


def mac_model_sentinel() -> Path:
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "solstone"
        / "parakeet"
        / "models"
        / ".install-complete"
    )


def model_paths() -> list[Path]:
    if sys.platform.startswith("linux"):
        return [linux_model_sentinel()]
    if sys.platform == "darwin":
        return [mac_model_sentinel()]
    return []


def step_install_models(ctx: SetupContext, step_index: int) -> StepResult:
    started_at = utc_now()
    if ctx.skip_models:
        print_step_skipped(step_index, "install_models", "--skip-models")
        return step_result(
            "install_models", "skipped", [], started_at, reason="--skip-models"
        )
    command = install_models_command(ctx)
    print_step_header(step_index, "install-models", command)
    try:
        rc = run_inherited(command, timeout=ctx.step_timeout_seconds)
    except subprocess.TimeoutExpired:
        return step_result(
            "install_models",
            "failed",
            model_paths(),
            started_at,
            {
                "message": (
                    f"install_models timed out after {ctx.step_timeout_seconds}s"
                ),
                "exit_code": 1,
            },
        )
    if rc != 0:
        return step_result(
            "install_models",
            "failed",
            model_paths(),
            started_at,
            {"message": "install-models failed", "exit_code": rc},
        )
    return step_result("install_models", "ok", model_paths(), started_at)


def step_skills(ctx: SetupContext, step_index: int) -> StepResult:
    started_at = utc_now()
    claude_dir = Path.home() / ".claude"
    skill_path = claude_dir / "skills" / "solstone" / "SKILL.md"
    if ctx.skip_skills:
        print_step_skipped(step_index, "skills", "--skip-skills")
        return step_result("skills", "skipped", [], started_at, reason="--skip-skills")
    if not claude_dir.exists():
        reason = f"Claude Code config not found at {claude_dir}"
        print_step_skipped(step_index, "skills", reason)
        return step_result(
            "skills", "skipped", [], started_at, reason="claude_config_missing"
        )
    command = skills_command()
    print_step_header(step_index, "skills", command)
    try:
        rc = run_inherited(command, timeout=ctx.step_timeout_seconds)
    except subprocess.TimeoutExpired:
        return step_result(
            "skills",
            "failed",
            [skill_path],
            started_at,
            {
                "message": f"skills timed out after {ctx.step_timeout_seconds}s",
                "exit_code": 1,
            },
        )
    if rc != 0:
        return step_result(
            "skills",
            "failed",
            [skill_path],
            started_at,
            {"message": "skills install failed", "exit_code": rc},
        )
    return step_result("skills", "ok", [skill_path], started_at)


def step_wrapper(ctx: SetupContext, step_index: int) -> StepResult:
    started_at = utc_now()
    wrapper_path = Path.home() / ".local" / "bin" / "sol"
    if not ctx.is_source_checkout:
        print_step_skipped(step_index, "wrapper", "packaged install")
        return step_result(
            "wrapper", "skipped", [], started_at, reason="packaged_install"
        )
    command = wrapper_command()
    print_step_header(step_index, "wrapper", command)
    try:
        rc = run_inherited(command, timeout=ctx.step_timeout_seconds)
    except subprocess.TimeoutExpired:
        return step_result(
            "wrapper",
            "failed",
            [wrapper_path],
            started_at,
            {
                "message": f"wrapper timed out after {ctx.step_timeout_seconds}s",
                "exit_code": 1,
            },
        )
    if rc != 0:
        return step_result(
            "wrapper",
            "failed",
            [wrapper_path],
            started_at,
            {"message": "wrapper install failed", "exit_code": rc},
        )
    return step_result("wrapper", "ok", [wrapper_path], started_at)


def service_artifact_path() -> Path | None:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "LaunchAgents" / "org.solpbc.solstone.plist"
    if sys.platform.startswith("linux"):
        return Path.home() / ".config" / "systemd" / "user" / "solstone.service"
    return None


def step_service(ctx: SetupContext, step_index: int) -> StepResult:
    started_at = utc_now()
    artifact = service_artifact_path()
    paths = [artifact] if artifact is not None else []
    if ctx.skip_service:
        print_step_skipped(step_index, "service", "--skip-service")
        return step_result(
            "service", "skipped", [], started_at, reason="--skip-service"
        )
    command = service_install_command(ctx)
    print_step_header(step_index, "service install", command)
    rc = run_inherited(command)
    if rc != 0:
        return step_result(
            "service",
            "failed",
            paths,
            started_at,
            {"message": "service install failed", "exit_code": rc},
        )

    from solstone.think.service import _up

    print(f"[step {step_index}/{TOTAL_STEPS}] running service up...")
    up_rc = int(_up(port=ctx.port))
    if up_rc != 0:
        return step_result(
            "service",
            "failed",
            paths,
            started_at,
            {"message": "service up failed", "exit_code": up_rc},
        )

    from solstone.think.health_cli import health_check

    print(f"[step {step_index}/{TOTAL_STEPS}] waiting for health...")
    for attempt in range(1, HEALTH_ATTEMPTS + 1):
        if health_check() == 0:
            return step_result("service", "ok", paths, started_at)
        if attempt < HEALTH_ATTEMPTS:
            time.sleep(HEALTH_SLEEP_SECONDS)
    return step_result(
        "service",
        "failed",
        paths,
        started_at,
        {"message": "service readiness timeout after 20s", "exit_code": 1},
    )


def dead_end_existing_journal(ctx: SetupContext) -> None:
    message = "\n".join(
        [
            (
                "sol setup: cannot proceed in non-interactive mode - "
                f"{ctx.journal_path} already contains journal data."
            ),
            "Setup will not auto-claim an existing journal.",
            "",
            "Retry with one of:",
            "  sol setup --accept-existing-journal",
            "  sol setup --journal /path/to/new-journal --accept-existing-journal",
            "",
            "Interactive escape:",
            "  sol setup",
            "",
            "Run 'sol setup --explain' for full step list.",
        ]
    )
    raise SetupDeadEnd(message, 2)


def dead_end_journal_is_file(ctx: SetupContext) -> None:
    message = (
        f"expected a directory at {ctx.journal_path}; got a regular file. "
        "Re-run with --journal <other-path>."
    )
    raise SetupDeadEnd(message, 2)


def dead_end_port_in_use(ctx: SetupContext) -> None:
    message = "\n".join(
        [
            (
                "sol setup: cannot proceed in non-interactive mode - "
                f"port {ctx.port} is already in use."
            ),
            "Setup will not choose a different service port silently.",
            "",
            "Retry with one of:",
            "  sol setup --port <port>",
            "  sol setup --skip-service",
            "",
            "Interactive escape:",
            "  sol setup",
            "",
            "Run 'sol setup --explain' for full step list.",
        ]
    )
    raise SetupDeadEnd(message, 2)


def print_plan(ctx: SetupContext, *, dry_run: bool) -> None:
    heading = "setup dry-run" if dry_run else "setup plan"
    print(f"{heading}:")
    print(f"  mode: {ctx.mode.value}")
    print(f"  journal: {ctx.journal_path} ({ctx.journal_source})")
    print(f"  port: {ctx.port} ({ctx.port_source})")
    print(f"  variant: {ctx.variant} ({ctx.variant_source})")
    timeout_resolved = ctx.args_resolved["step_timeout_seconds"]
    timeout_source = timeout_resolved["source"]
    print(f"  step_timeout_seconds: {ctx.step_timeout_seconds} ({timeout_source})")
    print(f"  source checkout: {ctx.is_source_checkout}")
    print()
    print(f"[step 1/6] {_STEP_NAME[step_doctor]}")
    print(f"  would run: {format_command(doctor_command(ctx))}")
    print(f"[step 2/6] {_STEP_NAME[step_journal]}")
    print(f"  would write: {ctx.config_path}")
    print(f"  would use journal: {ctx.journal_path}")
    print(f"[step 3/6] {_STEP_NAME[step_install_models]}")
    if ctx.skip_models:
        print("  skipped: --skip-models")
    else:
        print(f"  would run: {format_command(install_models_command(ctx))}")
    print(f"[step 4/6] {_STEP_NAME[step_skills]}")
    if ctx.skip_skills:
        print("  skipped: --skip-skills")
    else:
        print(f"  would run: {format_command(skills_command())}")
    print(f"[step 5/6] {_STEP_NAME[step_wrapper]}")
    if not ctx.is_source_checkout:
        print("  skipped: packaged install")
    else:
        print(f"  would run: {format_command(wrapper_command())}")
    print(f"[step 6/6] {_STEP_NAME[step_service]}")
    if ctx.skip_service:
        print("  skipped: --skip-service")
    else:
        print(f"  would run: {format_command(service_install_command(ctx))}")
        print(f"  would call: solstone.think.service._up(port={ctx.port})")
        print(
            f"  would call: think.health_cli.health_check() up to {HEALTH_ATTEMPTS} times"
        )


def print_failure(result: StepResult) -> None:
    error = result.error or {}
    message = error.get("message", "step failed")
    print(f"sol setup: {result.name} failed: {message}", file=sys.stderr)


def print_success_summary(ctx: SetupContext, manifest: dict[str, Any]) -> None:
    print()
    print("solstone is set up.")
    print()
    steps = manifest.get("steps", [])
    n_skipped_prior = sum(1 for step in steps if step.get("reason") == "prior_run_ok")
    n_skipped_other = sum(
        1
        for step in steps
        if step.get("status") == "skipped" and step.get("reason") != "prior_run_ok"
    )
    n_ran = TOTAL_STEPS - n_skipped_prior - n_skipped_other
    print(f"{n_skipped_prior} of {TOTAL_STEPS} steps already done; ran {n_ran}")
    print()
    print("artifacts:")
    paths = artifact_paths(ctx, manifest)
    if paths:
        for path in paths:
            print(f"  {path}")
    else:
        print("  none")
    print()
    if ctx.doctor_advisories:
        print("advisories from doctor:")
        for advisory in ctx.doctor_advisories:
            detail = advisory.get("detail")
            if detail:
                print(f"  - {detail}")
    else:
        print("advisories from doctor: none")
    print()
    if not ctx.skip_service:
        print(f"solstone is running at http://localhost:{ctx.port}")
        print()
    print("next: run 'sol observer install' to start observing.")


def artifact_paths(ctx: SetupContext, manifest: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for step in manifest.get("steps", []):
        if not isinstance(step, dict):
            continue
        for item in step.get("paths", []):
            if not isinstance(item, str) or item in seen:
                continue
            seen.add(item)
            paths.append(item)
    manifest_path = absolute_string(ctx.manifest_path)
    if (
        ctx.mode not in (SetupMode.DRY_RUN, SetupMode.EXPLAIN)
        and manifest_path not in seen
    ):
        paths.append(manifest_path)
    return paths


def print_prior_run_preface(ctx: SetupContext) -> None:
    status = prior_run_status(ctx)
    if status.state == "none":
        return
    if status.state == "clean":
        suffix = (
            "re-running all steps (--force)."
            if ctx.force
            else "verifying current state."
        )
        print(f"sol setup last ran cleanly on {status.timestamp}; {suffix}")
        if not ctx.force:
            print("Use --force to re-run all steps unconditionally.")
        return
    print(f"sol setup last run on {status.timestamp} left these steps incomplete:")
    for name in status.failed_steps:
        print(f"  - {name} (failed)")
    print("Re-running will verify state and re-run incomplete steps.")


def _resume_service(
    ctx: SetupContext, step_index: int, prior_step: dict
) -> StepResult | None:
    started_at = utc_now()
    from solstone.think.service import service_is_installed

    if not service_is_installed():
        return None

    from solstone.think.health_cli import health_check

    paths = prior_step.get("paths", [])
    if health_check() == 0:
        return step_result(
            "service", "skipped", paths, started_at, reason="prior_run_ok"
        )

    print(
        f"[step {step_index}/{TOTAL_STEPS}] service installed but unhealthy; restarting..."
    )
    run_inherited(
        [sys.executable, "-m", "solstone.think.sol_cli", "service", "restart"]
    )
    for attempt in range(1, HEALTH_ATTEMPTS + 1):
        if health_check() == 0:
            return step_result(
                "service",
                "ok",
                paths,
                started_at,
                reason="resumed_after_restart",
            )
        if attempt < HEALTH_ATTEMPTS:
            time.sleep(HEALTH_SLEEP_SECONDS)
    return None


_STEP_NAME: dict[Callable[[SetupContext, int], StepResult], str] = {
    step_doctor: "doctor",
    step_journal: "journal",
    step_install_models: "install_models",
    step_skills: "skills",
    step_wrapper: "wrapper",
    step_service: "service",
}

_STEPS: tuple[Callable[[SetupContext, int], StepResult], ...] = (
    step_doctor,
    step_journal,
    step_install_models,
    step_skills,
    step_wrapper,
    step_service,
)


def run_setup(ctx: SetupContext) -> int:
    if ctx.mode is SetupMode.EXPLAIN:
        print_plan(ctx, dry_run=False)
        return 0
    if ctx.mode is SetupMode.DRY_RUN:
        print_plan(ctx, dry_run=True)
        return 0

    print_prior_run_preface(ctx)
    prior_manifest = read_manifest(ctx) or {}
    prior = {} if ctx.force else prior_step_lookup(prior_manifest)
    manifest = initial_manifest(ctx)
    for index, step in enumerate(_STEPS, start=1):
        step_name = _STEP_NAME[step]
        prior_step = prior.get(step_name)
        started_at = utc_now()
        try:
            if can_skip(prior_step):
                if step is step_service:
                    result = _resume_service(ctx, index, prior_step)
                    if result is None:
                        result = step(ctx, index)
                else:
                    result = step_result(
                        step_name,
                        "skipped",
                        prior_step.get("paths", []),
                        started_at,
                        reason="prior_run_ok",
                    )
            else:
                result = step(ctx, index)
        except SetupDeadEnd:
            raise
        except Exception as exc:
            result = step_result(
                step_name,
                "failed",
                [],
                started_at,
                {
                    "message": str(exc) or exc.__class__.__name__,
                    "exit_code": 1,
                },
            )
            append_step(manifest, result)
            write_manifest(ctx, manifest)
            print_failure(result)
            return 1
        append_step(manifest, result)
        write_manifest(ctx, manifest)
        if result.status == "failed":
            print_failure(result)
            error = result.error or {}
            return int(error.get("exit_code", 1))

    manifest["completed_at"] = utc_now()
    write_manifest(ctx, manifest)
    print_success_summary(ctx, manifest)
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    try:
        ctx = resolve_context(args, raw_argv)
        return run_setup(ctx)
    except SetupDeadEnd as exc:
        print(exc.message, file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
