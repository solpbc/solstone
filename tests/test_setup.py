# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from solstone.think import health_cli, service, setup
from solstone.think.user_config import write_user_config


def patch_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def patch_source_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'solstone'\n")
    (repo / ".git").mkdir()
    monkeypatch.setattr(setup, "get_project_root", lambda: str(repo))
    monkeypatch.setattr(setup, "source_checkout", lambda: True)
    return repo


def patch_packaged_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "site-packages"
    root.mkdir()
    monkeypatch.setattr(setup, "get_project_root", lambda: str(root))
    monkeypatch.setattr(setup, "source_checkout", lambda: False)
    return root


def patch_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)


def doctor_payload(checks: list[dict[str, Any]] | None = None) -> str:
    return json.dumps(
        {
            "checks": checks or [],
            "summary": {
                "total": len(checks or []),
                "failed": 0,
                "warnings": 0,
                "skipped": 0,
            },
        }
    )


def port_advisory_payload() -> str:
    return doctor_payload(
        [
            {
                "name": "port_5015_free",
                "severity": "advisory",
                "status": "warn",
                "detail": "port 5015 is in use by pid 123",
                "fix": "kill 123",
            }
        ]
    )


def patch_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    *,
    doctor_stdout: str | None = None,
    doctor_returncode: int = 0,
    command_returncode: int = 0,
    doctor_timeout: bool = False,
    popen_timeout_command: list[str] | None = None,
) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "doctor" in command:
            if doctor_timeout:
                raise subprocess.TimeoutExpired(command, setup.DOCTOR_TIMEOUT_SECONDS)
            return subprocess.CompletedProcess(
                command,
                doctor_returncode,
                stdout=doctor_stdout if doctor_stdout is not None else doctor_payload(),
                stderr="doctor failed\n" if doctor_returncode else "",
            )
        return subprocess.CompletedProcess(command, command_returncode)

    class FakePopen:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            del kwargs
            self.command = command
            self.terminated = False
            calls.append(command)

        def wait(self, timeout: float | None = None) -> int:
            if self.command == popen_timeout_command and not self.terminated:
                raise subprocess.TimeoutExpired(self.command, timeout)
            return command_returncode

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.terminated = True

    monkeypatch.setattr(setup.subprocess, "run", fake_run)
    monkeypatch.setattr(setup.subprocess, "Popen", FakePopen)
    return calls


def patch_service_health(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "_up", lambda port=5015: 0)
    monkeypatch.setattr(health_cli, "health_check", lambda: 0)


STEP_NAMES = [
    "doctor",
    "journal",
    "install_models",
    "skills",
    "wrapper",
    "service",
]


def expected_doctor_command(port: int = 5015) -> list[str]:
    return [
        sys.executable,
        "-m",
        "solstone.think.sol_cli",
        "doctor",
        "--json",
        "--port",
        str(port),
    ]


def expected_install_models_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "solstone.think.sol_cli",
        "install-models",
        "--variant",
        "auto",
    ]


def expected_skills_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "solstone.think.sol_cli",
        "skills",
        "install",
        "--agent",
        "claude",
    ]


def expected_wrapper_command() -> list[str]:
    return [sys.executable, "-m", "solstone.think.install_guard", "install"]


def expected_service_install_command(port: int = 5015) -> list[str]:
    return [
        sys.executable,
        "-m",
        "solstone.think.sol_cli",
        "service",
        "install",
        "--port",
        str(port),
    ]


def expected_service_restart_command() -> list[str]:
    return [sys.executable, "-m", "solstone.think.sol_cli", "service", "restart"]


def assert_command(
    calls: list[list[str]], position: int, expected_argv: list[str]
) -> None:
    assert position < len(calls), (
        f"expected {position + 1}+ subprocess calls, got {len(calls)}"
    )
    assert calls[position] == expected_argv, (
        f"call[{position}] mismatch:\n  want: {expected_argv}\n  got:  {calls[position]}"
    )


def assert_step_names_and_statuses(
    manifest: dict[str, Any], statuses: list[str]
) -> None:
    assert [step["name"] for step in manifest["steps"]] == STEP_NAMES
    assert [step["status"] for step in manifest["steps"]] == statuses


def read_manifest(journal: Path) -> dict[str, Any]:
    return json.loads(
        (journal / "health" / "setup-state.json").read_text(encoding="utf-8")
    )


def touch_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def prior_artifact_paths(journal: Path) -> dict[str, list[Path]]:
    service_path = setup.service_artifact_path()
    return {
        "doctor": [],
        "journal": [setup.config_path(), journal],
        "install_models": setup.model_paths(),
        "skills": [Path.home() / ".claude" / "skills" / "solstone" / "SKILL.md"],
        "wrapper": [Path.home() / ".local" / "bin" / "sol"],
        "service": [service_path] if service_path is not None else [],
    }


def write_clean_prior_manifest(journal: Path) -> dict[str, list[Path]]:
    journal.mkdir(parents=True, exist_ok=True)
    paths_by_name = prior_artifact_paths(journal)
    for paths in paths_by_name.values():
        for path in paths:
            if path == journal:
                path.mkdir(parents=True, exist_ok=True)
            else:
                touch_file(path)
    started_at = "2026-05-02T21:29:42Z"
    completed_at = "2026-05-02T21:30:42Z"
    steps = [
        {
            "name": name,
            "status": "ok",
            "paths": [str(path.expanduser().resolve()) for path in paths_by_name[name]],
            "started_at": started_at,
            "finished_at": completed_at,
            "error": None,
        }
        for name in STEP_NAMES
    ]
    (journal / "health").mkdir(parents=True, exist_ok=True)
    (journal / "health" / "setup-state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "started_at": started_at,
                "completed_at": completed_at,
                "mode": "non_interactive",
                "args_resolved": {},
                "steps": steps,
            }
        ),
        encoding="utf-8",
    )
    return paths_by_name


def test_interactive_happy_path_default_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    patch_tty(monkeypatch)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    (home / ".claude").mkdir()
    calls = patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)

    rc = setup.main([])

    assert rc == 0
    journal = home / "Documents" / "journal"
    assert (home / ".config" / "solstone" / "config.toml").read_text(
        encoding="utf-8"
    ) == f'journal = "{journal}"\n'
    manifest = read_manifest(journal)
    assert_step_names_and_statuses(manifest, ["ok", "ok", "ok", "ok", "ok", "ok"])
    assert "solstone is running at http://localhost:5015" in capsys.readouterr().out
    assert_command(calls, 0, expected_doctor_command())
    assert_command(calls, 1, expected_install_models_command())
    assert_command(calls, 2, expected_skills_command())
    assert_command(calls, 3, expected_wrapper_command())
    assert_command(calls, 4, expected_service_install_command())
    assert len(calls) == 5


def test_interactive_happy_path_journal_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    patch_tty(monkeypatch)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    (home / ".claude").mkdir()
    calls = patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)
    journal = tmp_path / "custom-journal"

    rc = setup.main(["--journal", str(journal)])

    assert rc == 0
    assert (home / ".config" / "solstone" / "config.toml").read_text(
        encoding="utf-8"
    ) == f'journal = "{journal}"\n'
    assert read_manifest(journal)["args_resolved"]["journal"]["source"] == "cli"
    assert_command(calls, 3, expected_wrapper_command())
    assert len(calls) == 5


def test_non_interactive_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    (home / ".claude").mkdir()
    calls = patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)
    journal = tmp_path / "journal"

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 0
    manifest = read_manifest(journal)
    assert manifest["completed_at"] is not None
    assert_step_names_and_statuses(manifest, ["ok", "ok", "ok", "ok", "ok", "ok"])
    assert_command(calls, 4, expected_service_install_command())
    assert len(calls) == 5


@pytest.mark.parametrize("use_journal_flag", [False, True])
def test_non_interactive_dead_end_on_existing_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    use_journal_flag: bool,
) -> None:
    home = patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    calls = patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)
    journal = (
        tmp_path / "journal" if use_journal_flag else home / "Documents" / "journal"
    )
    (journal / "config").mkdir(parents=True)
    argv = ["--yes"]
    if use_journal_flag:
        argv.extend(["--journal", str(journal)])

    rc = setup.main(argv)

    assert rc == 2
    err = capsys.readouterr().err
    assert "already contains journal data" in err
    assert "--accept-existing-journal" in err
    assert_command(calls, 0, expected_doctor_command())
    assert len(calls) == 1


def test_dry_run_side_effect_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    calls = patch_subprocess(monkeypatch)
    journal = tmp_path / "journal"

    rc = setup.main(["--dry-run", "--journal", str(journal)])

    assert rc == 0
    assert calls == []
    assert not (home / ".config" / "solstone" / "config.toml").exists()
    assert not (journal / "health" / "setup-state.json").exists()
    assert "setup dry-run:" in capsys.readouterr().out


def test_explain_early_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    calls = patch_subprocess(monkeypatch)

    rc = setup.main(["--explain"])

    assert rc == 0
    assert calls == []
    assert "setup plan:" in capsys.readouterr().out


def test_manifest_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    (home / ".claude").mkdir()
    patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)
    journal = tmp_path / "journal"

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 0
    manifest = read_manifest(journal)
    assert manifest["schema_version"] == 1
    assert manifest["mode"] == "non_interactive"
    assert all(
        Path(path).is_absolute() for step in manifest["steps"] for path in step["paths"]
    )
    assert {step["status"] for step in manifest["steps"]} <= {"ok", "skipped", "failed"}


def test_persisted_journal_skips_existing_journal_check_non_interactive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    journal = tmp_path / "journal"
    (journal / "config").mkdir(parents=True)
    write_user_config(journal=str(journal))
    patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)

    rc = setup.main(["--yes"])

    assert rc == 0
    assert "already contains journal data" not in capsys.readouterr().err
    journal_step = next(
        step for step in read_manifest(journal)["steps"] if step["name"] == "journal"
    )
    assert journal_step["status"] == "ok"


def test_persisted_journal_skips_existing_journal_check_interactive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    patch_tty(monkeypatch)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    journal = tmp_path / "journal"
    (journal / "config").mkdir(parents=True)
    write_user_config(journal=str(journal))
    patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)

    def fail_on_prompt(path: Path) -> bool:
        raise AssertionError(f"unexpected existing-journal prompt for {path}")

    monkeypatch.setattr(setup, "prompt_accept_existing_journal", fail_on_prompt)

    rc = setup.main([])

    assert rc == 0
    assert "Use existing journal" not in capsys.readouterr().out


def test_existing_journal_dead_end_still_fires_when_path_not_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)
    journal = tmp_path / "journal"
    (journal / "config").mkdir(parents=True)

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 2
    assert "already contains journal data" in capsys.readouterr().err


def test_clean_rerun_preface_when_manifest_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)
    journal = tmp_path / "journal"
    journal.mkdir()
    completed_at = "2026-05-02T21:30:42Z"
    started_at = "2026-05-02T21:29:42Z"
    steps = [
        {
            "name": name,
            "status": "ok",
            "paths": [],
            "started_at": started_at,
            "finished_at": completed_at,
            "error": None,
        }
        for name in (
            "doctor",
            "journal",
            "install_models",
            "skills",
            "wrapper",
            "service",
        )
    ]
    (journal / "health").mkdir(parents=True, exist_ok=True)
    (journal / "health" / "setup-state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "started_at": started_at,
                "completed_at": completed_at,
                "mode": "non_interactive",
                "args_resolved": {},
                "steps": steps,
            }
        ),
        encoding="utf-8",
    )

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        f"sol setup last ran cleanly on {completed_at}; verifying current state."
    )
    assert lines[1] == "Use --force to re-run all steps unconditionally."


def test_partial_rerun_preface_when_steps_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)
    journal = tmp_path / "journal"
    journal.mkdir()
    started_at = "2026-05-02T21:29:42Z"
    (journal / "health").mkdir(parents=True, exist_ok=True)
    (journal / "health" / "setup-state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "started_at": started_at,
                "completed_at": None,
                "mode": "non_interactive",
                "args_resolved": {},
                "steps": [
                    {
                        "name": "install_models",
                        "status": "failed",
                        "paths": [],
                        "started_at": started_at,
                        "finished_at": "2026-05-02T21:30:42Z",
                        "error": {"message": "install failed", "exit_code": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 0
    assert (
        f"sol setup last run on {started_at} left these steps incomplete:\n"
        "  - install_models (failed)\n"
        "Re-running will verify state and re-run incomplete steps."
    ) in capsys.readouterr().out


def test_no_preface_without_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)
    journal = tmp_path / "journal"

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "last ran cleanly" not in out and "left these steps incomplete" not in out


def test_force_flag_changes_preface_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)
    journal = tmp_path / "journal"
    journal.mkdir()
    completed_at = "2026-05-02T21:30:42Z"
    started_at = "2026-05-02T21:29:42Z"
    steps = [
        {
            "name": name,
            "status": "ok",
            "paths": [],
            "started_at": started_at,
            "finished_at": completed_at,
            "error": None,
        }
        for name in (
            "doctor",
            "journal",
            "install_models",
            "skills",
            "wrapper",
            "service",
        )
    ]
    (journal / "health").mkdir(parents=True, exist_ok=True)
    (journal / "health" / "setup-state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "started_at": started_at,
                "completed_at": completed_at,
                "mode": "non_interactive",
                "args_resolved": {},
                "steps": steps,
            }
        ),
        encoding="utf-8",
    )

    rc = setup.main(["--yes", "--force", "--journal", str(journal)])

    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith(
        f"sol setup last ran cleanly on {completed_at}; re-running all steps (--force)."
    )
    assert "Use --force to re-run all steps unconditionally." not in out


def test_partial_completion_runs_remaining_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    (home / ".claude").mkdir()
    journal = tmp_path / "journal"
    journal.mkdir()
    (journal / "health").mkdir(parents=True, exist_ok=True)
    (journal / "health" / "setup-state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "steps": [
                    {"name": "doctor", "status": "ok"},
                    {"name": "service", "status": "failed"},
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 0
    manifest = read_manifest(journal)
    assert_step_names_and_statuses(manifest, ["skipped", "ok", "ok", "ok", "ok", "ok"])
    assert manifest["steps"][0]["reason"] == "prior_run_ok"
    assert_command(calls, 0, expected_install_models_command())
    assert_command(calls, 1, expected_skills_command())
    assert_command(calls, 2, expected_wrapper_command())
    assert_command(calls, 3, expected_service_install_command())
    assert len(calls) == 4


def test_port_in_use_default_non_interactive_dead_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    calls = patch_subprocess(
        monkeypatch,
        doctor_stdout=port_advisory_payload(),
    )
    journal = tmp_path / "journal"

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 2
    assert "port 5015 is already in use" in capsys.readouterr().err
    assert_command(calls, 0, expected_doctor_command())
    assert len(calls) == 1


def test_interactive_port_in_use_prompts_for_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    patch_tty(monkeypatch)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    (home / ".claude").mkdir()
    answers = iter(["1", "8080"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    calls = patch_subprocess(monkeypatch, doctor_stdout=port_advisory_payload())
    patch_service_health(monkeypatch)

    rc = setup.main([])

    assert rc == 0
    journal = home / "Documents" / "journal"
    assert read_manifest(journal)["args_resolved"]["port"] == {
        "value": 8080,
        "source": "prompt",
    }
    assert_command(calls, 0, expected_doctor_command())
    assert_command(calls, 4, expected_service_install_command(port=8080))
    assert len(calls) == 5


def test_interactive_port_in_use_proceed_anyway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    patch_tty(monkeypatch)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    (home / ".claude").mkdir()
    answers = iter(["2"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    calls = patch_subprocess(monkeypatch, doctor_stdout=port_advisory_payload())
    patch_service_health(monkeypatch)

    rc = setup.main([])

    assert rc == 0
    assert_command(calls, 0, expected_doctor_command())
    assert_command(calls, 4, expected_service_install_command(port=5015))
    assert len(calls) == 5


def test_interactive_port_in_use_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    patch_tty(monkeypatch)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    answers = iter(["3"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    calls = patch_subprocess(monkeypatch, doctor_stdout=port_advisory_payload())

    rc = setup.main([])

    assert rc == 2
    assert "setup aborted by user" in capsys.readouterr().err
    assert_command(calls, 0, expected_doctor_command())
    assert expected_service_install_command() not in calls
    assert len(calls) == 1


def test_doctor_timeout_records_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    journal = tmp_path / "journal"
    patch_subprocess(monkeypatch, doctor_timeout=True)

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 1
    step = read_manifest(journal)["steps"][-1]
    assert step["name"] == "doctor"
    assert step["status"] == "failed"
    assert "timed out after 30s" in step["error"]["message"]
    assert step["error"]["exit_code"] == 1


def test_install_models_timeout_records_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    journal = tmp_path / "journal"
    patch_subprocess(
        monkeypatch,
        popen_timeout_command=expected_install_models_command(),
    )

    rc = setup.main(["--yes", "--journal", str(journal), "--step-timeout-seconds", "1"])

    assert rc == 1
    step = read_manifest(journal)["steps"][-1]
    assert step["name"] == "install_models"
    assert step["status"] == "failed"
    assert "timed out after 1s" in step["error"]["message"]
    assert step["error"]["exit_code"] == 1


def test_step_timeout_seconds_passes_through_to_help_and_explain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        setup.main(["--help"])
    assert raised.value.code == 0
    assert "--step-timeout-seconds" in capsys.readouterr().out

    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)

    rc = setup.main(["--explain", "--yes"])

    assert rc == 0
    assert "step_timeout_seconds: 1800" in capsys.readouterr().out


def test_empty_journal_arg_rejected_at_parse_time(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        setup.main(["--journal", ""])

    assert raised.value.code == 2
    assert "--journal must not be empty" in capsys.readouterr().err


@pytest.mark.parametrize("port", ["0", "99999"])
def test_port_out_of_range_rejected_at_parse_time(
    capsys: pytest.CaptureFixture[str],
    port: str,
) -> None:
    with pytest.raises(SystemExit) as raised:
        setup.main(["--port", port])

    assert raised.value.code == 2
    assert "--port must be in 1024-65535" in capsys.readouterr().err


def test_packaged_install_runs_service_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_packaged_install(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    calls = patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)
    journal = tmp_path / "journal"

    rc = setup.main(
        ["--yes", "--journal", str(journal), "--skip-models", "--skip-skills"]
    )

    assert rc == 0
    out = capsys.readouterr().out
    unsupported_message = " ".join(
        ["packaged-install service support", "is not implemented"]
    )
    assert unsupported_message not in out
    assert "solstone is running at http://localhost:5015" in out
    assert_command(calls, 0, expected_doctor_command())
    assert_command(
        calls,
        1,
        [
            sys.executable,
            "-m",
            "solstone.think.sol_cli",
            "service",
            "install",
            "--port",
            "5015",
        ],
    )
    assert len(calls) == 2
    steps = read_manifest(journal)["steps"]
    assert steps[-2]["status"] == "skipped"
    assert steps[-2]["reason"] == "packaged_install"
    assert steps[-1]["name"] == "service"
    assert steps[-1]["status"] == "ok"


def test_no_claude_config_skips_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    calls = patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)
    journal = tmp_path / "journal"

    rc = setup.main(["--yes", "--journal", str(journal), "--skip-models"])

    assert rc == 0
    assert "Claude Code config not found" in capsys.readouterr().out
    assert_command(calls, 0, expected_doctor_command())
    assert_command(calls, 1, expected_wrapper_command())
    assert_command(calls, 2, expected_service_install_command())
    assert len(calls) == 3
    skill_step = next(
        step for step in read_manifest(journal)["steps"] if step["name"] == "skills"
    )
    assert skill_step["status"] == "skipped"


def test_resumption_skips_completed_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    journal = tmp_path / "journal"
    write_clean_prior_manifest(journal)
    calls = patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 0
    assert calls == []
    manifest = read_manifest(journal)
    assert_step_names_and_statuses(manifest, ["skipped"] * 6)
    assert {step["reason"] for step in manifest["steps"]} == {"prior_run_ok"}


def test_resumption_runs_step_when_artifact_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    journal = tmp_path / "journal"
    paths_by_name = write_clean_prior_manifest(journal)
    paths_by_name["wrapper"][0].unlink()
    calls = patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 0
    assert_command(calls, 0, expected_wrapper_command())
    assert len(calls) == 1
    manifest = read_manifest(journal)
    assert_step_names_and_statuses(
        manifest, ["skipped", "skipped", "skipped", "skipped", "ok", "skipped"]
    )


def test_resumption_wedged_service_restarts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    journal = tmp_path / "journal"
    write_clean_prior_manifest(journal)
    calls = patch_subprocess(monkeypatch)
    health_results = iter([1, 0])
    monkeypatch.setattr(health_cli, "health_check", lambda: next(health_results, 0))

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 0
    assert_command(calls, 0, expected_service_restart_command())
    assert len(calls) == 1
    service_step = read_manifest(journal)["steps"][-1]
    assert service_step["status"] == "ok"
    assert service_step["reason"] == "resumed_after_restart"


def test_resumption_wedged_service_falls_through_when_restart_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    monkeypatch.setattr(setup, "HEALTH_ATTEMPTS", 1)
    monkeypatch.setattr(setup, "HEALTH_SLEEP_SECONDS", 0)
    monkeypatch.setattr(service, "_up", lambda port=5015: 0)
    journal = tmp_path / "journal"
    write_clean_prior_manifest(journal)
    calls = patch_subprocess(monkeypatch)
    monkeypatch.setattr(health_cli, "health_check", lambda: 1)

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 1
    assert_command(calls, 0, expected_service_restart_command())
    assert_command(calls, 1, expected_service_install_command())
    assert len(calls) == 2


def test_force_skips_resumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    journal = tmp_path / "journal"
    write_clean_prior_manifest(journal)
    calls = patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)

    rc = setup.main(["--yes", "--force", "--journal", str(journal)])

    assert rc == 0
    assert_command(calls, 0, expected_doctor_command())
    assert_command(calls, 1, expected_install_models_command())
    assert_command(calls, 2, expected_skills_command())
    assert_command(calls, 3, expected_wrapper_command())
    assert_command(calls, 4, expected_service_install_command())
    assert len(calls) == 5
    manifest = read_manifest(journal)
    assert_step_names_and_statuses(manifest, ["ok", "ok", "ok", "ok", "ok", "ok"])
    assert all(step["reason"] is None for step in manifest["steps"])


def test_step_exception_records_failed_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    journal = tmp_path / "journal"

    def boom(ctx: setup.SetupContext, step_index: int) -> setup.StepResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(setup, "_STEPS", (boom,))
    monkeypatch.setattr(setup, "_STEP_NAME", {boom: "doctor"})

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 1
    manifest = read_manifest(journal)
    assert len(manifest["steps"]) == 1
    step = manifest["steps"][0]
    assert step["name"] == "doctor"
    assert step["status"] == "failed"
    assert step["error"]["message"] == "boom"


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(7)])
def test_base_exceptions_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exc: BaseException,
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    journal = tmp_path / "journal"

    def boom(ctx: setup.SetupContext, step_index: int) -> setup.StepResult:
        raise exc

    monkeypatch.setattr(setup, "_STEPS", (boom,))
    monkeypatch.setattr(setup, "_STEP_NAME", {boom: "doctor"})

    with pytest.raises(type(exc)) as raised:
        setup.main(["--yes", "--journal", str(journal)])

    if isinstance(exc, SystemExit):
        assert raised.value.code == 7
    assert not (journal / "health" / "setup-state.json").exists()


def test_env_journal_overrides_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    config_journal = tmp_path / "from_config"
    env_journal = tmp_path / "from_env"
    write_user_config(journal=str(config_journal))
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(env_journal))
    calls = patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)

    rc = setup.main(["--yes"])

    assert rc == 0
    assert (home / ".config" / "solstone" / "config.toml").read_text(
        encoding="utf-8"
    ) == f'journal = "{env_journal}"\n'
    manifest = read_manifest(env_journal)
    assert manifest["args_resolved"]["journal"]["source"] == "env"
    assert env_journal.is_dir()
    assert not config_journal.exists()
    assert_command(calls, 0, expected_doctor_command())


def test_journal_is_regular_file_dead_ends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    journal_file = tmp_path / "journal-file"
    journal_file.write_text("not a directory", encoding="utf-8")
    calls = patch_subprocess(monkeypatch)

    rc = setup.main(["--yes", "--journal", str(journal_file)])

    assert rc == 2
    assert calls == []
    assert "directory" in capsys.readouterr().err
    assert not (journal_file / "health" / "setup-state.json").exists()


def test_doctor_parse_failure_records_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    journal = tmp_path / "journal"
    patch_subprocess(monkeypatch, doctor_stdout="not json")

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 1
    manifest = read_manifest(journal)
    assert len(manifest["steps"]) == 1
    step = manifest["steps"][0]
    assert step["name"] == "doctor"
    assert step["status"] == "failed"
    assert "doctor JSON parse failed" in step["error"]["message"]


def test_invalid_manifest_treated_as_no_prior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    (home / ".claude").mkdir()
    journal = tmp_path / "journal"
    journal.mkdir()
    (journal / "health").mkdir(parents=True, exist_ok=True)
    (journal / "health" / "setup-state.json").write_text("{", encoding="utf-8")
    calls = patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "last ran cleanly" not in out and "left these steps incomplete" not in out
    assert_command(calls, 0, expected_doctor_command())
    assert_command(calls, 1, expected_install_models_command())
    assert_command(calls, 2, expected_skills_command())
    assert_command(calls, 3, expected_wrapper_command())
    assert_command(calls, 4, expected_service_install_command())
    assert len(calls) == 5


def test_port_propagates_to_subprocess_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    (home / ".claude").mkdir()
    journal = tmp_path / "journal"
    calls = patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)

    rc = setup.main(["--yes", "--journal", str(journal), "--port", "8080"])

    assert rc == 0
    assert_command(calls, 0, expected_doctor_command(port=8080))
    assert_command(calls, 4, expected_service_install_command(port=8080))
    assert len(calls) == 5
