# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from think import health_cli, service, setup


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
    return repo


def patch_packaged_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "site-packages"
    root.mkdir()
    monkeypatch.setattr(setup, "get_project_root", lambda: str(root))
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


def patch_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    *,
    doctor_stdout: str | None = None,
    doctor_returncode: int = 0,
    command_returncode: int = 0,
) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "doctor" in command:
            return subprocess.CompletedProcess(
                command,
                doctor_returncode,
                stdout=doctor_stdout if doctor_stdout is not None else doctor_payload(),
                stderr="doctor failed\n" if doctor_returncode else "",
            )
        return subprocess.CompletedProcess(command, command_returncode)

    monkeypatch.setattr(setup.subprocess, "run", fake_run)
    return calls


def patch_service_health(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "_up", lambda port=5015: 0)
    monkeypatch.setattr(health_cli, "health_check", lambda: 0)


def command_contains(calls: list[list[str]], *parts: str) -> bool:
    return any(all(part in command for part in parts) for command in calls)


def read_manifest(journal: Path) -> dict[str, Any]:
    return json.loads((journal / ".setup-state.json").read_text(encoding="utf-8"))


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
    assert [step["name"] for step in manifest["steps"]] == [
        "doctor",
        "journal",
        "install_models",
        "skills",
        "wrapper",
        "service",
    ]
    assert "solstone is running at http://localhost:5015" in capsys.readouterr().out
    assert command_contains(calls, "install-models")
    assert command_contains(calls, "skills", "claude")
    assert command_contains(calls, "think.install_guard", "install")


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
    assert command_contains(calls, "think.install_guard", "install")


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
    assert len(manifest["steps"]) == 6
    assert command_contains(calls, "service", "install")


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
    assert not command_contains(calls, "install-models")
    assert not command_contains(calls, "skills")
    assert not command_contains(calls, "think.install_guard")
    assert not command_contains(calls, "service", "install")


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
    assert not (journal / ".setup-state.json").exists()
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


def test_idempotent_rerun_short_circuits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    (home / ".claude").mkdir()
    journal = tmp_path / "journal"
    journal.mkdir()
    (journal / ".setup-state.json").write_text(
        json.dumps({"schema_version": 1, "completed_at": "2026-05-02T21:30:42Z"}),
        encoding="utf-8",
    )
    calls = patch_subprocess(monkeypatch)
    patch_service_health(monkeypatch)

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 0
    assert command_contains(calls, "doctor")
    assert command_contains(calls, "install-models")
    assert command_contains(calls, "think.install_guard")
    assert read_manifest(journal)["completed_at"] is not None


def test_partial_completion_resumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = patch_home(monkeypatch, tmp_path)
    patch_source_checkout(monkeypatch, tmp_path)
    monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
    (home / ".claude").mkdir()
    journal = tmp_path / "journal"
    journal.mkdir()
    (journal / ".setup-state.json").write_text(
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
    assert [step["name"] for step in read_manifest(journal)["steps"]] == [
        "doctor",
        "journal",
        "install_models",
        "skills",
        "wrapper",
        "service",
    ]
    assert command_contains(calls, "service", "install")


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
        doctor_stdout=doctor_payload(
            [
                {
                    "name": "port_5015_free",
                    "severity": "advisory",
                    "status": "warn",
                    "detail": "port 5015 is in use by pid 123",
                    "fix": "kill 123",
                }
            ]
        ),
    )
    journal = tmp_path / "journal"

    rc = setup.main(["--yes", "--journal", str(journal)])

    assert rc == 2
    assert "port 5015 is already in use" in capsys.readouterr().err
    assert command_contains(calls, "doctor")
    assert not command_contains(calls, "install-models")


def test_packaged_install_skips_service(
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
    assert "packaged-install service support is not implemented in v1" in out
    assert not command_contains(calls, "think.install_guard")
    assert not command_contains(calls, "service", "install")
    assert [step["status"] for step in read_manifest(journal)["steps"]][-2:] == [
        "skipped",
        "skipped",
    ]


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
    assert not command_contains(calls, "skills", "install")
    skill_step = next(
        step for step in read_manifest(journal)["steps"] if step["name"] == "skills"
    )
    assert skill_step["status"] == "skipped"
