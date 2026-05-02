# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from think import install_guard

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def doctor():
    from think import doctor as doctor_module

    yield doctor_module


@pytest.fixture
def home_root(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def args(doctor, *, port: int = 5015):
    return doctor.Args(verbose=False, json=False, port=port)


def make_repo(tmp_path: Path, *, worktree: bool = False) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    if worktree:
        (repo / ".git").write_text("gitdir: /tmp/worktree\n", encoding="utf-8")
    else:
        (repo / ".git").mkdir()
    return repo


def ensure_expected_target(repo: Path) -> Path:
    target = repo / ".venv" / "bin" / "sol"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")
    return target


def make_alias(home_root: Path, target: Path | str) -> Path:
    alias = home_root / ".local" / "bin" / "sol"
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(target)
    return alias


def other_target(tmp_path: Path) -> Path:
    target = tmp_path / "other" / ".venv" / "bin" / "sol"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")
    return target


def test_install_guard_import_succeeds_when_frontmatter_is_shadowed(tmp_path):
    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir()
    (shadow_dir / "frontmatter.py").write_text(
        'raise ImportError("blocked for test")\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    pythonpath_parts = [str(shadow_dir), str(ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from think.install_guard import parse_wrapper; print('ok')",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


class TestPythonVersion:
    def test_ok(self, doctor):
        result = doctor.python_version_check(args(doctor))
        assert result.status == "ok"

    def test_fail_when_too_old(self, doctor, monkeypatch):
        monkeypatch.setattr(doctor.sys, "version_info", (3, 9, 18))
        result = doctor.python_version_check(args(doctor))
        assert result.status == "fail"
        assert "does not satisfy" in result.detail


class TestUvInstalled:
    def test_ok(self, doctor, monkeypatch):
        monkeypatch.setattr(
            doctor,
            "run_probe",
            lambda *_args, **_kwargs: doctor.ProbeOutput("uv 0.10.0\n", "", 0),
        )
        result = doctor.uv_installed_check(args(doctor))
        assert result.status == "ok"

    def test_missing(self, doctor, monkeypatch):
        def raise_missing(*_args, **_kwargs):
            raise FileNotFoundError

        monkeypatch.setattr(doctor.subprocess, "run", raise_missing)
        result = doctor.uv_installed_check(args(doctor))
        assert result.status == "fail"
        assert "probe command not found" in result.detail

    def test_fail_when_too_old(self, doctor, monkeypatch):
        monkeypatch.setattr(
            doctor,
            "run_probe",
            lambda *_args, **_kwargs: doctor.ProbeOutput("uv 0.7.0\n", "", 0),
        )
        result = doctor.uv_installed_check(args(doctor))
        assert result.status == "fail"
        assert "older than required" in result.detail


class TestVenvConsistent:
    def test_skip_when_absent(self, doctor, monkeypatch, tmp_path):
        monkeypatch.setattr(doctor, "ROOT", tmp_path)
        result = doctor.venv_consistent_check(args(doctor))
        assert result.status == "skip"

    def test_ok_when_consistent(self, doctor, monkeypatch, tmp_path):
        monkeypatch.setattr(doctor, "ROOT", tmp_path)
        python_bin = tmp_path / ".venv" / "bin" / "python"
        python_bin.parent.mkdir(parents=True)
        python_bin.write_text("", encoding="utf-8")
        monkeypatch.setattr(
            doctor,
            "run_probe",
            lambda *_args, **_kwargs: doctor.ProbeOutput(
                f"{tmp_path / '.venv'}\n", "", 0
            ),
        )
        result = doctor.venv_consistent_check(args(doctor))
        assert result.status == "ok"

    def test_fail_when_inconsistent(self, doctor, monkeypatch, tmp_path):
        monkeypatch.setattr(doctor, "ROOT", tmp_path)
        python_bin = tmp_path / ".venv" / "bin" / "python"
        python_bin.parent.mkdir(parents=True)
        python_bin.write_text("", encoding="utf-8")
        monkeypatch.setattr(
            doctor,
            "run_probe",
            lambda *_args, **_kwargs: doctor.ProbeOutput("/tmp/elsewhere\n", "", 0),
        )
        result = doctor.venv_consistent_check(args(doctor))
        assert result.status == "fail"


class TestSolImportable:
    def test_skip_when_absent(self, doctor, monkeypatch, tmp_path):
        monkeypatch.setattr(doctor, "ROOT", tmp_path)
        result = doctor.sol_importable_check(args(doctor))
        assert result.status == "skip"

    def test_ok(self, doctor, monkeypatch, tmp_path):
        monkeypatch.setattr(doctor, "ROOT", tmp_path)
        python_bin = tmp_path / ".venv" / "bin" / "python"
        python_bin.parent.mkdir(parents=True)
        python_bin.write_text("", encoding="utf-8")
        monkeypatch.setattr(
            doctor,
            "run_probe",
            lambda *_args, **_kwargs: doctor.ProbeOutput("", "", 0),
        )
        result = doctor.sol_importable_check(args(doctor))
        assert result.status == "ok"
        assert (
            result.detail == "from think.sol_cli import main succeeded outside repo cwd"
        )

    def test_fail_on_module_not_found(self, doctor, monkeypatch, tmp_path):
        monkeypatch.setattr(doctor, "ROOT", tmp_path)
        python_bin = tmp_path / ".venv" / "bin" / "python"
        python_bin.parent.mkdir(parents=True)
        python_bin.write_text("", encoding="utf-8")
        monkeypatch.setattr(
            doctor,
            "run_probe",
            lambda *_args, **_kwargs: doctor.ProbeOutput(
                "",
                "Traceback (most recent call last):\nModuleNotFoundError: No module named 'think'\n",
                1,
            ),
        )
        result = doctor.sol_importable_check(args(doctor))
        assert result.status == "fail"
        assert result.detail == "ModuleNotFoundError: No module named 'think'"

    def test_fail_on_other_exception(self, doctor, monkeypatch, tmp_path):
        monkeypatch.setattr(doctor, "ROOT", tmp_path)
        python_bin = tmp_path / ".venv" / "bin" / "python"
        python_bin.parent.mkdir(parents=True)
        python_bin.write_text("", encoding="utf-8")
        monkeypatch.setattr(
            doctor,
            "run_probe",
            lambda *_args, **_kwargs: doctor.ProbeOutput(
                "", "SyntaxError: broken import\n", 1
            ),
        )
        result = doctor.sol_importable_check(args(doctor))
        assert result.status == "fail"
        assert result.detail == "SyntaxError: broken import"


class TestNpxChecks:
    def test_npx_on_path_ok(self, doctor, monkeypatch):
        monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/npx")
        result = doctor.npx_on_path_check(args(doctor))
        assert result.status == "ok"

    def test_npx_on_path_fail(self, doctor, monkeypatch):
        monkeypatch.setattr(doctor.shutil, "which", lambda _name: None)
        result = doctor.npx_on_path_check(args(doctor))
        assert result.status == "fail"

    def test_npx_non_interactive_ok(self, doctor, monkeypatch):
        monkeypatch.setattr(
            doctor,
            "run_probe",
            lambda *_args, **_kwargs: doctor.ProbeOutput("10.1.0\n", "", 0),
        )
        result = doctor.npx_non_interactive_check(args(doctor))
        assert result.status == "ok"

    def test_npx_non_interactive_fail_on_nonzero(self, doctor, monkeypatch):
        monkeypatch.setattr(
            doctor,
            "run_probe",
            lambda *_args, **_kwargs: doctor.make_result(
                doctor.CHECK_MAP["npx_non_interactive"],
                "fail",
                "probe exited 1: boom",
            ),
        )
        result = doctor.npx_non_interactive_check(args(doctor))
        assert result.status == "fail"

    def test_npx_non_interactive_fail_on_timeout(self, doctor, monkeypatch):
        def raise_timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(["npx"], timeout=2.0)

        monkeypatch.setattr(doctor.subprocess, "run", raise_timeout)
        result = doctor.npx_non_interactive_check(args(doctor))
        assert result.status == "fail"
        assert "timed out" in result.detail

    def test_npx_non_interactive_fail_on_empty_stdout(self, doctor, monkeypatch):
        monkeypatch.setattr(
            doctor,
            "run_probe",
            lambda *_args, **_kwargs: doctor.ProbeOutput("", "", 0),
        )
        result = doctor.npx_non_interactive_check(args(doctor))
        assert result.status == "fail"
        assert "unexpected output" in result.detail


class TestPortCheck:
    def test_severity_is_advisory(self, doctor):
        assert doctor.CHECK_MAP["port_5015_free"].severity == "advisory"

    def test_skip_when_lsof_missing(self, doctor, monkeypatch):
        monkeypatch.setattr(doctor.shutil, "which", lambda _name: None)
        result = doctor.port_5015_free_check(args(doctor))
        assert result.status == "skip"

    def test_ok_when_port_free(self, doctor, monkeypatch):
        monkeypatch.setattr(
            doctor,
            "import_install_guard",
            lambda: (_ for _ in ()).throw(ImportError("skip worktree guard")),
        )
        monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/lsof")
        monkeypatch.setattr(
            doctor,
            "run_probe",
            lambda *_args, **_kwargs: doctor.ProbeOutput("", "", 1),
        )
        result = doctor.port_5015_free_check(args(doctor))
        assert result.status == "ok"
        assert "is free" in result.detail

    def test_ok_when_owned_by_repo_sol(self, doctor, monkeypatch, tmp_path):
        monkeypatch.setattr(doctor, "ROOT", tmp_path)
        sol_bin = tmp_path / ".venv" / "bin" / "sol"
        sol_bin.parent.mkdir(parents=True)
        sol_bin.write_text("", encoding="utf-8")
        monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/lsof")
        monkeypatch.setattr(
            doctor,
            "run_probe",
            lambda *_args, **_kwargs: doctor.ProbeOutput("p123\n", "", 0),
        )
        monkeypatch.setattr(doctor, "resolve_alias_target", lambda: None)
        monkeypatch.setattr(doctor.os, "readlink", lambda _path: str(sol_bin))
        result = doctor.port_5015_free_check(args(doctor))
        assert result.status == "ok"
        assert "this repo's solstone" in result.detail

    def test_warn_when_exe_not_owned_even_if_name_mentions_sol(
        self, doctor, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(doctor, "ROOT", tmp_path)
        monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/lsof")
        monkeypatch.setattr(
            doctor,
            "run_probe",
            lambda *_args, **_kwargs: doctor.ProbeOutput("p123\n", "", 0),
        )
        monkeypatch.setattr(doctor, "resolve_alias_target", lambda: None)
        monkeypatch.setattr(doctor.os, "readlink", lambda _path: "/usr/bin/python3")
        result = doctor.port_5015_free_check(args(doctor))
        assert result.status == "warn"
        assert result.severity == "advisory"
        assert "/usr/bin/python3" in result.detail

    def test_warn_on_lsof_timeout(self, doctor, monkeypatch):
        monkeypatch.setattr(
            doctor,
            "import_install_guard",
            lambda: (_ for _ in ()).throw(ImportError("skip worktree guard")),
        )
        monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/lsof")

        def raise_timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(["lsof"], timeout=1.0)

        monkeypatch.setattr(doctor.subprocess, "run", raise_timeout)
        result = doctor.port_5015_free_check(args(doctor))
        assert result.status == "warn"
        assert result.severity == "advisory"
        assert "timed out" in result.detail

    def test_resolve_alias_target_reads_managed_wrapper_sol_bin(
        self, doctor, home_root, tmp_path
    ):
        sol_bin = tmp_path / "repo" / ".venv" / "bin" / "sol"
        sol_bin.parent.mkdir(parents=True, exist_ok=True)
        sol_bin.write_text("", encoding="utf-8")
        alias = home_root / ".local" / "bin" / "sol"
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text(
            install_guard.render_wrapper(
                str((tmp_path / "journal").resolve()),
                str(sol_bin),
            ),
            encoding="utf-8",
        )
        alias.chmod(0o755)

        assert doctor.resolve_alias_target() == sol_bin.resolve()


class TestDiskSpace:
    def test_warn_when_low(self, doctor, monkeypatch):
        monkeypatch.setattr(
            doctor.shutil,
            "disk_usage",
            lambda _root: SimpleNamespace(total=100, used=95, free=5 * 1024**3),
        )
        result = doctor.disk_space_check(args(doctor))
        assert result.status == "warn"

    def test_ok_when_sufficient(self, doctor, monkeypatch):
        monkeypatch.setattr(
            doctor.shutil,
            "disk_usage",
            lambda _root: SimpleNamespace(total=100, used=80, free=20 * 1024**3),
        )
        result = doctor.disk_space_check(args(doctor))
        assert result.status == "ok"


class TestConfigDirReadable:
    def test_ok(self, doctor, monkeypatch, home_root):
        config_dir = home_root / ".config"
        config_dir.mkdir()
        result = doctor.config_dir_readable_check(args(doctor))
        assert result.status == "ok"

    def test_fail_when_home_unwritable(self, doctor, monkeypatch, home_root):
        def fake_access(path, mode):
            if Path(path) == home_root:
                return False
            return True

        monkeypatch.setattr(doctor.os, "access", fake_access)
        result = doctor.config_dir_readable_check(args(doctor))
        assert result.status == "fail"


class TestStaleAliasSymlink:
    def setup_import(self, doctor, monkeypatch):
        monkeypatch.setattr(
            doctor,
            "import_install_guard",
            lambda: (install_guard.AliasState, install_guard.check_alias),
        )

    def test_absent_ok(self, doctor, monkeypatch, home_root, tmp_path):
        self.setup_import(doctor, monkeypatch)
        repo = make_repo(tmp_path)
        monkeypatch.setattr(doctor, "ROOT", repo)
        result = doctor.stale_alias_symlink_check(args(doctor))
        assert result.status == "ok"

    def test_owned_ok(self, doctor, monkeypatch, home_root, tmp_path):
        self.setup_import(doctor, monkeypatch)
        repo = make_repo(tmp_path)
        make_alias(home_root, ensure_expected_target(repo))
        monkeypatch.setattr(doctor, "ROOT", repo)
        result = doctor.stale_alias_symlink_check(args(doctor))
        assert result.status == "ok"

    def test_cross_repo_fail(self, doctor, monkeypatch, home_root, tmp_path):
        self.setup_import(doctor, monkeypatch)
        repo = make_repo(tmp_path)
        make_alias(home_root, other_target(tmp_path))
        monkeypatch.setattr(doctor, "ROOT", repo)
        result = doctor.stale_alias_symlink_check(args(doctor))
        assert result.status == "fail"

    def test_dangling_fail(self, doctor, monkeypatch, home_root, tmp_path):
        self.setup_import(doctor, monkeypatch)
        repo = make_repo(tmp_path)
        missing = tmp_path / "missing" / ".venv" / "bin" / "sol"
        make_alias(home_root, missing)
        monkeypatch.setattr(doctor, "ROOT", repo)
        result = doctor.stale_alias_symlink_check(args(doctor))
        assert result.status == "fail"

    def test_not_symlink_fail(self, doctor, monkeypatch, home_root, tmp_path):
        self.setup_import(doctor, monkeypatch)
        repo = make_repo(tmp_path)
        alias = home_root / ".local" / "bin" / "sol"
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text("not a symlink", encoding="utf-8")
        monkeypatch.setattr(doctor, "ROOT", repo)
        result = doctor.stale_alias_symlink_check(args(doctor))
        assert result.status == "fail"

    def test_worktree_skip(self, doctor, monkeypatch, home_root, tmp_path):
        self.setup_import(doctor, monkeypatch)
        repo = make_repo(tmp_path, worktree=True)
        monkeypatch.setattr(doctor, "ROOT", repo)
        result = doctor.stale_alias_symlink_check(args(doctor))
        assert result.status == "skip"

    def test_import_failure_skips(self, doctor, monkeypatch):
        monkeypatch.setattr(
            doctor,
            "import_install_guard",
            lambda: (_ for _ in ()).throw(ImportError("boom")),
        )
        result = doctor.stale_alias_symlink_check(args(doctor))
        assert result.status == "skip"
        assert "could not import think.install_guard" in result.detail


class TestMacosFirewall:
    def test_skip_on_linux(self, doctor, monkeypatch):
        monkeypatch.setattr(doctor, "platform_tag", lambda: "linux")
        result = doctor.macos_firewall_localhost_check(args(doctor))
        assert result.status == "skip"

    def test_ok_when_off(self, doctor, monkeypatch):
        monkeypatch.setattr(doctor, "platform_tag", lambda: "darwin")
        monkeypatch.setattr(doctor.Path, "exists", lambda self: True)
        monkeypatch.setattr(
            doctor,
            "run_probe",
            lambda *_args, **_kwargs: doctor.ProbeOutput(
                "Firewall is disabled.\n", "", 0
            ),
        )
        result = doctor.macos_firewall_localhost_check(args(doctor))
        assert result.status == "ok"

    def test_warn_when_blockall_on(self, doctor, monkeypatch):
        monkeypatch.setattr(doctor, "platform_tag", lambda: "darwin")
        monkeypatch.setattr(doctor.Path, "exists", lambda self: True)
        outputs = iter(
            [
                doctor.ProbeOutput("Firewall is enabled. (State = 1)\n", "", 0),
                doctor.ProbeOutput(
                    "Block all incoming connections is enabled.\n", "", 0
                ),
            ]
        )
        monkeypatch.setattr(
            doctor, "run_probe", lambda *_args, **_kwargs: next(outputs)
        )
        result = doctor.macos_firewall_localhost_check(args(doctor))
        assert result.status == "warn"

    def test_ok_when_blockall_off(self, doctor, monkeypatch):
        monkeypatch.setattr(doctor, "platform_tag", lambda: "darwin")
        monkeypatch.setattr(doctor.Path, "exists", lambda self: True)
        outputs = iter(
            [
                doctor.ProbeOutput("Firewall is enabled. (State = 1)\n", "", 0),
                doctor.ProbeOutput(
                    "Block all incoming connections is disabled.\n", "", 0
                ),
            ]
        )
        monkeypatch.setattr(
            doctor, "run_probe", lambda *_args, **_kwargs: next(outputs)
        )
        result = doctor.macos_firewall_localhost_check(args(doctor))
        assert result.status == "ok"


class TestLaunchdStalePlist:
    def test_skip_on_linux(self, doctor, monkeypatch):
        monkeypatch.setattr(doctor, "platform_tag", lambda: "linux")
        result = doctor.launchd_stale_plist_check(args(doctor))
        assert result.status == "skip"

    def test_skip_when_absent(self, doctor, monkeypatch, home_root):
        monkeypatch.setattr(doctor, "platform_tag", lambda: "darwin")
        result = doctor.launchd_stale_plist_check(args(doctor))
        assert result.status == "skip"

    def test_fail_when_target_missing(self, doctor, monkeypatch, home_root):
        monkeypatch.setattr(doctor, "platform_tag", lambda: "darwin")
        plist_path = (
            home_root / "Library" / "LaunchAgents" / "org.solpbc.solstone.plist"
        )
        plist_path.parent.mkdir(parents=True)
        plist_path.write_bytes(
            plistlib.dumps({"ProgramArguments": ["/tmp/missing-sol"]})
        )
        result = doctor.launchd_stale_plist_check(args(doctor))
        assert result.status == "fail"

    def test_ok_when_target_exists(self, doctor, monkeypatch, home_root, tmp_path):
        monkeypatch.setattr(doctor, "platform_tag", lambda: "darwin")
        exe = tmp_path / "sol"
        exe.write_text("", encoding="utf-8")
        plist_path = (
            home_root / "Library" / "LaunchAgents" / "org.solpbc.solstone.plist"
        )
        plist_path.parent.mkdir(parents=True)
        plist_path.write_bytes(plistlib.dumps({"ProgramArguments": [str(exe)]}))
        result = doctor.launchd_stale_plist_check(args(doctor))
        assert result.status == "ok"


class TestTccChecks:
    def test_screen_skip_on_linux(self, doctor, monkeypatch):
        monkeypatch.setattr(doctor, "platform_tag", lambda: "linux")
        result = doctor.screen_recording_permission_check(args(doctor))
        assert result.status == "skip"

    def test_screen_skip_on_darwin(self, doctor, monkeypatch):
        monkeypatch.setattr(doctor, "platform_tag", lambda: "darwin")
        result = doctor.screen_recording_permission_check(args(doctor))
        assert result.status == "skip"
        assert "no adopted non-prompting probe" in result.detail

    def test_microphone_skip_on_linux(self, doctor, monkeypatch):
        monkeypatch.setattr(doctor, "platform_tag", lambda: "linux")
        result = doctor.microphone_permission_check(args(doctor))
        assert result.status == "skip"

    def test_microphone_skip_on_darwin(self, doctor, monkeypatch):
        monkeypatch.setattr(doctor, "platform_tag", lambda: "darwin")
        result = doctor.microphone_permission_check(args(doctor))
        assert result.status == "skip"
        assert "no adopted non-prompting probe" in result.detail


class TestJsonAndExitCodes:
    def test_json_output(self, doctor, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "run_checks",
            lambda _args: [
                doctor.CheckResult("a", "blocker", "ok", "fine", None),
                doctor.CheckResult("b", "advisory", "warn", "careful", "fix me"),
            ],
        )
        rc = doctor.main(["--json"])
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert sorted(payload) == ["checks", "summary"]
        assert set(payload["checks"][0]) == {
            "name",
            "severity",
            "status",
            "detail",
            "fix",
        }

    def test_exit_code_matrix(self, doctor, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "run_checks",
            lambda _args: [doctor.CheckResult("a", "blocker", "fail", "boom", None)],
        )
        assert doctor.main([]) == 1
        capsys.readouterr()

        monkeypatch.setattr(
            doctor,
            "run_checks",
            lambda _args: [doctor.CheckResult("a", "advisory", "fail", "boom", None)],
        )
        assert doctor.main([]) == 0
        capsys.readouterr()

        monkeypatch.setattr(
            doctor,
            "run_checks",
            lambda _args: [doctor.CheckResult("a", "blocker", "skip", "skip", None)],
        )
        assert doctor.main([]) == 0

    def test_summary_line_format(self, doctor, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "run_checks",
            lambda _args: [
                doctor.CheckResult("a", "blocker", "fail", "boom", None),
                doctor.CheckResult("b", "advisory", "warn", "warn", None),
                doctor.CheckResult("c", "blocker", "skip", "skip", None),
            ],
        )
        doctor.main([])
        output = capsys.readouterr().out.strip().splitlines()
        assert output[-1] == "doctor: 3 checks, 1 failed, 1 warnings, 1 skipped"


def test_sol_doctor_subprocess_json_shape():
    """End-to-end: `sol doctor --json` via the venv entry point produces valid diagnostic JSON."""
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "think.sol_cli", "doctor", "--json"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=60,
    )
    # Exit code: 0 if all checks pass, 1 if any blocker fails. Either is valid
    # here; this test asserts CLI routing and payload shape, not machine health.
    assert result.returncode in (
        0,
        1,
    ), f"unexpected exit code {result.returncode}: {result.stderr}"
    payload = json.loads(result.stdout)
    assert "checks" in payload and isinstance(payload["checks"], list)
    assert "summary" in payload and isinstance(payload["summary"], dict)
    assert len(payload["checks"]) >= 1


class TestMakefileIntegration:
    def test_dry_run_install_does_not_run_doctor(self):
        result = subprocess.run(
            ["make", "--dry-run", "-B", "install"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert result.returncode == 0
        lines = result.stdout.splitlines()
        assert all("python3 scripts/doctor.py" not in line for line in lines)
        assert any("uv sync" in line for line in lines)
