# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import subprocess
from pathlib import Path

from solstone.observe.observer_install import linux, tmux


def test_linux_dry_run_snapshot(monkeypatch, args_factory, capsys):
    monkeypatch.setattr(Path, "home", lambda: Path("/home/jer"))
    monkeypatch.setattr(
        linux,
        "CONFIG_PATH",
        Path("/home/jer/.local/share/solstone-linux/config/config.json"),
    )
    monkeypatch.setattr(linux, "detect_distro", lambda: "fedora")
    monkeypatch.setattr(
        linux,
        "run_probe",
        lambda cmd, cwd=None: subprocess.CompletedProcess(cmd, 0, "ok\n", ""),
    )

    assert linux.LinuxDriver().run(args_factory(dry_run=True)) == 0

    expected = Path("tests/observer_install/snapshots/linux_dry_run.txt").read_text(
        encoding="utf-8"
    )
    assert capsys.readouterr().out == expected


def test_tmux_dry_run_snapshot(monkeypatch, args_factory, capsys):
    monkeypatch.setattr(Path, "home", lambda: Path("/home/jer"))
    monkeypatch.setattr(
        tmux,
        "CONFIG_PATH",
        Path("/home/jer/.local/share/solstone-tmux/config/config.json"),
    )
    monkeypatch.setattr(
        tmux,
        "run_probe",
        lambda cmd, cwd=None: subprocess.CompletedProcess(cmd, 0, "ok\n", ""),
    )

    assert tmux.TmuxDriver().run(args_factory(platform="tmux", dry_run=True)) == 0

    expected = Path("tests/observer_install/snapshots/tmux_dry_run.txt").read_text(
        encoding="utf-8"
    )
    assert capsys.readouterr().out == expected


def test_dry_run_writes_no_files(monkeypatch, observer_install_env, args_factory):
    monkeypatch.setattr(linux, "detect_distro", lambda: "fedora")
    monkeypatch.setattr(
        linux,
        "run_probe",
        lambda cmd, cwd=None: subprocess.CompletedProcess(cmd, 0, "ok\n", ""),
    )

    assert linux.LinuxDriver().run(args_factory(dry_run=True)) == 0

    assert not (observer_install_env.home / ".local" / "share" / "solstone").exists()
    assert not linux.CONFIG_PATH.exists()
