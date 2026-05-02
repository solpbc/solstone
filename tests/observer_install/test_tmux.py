# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import subprocess

from observe.observer_install import common, tmux


def test_tmux_happy_path_writes_config_and_marker(monkeypatch, args_factory):
    monkeypatch.setattr(tmux, "poll_status_until", lambda name: "connected")

    def fake_probe(cmd, *, cwd=None):
        text = " ".join(cmd)
        if text == "git rev-parse HEAD":
            return subprocess.CompletedProcess(cmd, 0, "tmuxsha\n", "")
        return subprocess.CompletedProcess(cmd, 0, "ok\n", "")

    steps: list[str] = []

    def fake_step(label, cmd, **kwargs):
        steps.append(label)
        return common.StepResult(subprocess.CompletedProcess(cmd, 0, "", ""))

    monkeypatch.setattr(tmux, "run_probe", fake_probe)
    monkeypatch.setattr(tmux, "run_step", fake_step)

    assert tmux.TmuxDriver().run(args_factory(platform="tmux")) == 0

    assert "run make install-service" in steps
    config = json.loads(tmux.CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["stream"] == "archon"
    assert config["status_indicator"] is True
    assert common.read_marker(tmux.INSTALL_NAME)["version"] == "tmuxsha"


def test_tmux_missing_tmux_warns_but_continues(monkeypatch, args_factory, capsys):
    monkeypatch.setattr(tmux, "poll_status_until", lambda name: "connected")

    def fake_probe(cmd, *, cwd=None):
        text = " ".join(cmd)
        if text == "sh -c command -v tmux":
            return subprocess.CompletedProcess(cmd, 1, "", "")
        if text == "git rev-parse HEAD":
            return subprocess.CompletedProcess(cmd, 0, "tmuxsha\n", "")
        return subprocess.CompletedProcess(cmd, 0, "ok\n", "")

    monkeypatch.setattr(tmux, "run_probe", fake_probe)
    monkeypatch.setattr(
        tmux,
        "run_step",
        lambda label, cmd, **kwargs: common.StepResult(
            subprocess.CompletedProcess(cmd, 0, "", "")
        ),
    )

    assert tmux.TmuxDriver().run(args_factory(platform="tmux")) == 0

    assert "warning: tmux not detected on PATH" in capsys.readouterr().out


def test_tmux_missing_uv_raises(monkeypatch, args_factory):
    def fake_probe(cmd, *, cwd=None):
        text = " ".join(cmd)
        code = 1 if "command -v uv" in text else 0
        return subprocess.CompletedProcess(cmd, code, "", "")

    monkeypatch.setattr(tmux, "run_probe", fake_probe)

    try:
        tmux.TmuxDriver().run(args_factory(platform="tmux"))
    except common.InstallError as exc:
        assert "missing required tool: uv" in str(exc)
        assert "https://docs.astral.sh/uv/" in exc.hint
    else:
        raise AssertionError("expected InstallError")


def test_tmux_second_run_after_install_is_noop(monkeypatch, args_factory, capsys):
    monkeypatch.setattr(tmux, "poll_status_until", lambda name: "connected")

    def fake_probe(cmd, *, cwd=None):
        text = " ".join(cmd)
        if text == "git remote get-url origin":
            return subprocess.CompletedProcess(cmd, 0, f"{tmux.SOURCE_URL}\n", "")
        if text == "git status --porcelain":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if text in {"git rev-parse HEAD", "git rev-parse @{u}"}:
            return subprocess.CompletedProcess(cmd, 0, "tmuxsha\n", "")
        if text == f"systemctl --user is-active {tmux.UNIT_NAME}":
            return subprocess.CompletedProcess(cmd, 0, "active\n", "")
        return subprocess.CompletedProcess(cmd, 0, "ok\n", "")

    steps: list[str] = []

    def fake_step(label, cmd, **kwargs):
        steps.append(label)
        if label.startswith("clone "):
            (common.xdg_install_dir(tmux.INSTALL_NAME) / ".git").mkdir(parents=True)
        return common.StepResult(subprocess.CompletedProcess(cmd, 0, "", ""))

    config_writes = 0
    marker_writes = 0
    original_write_config = tmux._write_config
    original_write_marker = tmux.write_marker

    def count_config_write(server_url, key, name):
        nonlocal config_writes
        config_writes += 1
        original_write_config(server_url, key, name)

    def count_marker_write(install_name, data):
        nonlocal marker_writes
        marker_writes += 1
        original_write_marker(install_name, data)

    monkeypatch.setattr(tmux, "run_probe", fake_probe)
    monkeypatch.setattr(tmux, "run_step", fake_step)
    monkeypatch.setattr(tmux, "_write_config", count_config_write)
    monkeypatch.setattr(tmux, "write_marker", count_marker_write)

    assert tmux.TmuxDriver().run(args_factory(platform="tmux")) == 0
    assert tmux.TmuxDriver().run(args_factory(platform="tmux")) == 0

    assert steps.count("run make install-service") == 1
    assert config_writes == 1
    assert marker_writes == 1
    assert "already installed" in capsys.readouterr().out
