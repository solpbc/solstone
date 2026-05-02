# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from apps.observer.utils import list_observers, save_observer
from observe.observer_install import common, linux


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("ID=fedora\n", "fedora"),
        ("ID=ubuntu\n", "debian-ubuntu"),
        ("ID=arch\n", "arch"),
        ("ID=opensuse-tumbleweed\n", "opensuse"),
        ("ID=unknown\nID_LIKE=debian\n", "debian-ubuntu"),
    ],
)
def test_detect_distro_from_os_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str, expected: str
):
    os_release = tmp_path / "os-release"
    os_release.write_text(content, encoding="utf-8")
    monkeypatch.setattr(linux, "OS_RELEASE_PATH", os_release)

    assert linux.detect_distro() == expected


def test_missing_uv_raises(monkeypatch: pytest.MonkeyPatch, args_factory):
    monkeypatch.setattr(linux, "detect_distro", lambda: "fedora")

    def fake_probe(cmd, *, cwd=None):
        text = " ".join(cmd)
        code = 1 if "command -v uv" in text else 0
        return subprocess.CompletedProcess(cmd, code, "", "")

    monkeypatch.setattr(linux, "run_probe", fake_probe)

    with pytest.raises(common.InstallError) as exc_info:
        linux.LinuxDriver().run(args_factory())

    assert "missing required tool: uv" in str(exc_info.value)
    assert "https://docs.astral.sh/uv/" in exc_info.value.hint


def test_missing_system_package_reports_install_command(
    monkeypatch: pytest.MonkeyPatch, args_factory
):
    monkeypatch.setattr(linux, "detect_distro", lambda: "fedora")

    def fake_probe(cmd, *, cwd=None):
        text = " ".join(cmd)
        if text == "rpm -q gtk4":
            return subprocess.CompletedProcess(cmd, 1, "", "")
        return subprocess.CompletedProcess(cmd, 0, "ok\n", "")

    monkeypatch.setattr(linux, "run_probe", fake_probe)

    with pytest.raises(common.InstallError) as exc_info:
        linux.LinuxDriver().run(args_factory())

    assert "gtk4" in str(exc_info.value)
    assert (
        "sudo dnf install python3-gobject gtk4 gstreamer1-plugins-base"
        in exc_info.value.hint
    )


def test_happy_path_writes_config_and_marker(
    monkeypatch: pytest.MonkeyPatch, args_factory
):
    monkeypatch.setattr(linux, "detect_distro", lambda: "fedora")
    monkeypatch.setattr(linux, "poll_status_until", lambda name: "connected")

    def fake_probe(cmd, *, cwd=None):
        text = " ".join(cmd)
        if text == "git rev-parse HEAD":
            return subprocess.CompletedProcess(cmd, 0, "abc123\n", "")
        if text == "git remote get-url origin":
            return subprocess.CompletedProcess(cmd, 0, f"{linux.SOURCE_URL}\n", "")
        return subprocess.CompletedProcess(cmd, 0, "ok\n", "")

    steps: list[str] = []

    def fake_step(label, cmd, **kwargs):
        steps.append(label)
        return common.StepResult(subprocess.CompletedProcess(cmd, 0, "", ""))

    monkeypatch.setattr(linux, "run_probe", fake_probe)
    monkeypatch.setattr(linux, "run_step", fake_step)

    assert linux.LinuxDriver().run(args_factory()) == 0

    assert "run make install-service" in steps
    config = json.loads(linux.CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["server_url"] == "http://127.0.0.1:5015"
    assert config["stream"] == "archon"
    assert config["key"]
    marker = common.read_marker(linux.INSTALL_NAME)
    assert marker["name"] == "archon"
    assert marker["version"] == "abc123"


def test_marker_present_no_upstream_changes_is_noop(
    monkeypatch: pytest.MonkeyPatch, observer_install_env, args_factory, capsys
):
    monkeypatch.setattr(linux, "detect_distro", lambda: "fedora")
    clone_dir = common.xdg_install_dir(linux.INSTALL_NAME)
    (clone_dir / ".git").mkdir(parents=True)
    common.write_marker(
        linux.INSTALL_NAME,
        {
            "name": "archon",
            "platform": "linux",
            "source": linux.SOURCE_URL,
            "installed_at": "2026-05-02T00:00:00Z",
            "last_run": "2026-05-02T00:00:00Z",
            "version": "abc123",
        },
    )
    save_observer(
        {
            "key": "abcdefgh",
            "name": "archon",
            "created_at": None,
            "last_seen": None,
            "last_segment": None,
            "enabled": True,
            "stats": {"segments_received": 0, "bytes_received": 0},
        }
    )
    linux.CONFIG_PATH.parent.mkdir(parents=True)
    linux.CONFIG_PATH.write_text('{"key": "abcdefgh"}\n', encoding="utf-8")

    def fake_probe(cmd, *, cwd=None):
        text = " ".join(cmd)
        if text == "git remote get-url origin":
            return subprocess.CompletedProcess(cmd, 0, f"{linux.SOURCE_URL}\n", "")
        if text == "git status --porcelain":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if text in {"git rev-parse HEAD", "git rev-parse @{u}"}:
            return subprocess.CompletedProcess(cmd, 0, "abc123\n", "")
        if text == f"systemctl --user is-active {linux.UNIT_NAME}":
            return subprocess.CompletedProcess(cmd, 0, "active\n", "")
        return subprocess.CompletedProcess(cmd, 0, "ok\n", "")

    steps: list[str] = []

    def fake_step(label, cmd, **kwargs):
        steps.append(label)
        return common.StepResult(subprocess.CompletedProcess(cmd, 0, "", ""))

    monkeypatch.setattr(linux, "run_probe", fake_probe)
    monkeypatch.setattr(linux, "run_step", fake_step)

    assert linux.LinuxDriver().run(args_factory()) == 0

    assert "run make install-service" not in steps
    assert "already installed" in capsys.readouterr().out
    assert common.read_marker(linux.INSTALL_NAME)["last_run"] == "2026-05-02T00:00:00Z"


def test_second_run_after_install_is_noop(
    monkeypatch: pytest.MonkeyPatch, args_factory, capsys
):
    monkeypatch.setattr(linux, "detect_distro", lambda: "fedora")
    monkeypatch.setattr(linux, "poll_status_until", lambda name: "connected")

    def fake_probe(cmd, *, cwd=None):
        text = " ".join(cmd)
        if text == "git remote get-url origin":
            return subprocess.CompletedProcess(cmd, 0, f"{linux.SOURCE_URL}\n", "")
        if text == "git status --porcelain":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if text in {"git rev-parse HEAD", "git rev-parse @{u}"}:
            return subprocess.CompletedProcess(cmd, 0, "abc123\n", "")
        if text == f"systemctl --user is-active {linux.UNIT_NAME}":
            return subprocess.CompletedProcess(cmd, 0, "active\n", "")
        return subprocess.CompletedProcess(cmd, 0, "ok\n", "")

    steps: list[str] = []

    def fake_step(label, cmd, **kwargs):
        steps.append(label)
        if label.startswith("clone "):
            (common.xdg_install_dir(linux.INSTALL_NAME) / ".git").mkdir(parents=True)
        return common.StepResult(subprocess.CompletedProcess(cmd, 0, "", ""))

    config_writes = 0
    marker_writes = 0
    original_write_config = linux._write_config
    original_write_marker = linux.write_marker

    def count_config_write(server_url, key, name):
        nonlocal config_writes
        config_writes += 1
        original_write_config(server_url, key, name)

    def count_marker_write(install_name, data):
        nonlocal marker_writes
        marker_writes += 1
        original_write_marker(install_name, data)

    monkeypatch.setattr(linux, "run_probe", fake_probe)
    monkeypatch.setattr(linux, "run_step", fake_step)
    monkeypatch.setattr(linux, "_write_config", count_config_write)
    monkeypatch.setattr(linux, "write_marker", count_marker_write)

    assert linux.LinuxDriver().run(args_factory()) == 0
    assert linux.LinuxDriver().run(args_factory()) == 0

    assert steps.count("run make install-service") == 1
    assert config_writes == 1
    assert marker_writes == 1
    assert "already installed" in capsys.readouterr().out


def test_force_revokes_and_recreates_registration(monkeypatch: pytest.MonkeyPatch):
    save_observer(
        {
            "key": "old-key",
            "name": "archon",
            "created_at": 1,
            "last_seen": None,
            "last_segment": None,
            "enabled": True,
            "stats": {"segments_received": 0, "bytes_received": 0},
        }
    )
    monkeypatch.setattr("observe.observer_cli._generate_key", lambda: "new-key")

    result = common.create_or_reuse_registration("archon", force=True)

    observers = list_observers()
    assert result.key == "new-key"
    assert any(
        observer.get("key") == "old-key" and observer.get("revoked")
        for observer in observers
    )
    assert any(
        observer.get("key") == "new-key" and not observer.get("revoked")
        for observer in observers
    )
