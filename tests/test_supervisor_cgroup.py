# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import os
import signal
import sys

import pytest

from solstone.think import supervisor

LINUX_ONLY = pytest.mark.skipif(
    sys.platform != "linux", reason="cgroup sweep is Linux-only"
)


def test_parse_self_cgroup_path_v2():
    text = (
        "0::/user.slice/user-1000.slice/user@1000.service/app.slice/solstone.service\n"
    )
    assert (
        supervisor._parse_self_cgroup_path(text)
        == "/user.slice/user-1000.slice/user@1000.service/app.slice/solstone.service"
    )
    assert supervisor._parse_self_cgroup_path("1:name=systemd:/ignored\n") is None
    assert supervisor._parse_self_cgroup_path("0::\n") is None


def test_is_systemd_service_cgroup_matches_solstone_service():
    path = "/user.slice/user-1000.slice/user@1000.service/app.slice/solstone.service"
    assert supervisor._is_systemd_service_cgroup(path) is True


def test_is_systemd_service_cgroup_rejects_terminal_scope():
    path = "/user.slice/user-1000.slice/session-2.scope"
    assert supervisor._is_systemd_service_cgroup(path) is False
    assert supervisor._is_systemd_service_cgroup(None) is False


@LINUX_ONLY
def test_sweep_cgroup_skips_non_systemd_scope(monkeypatch):
    kills = []
    monkeypatch.setattr(supervisor, "_read_self_cgroup_path", lambda: "session-2.scope")
    monkeypatch.setattr(
        supervisor.os, "kill", lambda pid, sig: kills.append((pid, sig))
    )

    assert supervisor._sweep_cgroup_at_startup() == 0
    assert kills == []


@LINUX_ONLY
def test_sweep_cgroup_terms_and_kills_survivors(monkeypatch):
    own_pid = os.getpid()
    kills = []

    monkeypatch.setattr(
        supervisor,
        "_read_self_cgroup_path",
        lambda: (
            "/user.slice/user-1000.slice/user@1000.service/app.slice/solstone.service"
        ),
    )
    monkeypatch.setattr(
        supervisor.Path,
        "read_text",
        lambda self: f"{own_pid}\n111\n222\n",
    )
    monkeypatch.setattr(
        supervisor.os, "kill", lambda pid, sig: kills.append((pid, sig))
    )
    monkeypatch.setattr(supervisor.psutil, "pid_exists", lambda pid: pid == 222)
    monkeypatch.setattr(supervisor.time, "time", lambda: 100.0)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)

    assert supervisor._sweep_cgroup_at_startup(grace=0.0) == 2
    assert kills == [
        (111, signal.SIGTERM),
        (222, signal.SIGTERM),
        (222, signal.SIGKILL),
    ]


class _FakeProcess:
    def __init__(
        self,
        *,
        pid: int,
        name: str = "sol:sense",
        ppid: int = 1,
        username: str = "jer",
        name_error: Exception | None = None,
        ppid_error: Exception | None = None,
        username_error: Exception | None = None,
    ):
        self.pid = pid
        self._name = name
        self._ppid = ppid
        self._username = username
        self._name_error = name_error
        self._ppid_error = ppid_error
        self._username_error = username_error

    def name(self) -> str:
        if self._name_error:
            raise self._name_error
        return self._name

    def ppid(self) -> int:
        if self._ppid_error:
            raise self._ppid_error
        return self._ppid

    def username(self) -> str:
        if self._username_error:
            raise self._username_error
        return self._username


class TestOrphanSweep:
    def _patch_common(self, monkeypatch, procs):
        kills = []
        monkeypatch.setattr(supervisor.sys, "platform", "linux")
        monkeypatch.setattr(supervisor.getpass, "getuser", lambda: "jer")
        monkeypatch.setattr(supervisor.psutil, "process_iter", lambda _attrs: procs)
        monkeypatch.setattr(
            supervisor.os, "kill", lambda pid, sig: kills.append((pid, sig))
        )
        return kills

    def test_matching_targets_are_sigtermed(self, monkeypatch):
        procs = [_FakeProcess(pid=111), _FakeProcess(pid=222, name="sol:convey")]
        kills = self._patch_common(monkeypatch, procs)
        monkeypatch.setattr(supervisor.psutil, "pid_exists", lambda _pid: False)

        assert supervisor._sweep_orphaned_sol_processes() == 2
        assert kills == [(111, signal.SIGTERM), (222, signal.SIGTERM)]

    def test_non_matching_processes_are_ignored(self, monkeypatch):
        monkeypatch.setattr(supervisor.os, "getpid", lambda: 555)
        procs = [
            _FakeProcess(pid=111, username="other"),
            _FakeProcess(pid=222, ppid=2),
            _FakeProcess(pid=333, name="python"),
            _FakeProcess(pid=444, name="solstone:convey"),
            _FakeProcess(pid=555),
        ]
        kills = self._patch_common(monkeypatch, procs)

        assert supervisor._sweep_orphaned_sol_processes() == 0
        assert kills == []

    def test_survivors_after_grace_are_sigkilled(self, monkeypatch):
        procs = [_FakeProcess(pid=111), _FakeProcess(pid=222)]
        kills = self._patch_common(monkeypatch, procs)
        monkeypatch.setattr(supervisor.psutil, "pid_exists", lambda pid: pid == 222)
        monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)

        assert supervisor._sweep_orphaned_sol_processes(grace=0.0) == 2
        assert kills == [
            (111, signal.SIGTERM),
            (222, signal.SIGTERM),
            (222, signal.SIGKILL),
        ]

    def test_process_access_errors_are_swallowed(self, monkeypatch):
        procs = [
            _FakeProcess(pid=111, name_error=supervisor.psutil.NoSuchProcess(pid=111)),
            _FakeProcess(
                pid=222,
                username_error=supervisor.psutil.AccessDenied(pid=222),
            ),
            _FakeProcess(pid=333),
        ]
        kills = self._patch_common(monkeypatch, procs)
        monkeypatch.setattr(supervisor.psutil, "pid_exists", lambda _pid: False)

        assert supervisor._sweep_orphaned_sol_processes() == 1
        assert kills == [(333, signal.SIGTERM)]

    def test_non_linux_skips_without_killing(self, monkeypatch):
        kills = []
        monkeypatch.setattr(supervisor.sys, "platform", "darwin")
        monkeypatch.setattr(
            supervisor.psutil,
            "process_iter",
            lambda _attrs: pytest.fail("process_iter should not be called"),
        )
        monkeypatch.setattr(
            supervisor.os, "kill", lambda pid, sig: kills.append((pid, sig))
        )

        assert supervisor._sweep_orphaned_sol_processes() == 0
        assert kills == []
