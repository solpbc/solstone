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
