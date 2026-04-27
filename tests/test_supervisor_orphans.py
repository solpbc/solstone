# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import signal

import psutil

from think import supervisor


class FakeProcess:
    def __init__(self, pid: int, name: str, ppid: int):
        self._info = {"pid": pid, "name": name, "ppid": ppid}

    @property
    def info(self):
        return self._info


class VanishedProcess:
    @property
    def info(self):
        raise psutil.NoSuchProcess(pid=4)


def test_find_reparented_sol_workers_filters_name_and_ppid(monkeypatch):
    monkeypatch.setattr(supervisor.sys, "platform", "linux")
    monkeypatch.setattr(
        supervisor.psutil,
        "process_iter",
        lambda attrs: [
            FakeProcess(100, "sol:cortex", 1),
            FakeProcess(101, "sol:cortex", 999),
            FakeProcess(102, "bash", 1),
            VanishedProcess(),
        ],
    )

    assert supervisor._find_reparented_sol_workers() == [(100, "sol:cortex")]


def test_reap_orphan_workers_empty_is_quiet(monkeypatch):
    logs = []
    monkeypatch.setattr(supervisor.os, "kill", lambda *_args: logs.append("kill"))
    monkeypatch.setattr(
        supervisor.logging, "warning", lambda *_args: logs.append("log")
    )
    monkeypatch.setattr(
        supervisor.logging, "exception", lambda *_args: logs.append("log")
    )

    assert supervisor._reap_orphan_workers([]) == 0
    assert logs == []


def test_reap_orphan_workers_sigterms_then_sigkills_survivors(monkeypatch):
    calls = []
    times = iter([0.0, 0.0, 0.06])

    monkeypatch.setattr(
        supervisor.os, "kill", lambda pid, sig: calls.append((pid, sig))
    )
    monkeypatch.setattr(supervisor.time, "time", lambda: next(times))
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(supervisor.psutil, "pid_exists", lambda pid: pid == 100)

    assert (
        supervisor._reap_orphan_workers(
            [(100, "sol:cortex"), (101, "sol:sense")], grace=0.05
        )
        == 2
    )
    assert calls == [
        (100, signal.SIGTERM),
        (101, signal.SIGTERM),
        (100, signal.SIGKILL),
    ]


def test_find_reparented_sol_workers_noop_on_macos(monkeypatch):
    monkeypatch.setattr(supervisor.sys, "platform", "darwin")
    monkeypatch.setattr(
        supervisor.psutil,
        "process_iter",
        lambda _attrs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )

    assert supervisor._find_reparented_sol_workers() == []
