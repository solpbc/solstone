# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _read_config(journal_copy) -> dict:
    return json.loads((journal_copy / "config" / "journal.json").read_text("utf-8"))


def _write_config(journal_copy, payload: dict) -> None:
    (journal_copy / "config" / "journal.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _detect_lan_ipv4() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            host = sock.getsockname()[0]
    except OSError:
        return None
    return host if host and not host.startswith("127.") else None


def _launch_convey(journal_copy, port: int) -> subprocess.Popen[str]:
    code = (
        f"from convey import create_app; "
        f"from convey.cli import _resolve_bind_host, run_service; "
        f"app = create_app({str(journal_copy)!r}); "
        f"run_service(app, host=_resolve_bind_host(), port={port}, start_watcher=False)"
    )
    env = os.environ.copy()
    env["_SOLSTONE_JOURNAL_OVERRIDE"] = str(journal_copy)
    env["SOL_SKIP_SUPERVISOR_CHECK"] = "1"
    return subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_connect(
    host: str, port: int, *, should_succeed: bool, timeout: float = 10.0
):
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            try:
                sock.connect((host, port))
                return True
            except OSError as exc:
                last_error = exc
        time.sleep(0.1)
    if should_succeed:
        raise AssertionError(f"failed to connect to {host}:{port}: {last_error}")
    return False


@contextlib.contextmanager
def _running_convey(journal_copy, *, allow_network_access: bool):
    payload = _read_config(journal_copy)
    payload["convey"]["allow_network_access"] = allow_network_access
    _write_config(journal_copy, payload)
    port = _free_port()
    process = _launch_convey(journal_copy, port)
    try:
        _wait_for_connect("127.0.0.1", port, should_succeed=True)
        yield port
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if process.returncode not in (0, -15):
            stdout, stderr = process.communicate(timeout=1)
            raise AssertionError(
                f"convey exited unexpectedly ({process.returncode})\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only socket probe")
def test_convey_binds_localhost_only_when_network_access_disabled(journal_copy):
    lan_ip = _detect_lan_ipv4()
    if lan_ip is None:
        pytest.skip("no non-loopback IPv4 available for bind probe")

    with _running_convey(journal_copy, allow_network_access=False) as port:
        assert _wait_for_connect("127.0.0.1", port, should_succeed=True) is True
        assert _wait_for_connect(lan_ip, port, should_succeed=False) is False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only socket probe")
def test_convey_binds_all_interfaces_when_network_access_enabled(journal_copy):
    lan_ip = _detect_lan_ipv4()
    if lan_ip is None:
        pytest.skip("no non-loopback IPv4 available for bind probe")

    with _running_convey(journal_copy, allow_network_access=True) as port:
        assert _wait_for_connect("127.0.0.1", port, should_succeed=True) is True
        assert _wait_for_connect(lan_ip, port, should_succeed=True) is True
