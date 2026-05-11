# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import contextlib
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import requests
from werkzeug.security import generate_password_hash
from werkzeug.serving import make_server

from solstone.think.utils import write_service_port

RELAY_URL = "https://spl-relay-staging.jer-3f2.workers.dev"
CONVEY_PASSWORD = "pytest-link-pass"
_READY_LINE = "listen WS open"


def skip_unless_live_relay() -> None:
    if os.environ.get("SPL_RELAY_LIVE_TESTS", "1") == "0":
        pytest.skip(
            "SPL_RELAY_LIVE_TESTS=0; skipping live relay tests",
            allow_module_level=True,
        )
    try:
        response = requests.get(f"{RELAY_URL}/", timeout=5)
        if response.status_code != 200:
            pytest.skip(
                f"relay unreachable: status {response.status_code}",
                allow_module_level=True,
            )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"relay unreachable: {exc}", allow_module_level=True)


class LinkProcessCapture:
    def __init__(self, proc: subprocess.Popen[str]) -> None:
        self.proc = proc
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._threads = [
            threading.Thread(
                target=self._drain,
                args=(proc.stdout, self.stdout_lines, "stdout", self._queue),
                daemon=True,
            ),
            threading.Thread(
                target=self._drain,
                args=(proc.stderr, self.stderr_lines, "stderr", self._queue),
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def wait_for_line(self, needle: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(needle in line for line in self.stderr_lines):
                return
            if self.proc.poll() is not None:
                break
            remaining = max(0.0, deadline - time.monotonic())
            try:
                stream, line = self._queue.get(timeout=min(0.25, remaining))
            except queue.Empty:
                continue
            if stream == "stderr" and needle in line:
                return
        raise RuntimeError(
            f"link service never emitted {needle!r}; stderr tail:\n"
            + "".join(self.stderr_lines[-50:])
        )

    @property
    def stdout_text(self) -> str:
        return "".join(self.stdout_lines)

    @property
    def stderr_text(self) -> str:
        return "".join(self.stderr_lines)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        for pipe in (self.proc.stdout, self.proc.stderr):
            if pipe is not None:
                pipe.close()
        for thread in self._threads:
            thread.join(timeout=1)

    @staticmethod
    def _drain(
        pipe: Any,
        target: list[str],
        stream: str,
        line_queue: queue.Queue[tuple[str, str]],
    ) -> None:
        if pipe is None:
            return
        try:
            for line in pipe:
                target.append(line)
                line_queue.put((stream, line))
        finally:
            return


@contextlib.contextmanager
def running_convey_server(
    journal_path: Path,
    *,
    configure_app: Callable[[Any], None] | None = None,
) -> Iterator[str]:
    from solstone.convey import create_app
    from solstone.convey.secure_listener import (
        start_secure_listener,
        stop_secure_listener,
    )

    _prepare_journal(journal_path)
    app = create_app(str(journal_path))
    if configure_app is not None:
        configure_app(app)
    app.config["SECURE_LISTENER_ENABLED"] = True
    start_secure_listener(app)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    write_service_port("convey", server.server_port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _wait_for_tcp_port("127.0.0.1", server.server_port)
    _wait_for_tcp_port("127.0.0.1", 7657)
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        stop_secure_listener(app)
        server.shutdown()
        thread.join(timeout=5)


@contextlib.contextmanager
def running_link_service(
    journal_path: Path,
    *,
    relay_url: str = RELAY_URL,
) -> Iterator[LinkProcessCapture]:
    _prepare_journal(journal_path)
    repo_root = Path(__file__).resolve().parents[2]
    sol_bin = Path(sys.executable).with_name("sol")
    env = os.environ.copy()
    env["SOLSTONE_JOURNAL"] = str(journal_path)
    env["SOL_LINK_RELAY_URL"] = relay_url
    env["SOL_SKIP_SUPERVISOR_CHECK"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = f"{repo_root / '.venv' / 'bin'}:{env.get('PATH', '')}"
    proc = subprocess.Popen(
        [str(sol_bin), "link", "-v"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    capture = LinkProcessCapture(proc)
    try:
        capture.wait_for_line(_READY_LINE, timeout=15)
        yield capture
    finally:
        capture.stop()


def list_devices(base_url: str) -> list[dict[str, Any]]:
    response = requests.get(f"{base_url}/app/link/api/devices", timeout=10)
    response.raise_for_status()
    payload = response.json()
    assert isinstance(payload, dict)
    devices = payload.get("devices")
    assert isinstance(devices, list)
    return devices


def unpair_device(base_url: str, fingerprint: str) -> dict[str, Any]:
    response = requests.post(
        f"{base_url}/app/link/unpair",
        json={"fingerprint": fingerprint},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    assert isinstance(payload, dict)
    return payload


def runtime_texts(
    journal_path: Path,
    capture: LinkProcessCapture,
) -> dict[str, str]:
    out = {
        "stdout": capture.stdout_text,
        "stderr": capture.stderr_text,
    }
    extra_paths = [journal_path / "health" / "supervisor.log"]
    extra_paths.extend(sorted((journal_path / "link").glob("*.log")))
    for path in extra_paths:
        if path.exists():
            out[str(path.relative_to(journal_path))] = path.read_text("utf-8")
    return out


def _prepare_journal(journal_path: Path) -> None:
    config_path = journal_path / "config" / "journal.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "convey": {
                    "password_hash": generate_password_hash(CONVEY_PASSWORD),
                    "trust_localhost": True,
                },
                "setup": {"completed_at": 1},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _wait_for_tcp_port(host: str, port: int) -> None:
    deadline = time.monotonic() + 2.0
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"port {host}:{port} did not become ready: {last_error}")
