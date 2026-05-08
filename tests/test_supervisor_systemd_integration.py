# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc
"""Integration test: supervisor under systemd-user with wrapper-tee sibling.

Reproduces the production process tree that previously triggered a
crash-loop (BrokenPipeError -> exit 120) when the supervisor's startup
cgroup sweep SIGTERMed the wrapper's tee process substitution.

Skipped unless the host has a working systemd --user instance.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path

import pytest


def _systemd_user_available() -> tuple[bool, str]:
    if sys.platform != "linux":
        return False, "not linux"
    if shutil.which("systemctl") is None:
        return False, "systemctl not on PATH"
    try:
        version = subprocess.run(
            ["systemctl", "--user", "--version"],
            capture_output=True,
            timeout=5,
        )
        if version.returncode != 0:
            return False, "systemctl --user --version failed"

        probe = subprocess.run(
            ["systemctl", "--user", "list-units", "--no-pager"],
            capture_output=True,
            timeout=5,
        )
        if probe.returncode != 0:
            return False, "systemctl --user list-units failed (no user instance)"
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"systemctl probe error: {exc}"
    return True, ""


_AVAILABLE, _SKIP_REASON = _systemd_user_available()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _AVAILABLE,
        reason=_SKIP_REASON or "systemd --user unavailable",
    ),
]


@pytest.mark.timeout(60)
def test_supervisor_unit_stays_active_under_wrapper_tee(tmp_path: Path) -> None:
    """A wrapper-with-tee + python child must stay active >=30s under systemd-user."""
    result_file = tmp_path / "result.json"
    fake_supervisor = tmp_path / "fake_supervisor.py"
    fake_supervisor.write_text(
        textwrap.dedent(
            f"""\
            import json
            import os
            import time
            from pathlib import Path

            result = {{"pid": os.getpid()}}
            try:
                cgroup = Path("/proc/self/cgroup").read_text().strip()
                result["cgroup_line"] = cgroup
                tail = cgroup.split(":", 2)[-1] if ":" in cgroup else ""
                if tail.startswith("/"):
                    procs_path = (
                        Path("/sys/fs/cgroup") / tail.lstrip("/") / "cgroup.procs"
                    )
                    if procs_path.exists():
                        pids = [
                            int(value)
                            for value in procs_path.read_text().split()
                            if value
                        ]
                        result["sibling_pids"] = [
                            pid for pid in pids if pid != os.getpid()
                        ]
            except Exception as exc:
                result["error"] = repr(exc)

            Path({str(result_file)!r}).write_text(json.dumps(result))
            print("fake-supervisor: ready", flush=True)
            time.sleep(40)
            """
        )
    )

    log_file = tmp_path / "service.log"
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -e
            exec > >(tee -a {str(log_file)!r}) 2>&1
            exec {sys.executable!r} {str(fake_supervisor)!r}
            """
        )
    )
    wrapper.chmod(0o755)

    unit_name = f"solstone-itest-{uuid.uuid4().hex[:12]}.service"
    launch = subprocess.run(
        [
            "systemd-run",
            "--user",
            f"--unit={unit_name}",
            "--collect",
            "-p",
            "Type=simple",
            "-p",
            "Restart=on-failure",
            "-p",
            "RestartSec=5",
            "-p",
            "KillMode=control-group",
            "-p",
            "TimeoutStopSec=30",
            str(wrapper),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert launch.returncode == 0, (
        f"systemd-run failed: rc={launch.returncode}\n"
        f"stdout={launch.stdout}\nstderr={launch.stderr}"
    )

    try:
        deadline = time.time() + 10
        active = False
        last_stdout = ""
        last_stderr = ""
        while time.time() < deadline:
            active_result = subprocess.run(
                ["systemctl", "--user", "is-active", unit_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            last_stdout = active_result.stdout
            last_stderr = active_result.stderr
            if active_result.stdout.strip() == "active":
                active = True
                break
            time.sleep(0.5)
        assert active, (
            "unit never reached active within 10s "
            f"(last is-active output: {last_stdout!r} stderr: {last_stderr!r})"
        )

        time.sleep(30)
        active_result = subprocess.run(
            ["systemctl", "--user", "is-active", unit_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert active_result.stdout.strip() == "active", (
            "unit was not active after 30s "
            f"stdout={active_result.stdout!r} stderr={active_result.stderr!r}"
        )

        assert result_file.exists(), "fake supervisor never wrote result file"
    finally:
        subprocess.run(
            ["systemctl", "--user", "stop", unit_name],
            capture_output=True,
            timeout=10,
        )
        subprocess.run(
            ["systemctl", "--user", "reset-failed", unit_name],
            capture_output=True,
            timeout=5,
        )
