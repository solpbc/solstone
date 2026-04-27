# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc
"""Canary tests for the root conftest TMPDIR fallback prelude."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_ROOT_CONFTEST = Path(__file__).resolve().parent.parent / "conftest.py"
_NOTICE = (
    "solstone: pytest invoked without TMPDIR export; routing tmp dirs to "
    "/var/tmp. Prefer 'make test' to set TMPDIR at the shell level.\n"
)


def _install_root_conftest(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(conftest=_ROOT_CONFTEST.read_text(encoding="utf-8"))


def _run_nested(
    pytester: pytest.Pytester,
    env_overrides: dict[str, str | None] | None = None,
) -> subprocess.CompletedProcess[str]:
    basetemp = pytester.path / "basetemp"
    env = os.environ.copy()
    env.pop("TMPDIR", None)
    env.pop("_SOLSTONE_TMPDIR_FALLBACK_NOTIFIED", None)
    env.pop("_SOLSTONE_TMPDIR_FALLBACK_TARGET", None)
    if env_overrides:
        for key, value in env_overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(basetemp),
        ],
        cwd=pytester.path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_prelude_redirects_to_var_tmp(pytester: pytest.Pytester) -> None:
    _install_root_conftest(pytester)
    pytester.makepyfile(
        test_tmpdir="""
        import os
        import tempfile

        def test_redirects_to_var_tmp():
            assert tempfile.gettempdir() == "/var/tmp"
            assert os.environ["TMPDIR"] == "/var/tmp"
        """
    )

    result = _run_nested(pytester)

    assert result.returncode == 0, result.stderr + result.stdout
    assert "1 passed" in result.stdout


def test_notice_present_when_tmpdir_unset(pytester: pytest.Pytester) -> None:
    _install_root_conftest(pytester)
    pytester.makepyfile(
        test_notice="""
        def test_noop():
            assert True
        """
    )

    result = _run_nested(pytester)
    combined = result.stderr + result.stdout

    assert result.returncode == 0, combined
    assert _NOTICE in combined


def test_notice_absent_when_tmpdir_already_set(pytester: pytest.Pytester) -> None:
    _install_root_conftest(pytester)
    pytester.makepyfile(
        test_notice="""
        import os
        import tempfile

        def test_uses_existing_tmpdir():
            assert tempfile.gettempdir() == "/tmp"
            assert os.environ["TMPDIR"] == "/tmp"
        """
    )

    result = _run_nested(pytester, {"TMPDIR": "/tmp"})
    combined = result.stderr + result.stdout

    assert result.returncode == 0, combined
    assert "solstone: pytest invoked without TMPDIR" not in combined


def test_notice_single_fire_across_workers(pytester: pytest.Pytester) -> None:
    _install_root_conftest(pytester)
    pytester.makepyfile(
        test_notice="""
        def test_noop():
            assert True
        """
    )

    first = _run_nested(pytester)
    second = _run_nested(pytester, {"_SOLSTONE_TMPDIR_FALLBACK_NOTIFIED": "1"})
    combined = first.stderr + first.stdout + second.stderr + second.stdout

    assert first.returncode == 0, combined
    assert second.returncode == 0, combined
    assert combined.count(_NOTICE) == 1


def test_unwritable_target_degrades_visibly(
    pytester: pytest.Pytester, tmp_path: Path
) -> None:
    _install_root_conftest(pytester)
    pytester.makepyfile(
        test_notice="""
        import os

        def test_tmpdir_stays_unset():
            assert os.environ.get("TMPDIR") is None
        """
    )
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    os.chmod(blocked, 0)

    try:
        result = _run_nested(
            pytester, {"_SOLSTONE_TMPDIR_FALLBACK_TARGET": str(blocked)}
        )
    finally:
        os.chmod(blocked, 0o700)

    combined = result.stderr + result.stdout
    notice = (
        "solstone: pytest invoked without TMPDIR export and fallback target "
        f"{blocked} is not writable; leaving TMPDIR unset.\n"
    )

    assert result.returncode == 0, combined
    assert notice in combined
    assert "routing tmp dirs to" not in combined


def test_subprocess_pytest_lands_in_var_tmp(pytester: pytest.Pytester) -> None:
    _install_root_conftest(pytester)
    test_path = pytester.makepyfile(
        test_subprocess="""
        import os
        import tempfile

        def test_redirects_in_fresh_subprocess():
            assert tempfile.gettempdir() == "/var/tmp"
            assert os.environ["TMPDIR"] == "/var/tmp"
        """
    )
    env = os.environ.copy()
    env.pop("TMPDIR", None)
    env.pop("_SOLSTONE_TMPDIR_FALLBACK_NOTIFIED", None)
    env.pop("_SOLSTONE_TMPDIR_FALLBACK_TARGET", None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(pytester.path / "basetemp"),
            str(test_path),
        ],
        cwd=pytester.path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "1 passed" in result.stdout
