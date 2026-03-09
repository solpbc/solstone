# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for desktop session environment recovery."""

import os
import subprocess
from unittest.mock import patch

from observe.linux.observer import _recover_session_env, check_session_ready


class TestRecoverSessionEnv:
    """Tests for _recover_session_env()."""

    def test_noop_when_vars_already_set(self, monkeypatch):
        """Should not call systemctl when all vars are present."""
        monkeypatch.setenv("DISPLAY", ":1")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")

        with patch("observe.linux.observer.subprocess.run") as mock_run:
            _recover_session_env()
            mock_run.assert_not_called()

    def test_recovers_missing_vars(self, monkeypatch):
        """Should recover missing vars from systemctl output."""
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")

        systemctl_output = (
            "HOME=/home/user\n"
            "DISPLAY=:0\n"
            "WAYLAND_DISPLAY=wayland-0\n"
            "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus\n"
            "XDG_SESSION_TYPE=wayland\n"
        )
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=systemctl_output, stderr=""
        )
        with patch("observe.linux.observer.subprocess.run", return_value=mock_result):
            _recover_session_env()

        assert os.environ.get("DISPLAY") == ":0"
        assert os.environ.get("WAYLAND_DISPLAY") == "wayland-0"
        assert (
            os.environ.get("DBUS_SESSION_BUS_ADDRESS") == "unix:path=/run/user/1000/bus"
        )

    def test_recovers_only_missing_vars(self, monkeypatch):
        """Should not overwrite vars that are already set."""
        monkeypatch.setenv("DISPLAY", ":5")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")

        systemctl_output = "DISPLAY=:0\nWAYLAND_DISPLAY=wayland-0\n"
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=systemctl_output, stderr=""
        )
        with patch("observe.linux.observer.subprocess.run", return_value=mock_result):
            _recover_session_env()

        assert os.environ.get("DISPLAY") == ":5"  # unchanged
        assert os.environ.get("WAYLAND_DISPLAY") == "wayland-0"  # recovered

    def test_sets_xdg_runtime_dir_if_missing(self, monkeypatch):
        """Should set XDG_RUNTIME_DIR from uid when missing."""
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="DISPLAY=:0\n", stderr=""
        )
        with patch("observe.linux.observer.subprocess.run", return_value=mock_result):
            _recover_session_env()

        assert os.environ.get("XDG_RUNTIME_DIR") == f"/run/user/{os.getuid()}"

    def test_handles_systemctl_failure(self, monkeypatch):
        """Should silently handle systemctl failure."""
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error"
        )
        with patch("observe.linux.observer.subprocess.run", return_value=mock_result):
            _recover_session_env()

        assert not os.environ.get("DISPLAY")

    def test_handles_systemctl_not_found(self, monkeypatch):
        """Should silently handle missing systemctl binary."""
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")

        with patch(
            "observe.linux.observer.subprocess.run", side_effect=FileNotFoundError
        ):
            _recover_session_env()

        assert not os.environ.get("DISPLAY")

    def test_handles_systemctl_timeout(self, monkeypatch):
        """Should silently handle systemctl timeout."""
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")

        with patch(
            "observe.linux.observer.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="systemctl", timeout=5),
        ):
            _recover_session_env()

        assert not os.environ.get("DISPLAY")

    def test_ignores_empty_values(self, monkeypatch):
        """Should not set vars with empty values."""
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")

        systemctl_output = "DISPLAY=\nWAYLAND_DISPLAY=wayland-0\n"
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=systemctl_output, stderr=""
        )
        with patch("observe.linux.observer.subprocess.run", return_value=mock_result):
            _recover_session_env()

        assert not os.environ.get("DISPLAY")
        assert os.environ.get("WAYLAND_DISPLAY") == "wayland-0"


class TestCheckSessionReady:
    """Tests for check_session_ready() with env recovery integration."""

    def test_ready_after_recovery(self, monkeypatch):
        """Should pass after recovering vars from systemd."""
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")

        systemctl_output = (
            "DISPLAY=:0\n"
            "WAYLAND_DISPLAY=wayland-0\n"
            "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus\n"
        )
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=systemctl_output, stderr=""
        )
        with (
            patch("observe.linux.observer.subprocess.run", return_value=mock_result),
            patch("observe.linux.observer.shutil.which", return_value=None),
        ):
            result = check_session_ready()

        assert result is None

    def test_fails_when_recovery_incomplete(self, monkeypatch):
        """Should fail when recovery doesn't provide display vars."""
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")

        # systemctl returns nothing useful
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="HOME=/home/user\n", stderr=""
        )
        with patch("observe.linux.observer.subprocess.run", return_value=mock_result):
            result = check_session_ready()

        assert result is not None
        assert "DISPLAY" in result
