# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think/service.py - cross-platform service management."""

from __future__ import annotations

import plistlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from think import service


class TestPlatform:
    def test_darwin(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        assert service._platform() == "darwin"

    def test_linux(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        assert service._platform() == "linux"

    def test_unsupported(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "platform", "win32")
        with pytest.raises(SystemExit):
            service._platform()
        assert "unsupported platform" in capsys.readouterr().err


class TestPlistGeneration:
    def test_round_trip(self):
        env = {
            "HOME": "/Users/test",
            "PATH": "/usr/bin",
            "_SOLSTONE_JOURNAL_OVERRIDE": "/Users/test/journal",
        }
        data = service._generate_plist(env)
        plist = plistlib.loads(data)
        assert plist["Label"] == "org.solpbc.solstone"
        assert plist["ProgramArguments"][1] == "supervisor"
        assert plist["EnvironmentVariables"] == env
        assert plist["KeepAlive"] is True
        assert plist["RunAtLoad"] is True
        assert "launchd-stdout.log" in plist["StandardOutPath"]
        assert "launchd-stderr.log" in plist["StandardErrorPath"]


class TestSystemdUnit:
    def test_unit_content(self):
        env = {
            "HOME": "/home/test",
            "PATH": "/usr/bin",
            "_SOLSTONE_JOURNAL_OVERRIDE": "/home/test/journal",
        }
        unit = service._generate_systemd_unit(env)
        lines = unit.splitlines()

        # Section headers must start at column 0 (no leading whitespace)
        assert "[Unit]" == lines[0]
        assert any(line == "[Service]" for line in lines)
        assert any(line == "[Install]" for line in lines)

        assert "Type=simple" in unit
        assert "Restart=on-failure" in unit
        assert "ExecStart=" in unit
        assert "supervisor" in unit
        assert "Environment=HOME=/home/test" in unit
        assert "Environment=_SOLSTONE_JOURNAL_OVERRIDE=/home/test/journal" in unit
        assert "WantedBy=default.target" in unit


class TestEnvCollection:
    def test_captures_api_keys(self, monkeypatch, tmp_path):
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("REVAI_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("PLAUD_ACCESS_TOKEN", raising=False)

        env = service._collect_env()
        assert env["ANTHROPIC_API_KEY"] == "sk-test"
        assert env["OPENAI_API_KEY"] == "sk-openai"
        assert "GOOGLE_API_KEY" not in env

    def test_includes_venv_in_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
        for key in service._API_KEYS:
            monkeypatch.delenv(key, raising=False)

        env = service._collect_env()
        venv_bin = str(Path(sys.executable).parent)
        assert env["PATH"].startswith(venv_bin)

    def test_journal_path_is_absolute(self, monkeypatch, tmp_path):
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
        for key in service._API_KEYS:
            monkeypatch.delenv(key, raising=False)

        env = service._collect_env()
        assert Path(env["_SOLSTONE_JOURNAL_OVERRIDE"]).is_absolute()


class TestStatus:
    def test_not_installed_linux(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(
            service, "_unit_path", lambda: tmp_path / "nonexistent.service"
        )

        result = service._status()
        assert result == 1
        output = capsys.readouterr().out
        assert "not installed" in output

    def test_not_installed_darwin(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(
            service, "_plist_path", lambda: tmp_path / "nonexistent.plist"
        )

        result = service._status()
        assert result == 1
        output = capsys.readouterr().out
        assert "not installed" in output


class TestInstall:
    def test_linux_idempotent(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
        for key in service._API_KEYS:
            monkeypatch.delenv(key, raising=False)

        unit_path = tmp_path / "solstone.service"
        monkeypatch.setattr(service, "_unit_path", lambda: unit_path)

        with patch("think.service.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = service._install()
            assert result == 0
            assert unit_path.exists()

            result = service._install()
            assert result == 0
            assert unit_path.exists()

        assert "Wrote" in capsys.readouterr().out


class TestLingerCheck:
    def test_warns_when_linger_disabled(self, capsys):
        mock_result = MagicMock(returncode=0, stdout="Linger=no\n")
        with patch("think.service.subprocess.run", return_value=mock_result):
            service._check_linger()
        output = capsys.readouterr().out
        assert "linger is not enabled" in output.lower()

    def test_silent_when_linger_enabled(self, capsys):
        mock_result = MagicMock(returncode=0, stdout="Linger=yes\n")
        with patch("think.service.subprocess.run", return_value=mock_result):
            service._check_linger()
        output = capsys.readouterr().out
        assert "linger" not in output.lower()

    def test_silent_when_loginctl_missing(self, capsys):
        with patch("think.service.subprocess.run", side_effect=FileNotFoundError):
            service._check_linger()
        output = capsys.readouterr().out
        assert output == ""


class TestRegistry:
    def test_service_command_registered(self):
        import sol

        assert "service" in sol.COMMANDS
        assert sol.COMMANDS["service"] == "think.service"

    def test_up_alias(self):
        import sol

        assert "up" in sol.ALIASES
        assert sol.ALIASES["up"] == ("think.service", ["up"])

    def test_down_alias(self):
        import sol

        assert "down" in sol.ALIASES
        assert sol.ALIASES["down"] == ("think.service", ["down"])

    def test_service_group_exists(self):
        import sol

        assert "Service" in sol.GROUPS
        assert "service" in sol.GROUPS["Service"]


class TestMain:
    def test_no_args_shows_usage(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["sol service"])
        with pytest.raises(SystemExit):
            service.main()
        output = capsys.readouterr().out
        assert "Usage:" in output

    def test_unknown_subcommand(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["sol service", "bogus"])
        with pytest.raises(SystemExit):
            service.main()
        assert "Unknown subcommand" in capsys.readouterr().err
