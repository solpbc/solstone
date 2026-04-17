# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for sol.py unified CLI."""

import sys
from unittest.mock import MagicMock, patch

import pytest

import sol


class TestResolveCommand:
    """Tests for resolve_command() function."""

    def test_resolve_known_command(self):
        """Test resolving a known command from registry."""
        module_path, preset_args = sol.resolve_command("import")
        assert module_path == "think.importers.cli"
        assert preset_args == []

    def test_resolve_direct_module_path(self):
        """Test resolving a direct module path with dot."""
        module_path, preset_args = sol.resolve_command("think.importers.cli")
        assert module_path == "think.importers.cli"
        assert preset_args == []

    def test_resolve_nested_module_path(self):
        """Test resolving a deeply nested module path."""
        module_path, preset_args = sol.resolve_command("observe.linux.observer")
        assert module_path == "observe.linux.observer"
        assert preset_args == []

    def test_resolve_unknown_command_raises(self):
        """Test that unknown command raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            sol.resolve_command("nonexistent")
        assert "Unknown command: nonexistent" in str(exc_info.value)

    def test_resolve_alias_with_preset_args(self):
        """Test resolving an alias that includes preset arguments."""
        # Add a test alias
        sol.ALIASES["test-alias"] = ("think.indexer", ["--rescan"])
        try:
            module_path, preset_args = sol.resolve_command("test-alias")
            assert module_path == "think.indexer"
            assert preset_args == ["--rescan"]
        finally:
            del sol.ALIASES["test-alias"]

    def test_alias_takes_precedence_over_command(self):
        """Test that aliases override commands with same name."""
        # Add an alias that shadows a command
        sol.ALIASES["import"] = ("think.cluster", ["--force"])
        try:
            module_path, preset_args = sol.resolve_command("import")
            assert module_path == "think.cluster"
            assert preset_args == ["--force"]
        finally:
            del sol.ALIASES["import"]


class TestRunCommand:
    """Tests for run_command() function."""

    def test_run_command_success(self):
        """Test running a command that exits cleanly."""
        mock_module = MagicMock()
        mock_module.main = MagicMock()

        with patch("importlib.import_module", return_value=mock_module):
            exit_code = sol.run_command("test.module")
            assert exit_code == 0
            mock_module.main.assert_called_once()

    def test_run_command_with_system_exit(self):
        """Test running a command that calls sys.exit(0)."""
        mock_module = MagicMock()
        mock_module.main = MagicMock(side_effect=SystemExit(0))

        with patch("importlib.import_module", return_value=mock_module):
            exit_code = sol.run_command("test.module")
            assert exit_code == 0

    def test_run_command_with_nonzero_exit(self):
        """Test running a command that calls sys.exit(1)."""
        mock_module = MagicMock()
        mock_module.main = MagicMock(side_effect=SystemExit(1))

        with patch("importlib.import_module", return_value=mock_module):
            exit_code = sol.run_command("test.module")
            assert exit_code == 1

    def test_run_command_with_string_exit(self, capsys):
        """Test running a command that raises SystemExit with a string message."""
        mock_module = MagicMock()
        mock_module.main = MagicMock(side_effect=SystemExit("Error: something failed"))

        with patch("importlib.import_module", return_value=mock_module):
            exit_code = sol.run_command("test.module")
            assert exit_code == 1

        captured = capsys.readouterr()
        assert "Error: something failed" in captured.err

    def test_run_command_import_error(self):
        """Test handling ImportError for nonexistent module."""
        with patch(
            "importlib.import_module", side_effect=ImportError("No module named 'fake'")
        ):
            exit_code = sol.run_command("fake.module")
            assert exit_code == 1

    def test_run_command_no_main_function(self):
        """Test handling module without main() function."""
        mock_module = MagicMock(spec=[])  # No 'main' attribute

        with patch("importlib.import_module", return_value=mock_module):
            exit_code = sol.run_command("test.module")
            assert exit_code == 1


class TestGetStatus:
    """Tests for get_status() function."""

    def test_status_with_override(self, monkeypatch, tmp_path):
        """Test status when journal override is set and exists."""
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

        status = sol.get_status()
        assert status["journal_path"] == str(tmp_path)
        assert status["journal_source"] == "override"
        assert status["journal_exists"] is True

    def test_status_with_nonexistent_journal(self, monkeypatch, tmp_path):
        """Test status when override points to nonexistent dir."""
        nonexistent = tmp_path / "nonexistent"
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(nonexistent))

        status = sol.get_status()
        assert status["journal_path"] == str(nonexistent)
        assert status["journal_source"] == "override"
        assert status["journal_exists"] is False

    def test_status_without_override(self, monkeypatch):
        """Test status when no override is set uses project root."""
        monkeypatch.delenv("_SOLSTONE_JOURNAL_OVERRIDE", raising=False)
        status = sol.get_status()
        assert status["journal_path"].endswith("/journal")
        assert status["journal_source"] == "project"
        assert isinstance(status["journal_exists"], bool)


class TestMain:
    """Tests for main() function."""

    def test_main_no_args_shows_help(self, monkeypatch, capsys):
        """Test that running with no args shows help."""
        monkeypatch.setattr(sys, "argv", ["sol"])
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", "/tmp/test")

        sol.main()

        captured = capsys.readouterr()
        assert "sol - solstone unified CLI" in captured.out
        assert "Usage: sol <command>" in captured.out

    def test_main_help_flag(self, monkeypatch, capsys):
        """Test --help flag shows help."""
        monkeypatch.setattr(sys, "argv", ["sol", "--help"])
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", "/tmp/test")

        sol.main()

        captured = capsys.readouterr()
        assert "sol - solstone unified CLI" in captured.out

    def test_main_help_command_without_question(self, monkeypatch, capsys):
        """Test bare 'help' command shows static help."""
        monkeypatch.setattr(sys, "argv", ["sol", "help"])
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", "/tmp/test")

        sol.main()

        captured = capsys.readouterr()
        assert "sol - solstone unified CLI" in captured.out

    def test_main_version_flag(self, monkeypatch, capsys):
        """Test --version flag shows version."""
        monkeypatch.setattr(sys, "argv", ["sol", "--version"])

        sol.main()

        captured = capsys.readouterr()
        assert "sol (solstone)" in captured.out

    def test_main_path_flag(self, monkeypatch, capsys):
        """Test --path flag prints resolved journal path."""
        monkeypatch.setattr(sys, "argv", ["sol", "--path"])
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", "/tmp/test-journal")

        sol.main()

        captured = capsys.readouterr()
        assert captured.out.strip() == "/tmp/test-journal"

    def test_main_path_flag_default(self, monkeypatch, capsys):
        """Test --path prints project root journal when no override set."""
        monkeypatch.setattr(sys, "argv", ["sol", "--path"])
        monkeypatch.delenv("_SOLSTONE_JOURNAL_OVERRIDE", raising=False)
        sol.main()

        captured = capsys.readouterr()
        path = captured.out.strip()
        assert path != ""
        assert path.endswith("/journal")

    def test_main_root_command(self, monkeypatch, capsys):
        """Test 'root' command prints the project root directory."""
        monkeypatch.setattr(sys, "argv", ["sol", "root"])

        sol.main()

        captured = capsys.readouterr()
        path = captured.out.strip()
        assert path != ""
        # root should NOT end with /journal — that's --path
        assert not path.endswith("/journal")
        # should be a parent of the journal path
        assert (
            path.endswith("/solstone")
            or "/solstone" in path
            or path.endswith("/worktree")
        )

    def test_main_unknown_command_exits(self, monkeypatch):
        """Test that unknown command exits with code 1."""
        monkeypatch.setattr(sys, "argv", ["sol", "unknown-command"])

        with pytest.raises(SystemExit) as exc_info:
            sol.main()
        assert exc_info.value.code == 1

    def test_main_adjusts_sys_argv(self, monkeypatch):
        """Test that sys.argv is adjusted for subcommand."""
        monkeypatch.setattr(sys, "argv", ["sol", "import", "--day", "20250101"])

        captured_argv = []

        def mock_main():
            captured_argv.extend(sys.argv)

        mock_module = MagicMock()
        mock_module.main = mock_main

        with patch("importlib.import_module", return_value=mock_module):
            with pytest.raises(SystemExit):
                sol.main()

        assert captured_argv[0] == "sol import"
        assert "--day" in captured_argv
        assert "20250101" in captured_argv


class TestCommandRegistry:
    """Tests for command registry completeness."""

    def test_all_commands_have_modules(self):
        """Test that all registered commands point to valid module paths."""
        for cmd, module_path in sol.COMMANDS.items():
            assert "." in module_path, f"Command '{cmd}' has invalid module path"

    def test_groups_contain_valid_commands(self):
        """Test that all commands in groups exist in registry."""
        for group_name, commands in sol.GROUPS.items():
            for cmd in commands:
                assert cmd in sol.COMMANDS, (
                    f"Command '{cmd}' in group '{group_name}' not in registry"
                )

    def test_critical_commands_registered(self):
        """Test that critical commands are registered."""
        critical = ["import", "providers", "dream", "indexer", "transcribe"]
        for cmd in critical:
            assert cmd in sol.COMMANDS, f"Critical command '{cmd}' not registered"
