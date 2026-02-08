# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for todos CLI commands (sol call todos ...)."""

from typer.testing import CliRunner

from think.call import call_app

runner = CliRunner()


class TestTodosList:
    """Tests for 'sol call todos list' command."""

    def test_list_with_facet(self, todo_env):
        """List todos for a single day with --facet."""
        todo_env(
            [{"text": "Buy milk"}, {"text": "Walk dog", "completed": True}],
            day="20240101",
        )
        result = runner.invoke(
            call_app, ["todos", "list", "20240101", "--facet", "personal"]
        )
        assert result.exit_code == 0
        assert "Buy milk" in result.output
        assert "Walk dog" in result.output

    def test_list_all_facets(self, todo_env):
        """List todos across all facets when --facet is omitted."""
        todo_env([{"text": "Work task"}], day="20240101", facet="work")
        # Add a second facet's todos in the same journal
        todo_env([{"text": "Home task"}], day="20240101", facet="home")
        result = runner.invoke(call_app, ["todos", "list", "20240101"])
        assert result.exit_code == 0
        assert "Work task" in result.output
        assert "Home task" in result.output

    def test_list_empty_day(self, todo_env):
        """Empty day shows no-todos message."""
        todo_env([], day="20240101")
        result = runner.invoke(
            call_app, ["todos", "list", "20240101", "--facet", "personal"]
        )
        assert result.exit_code == 0
        assert "No todos" in result.output

    def test_list_invalid_range(self, todo_env):
        """--to before day produces an error."""
        todo_env([], day="20240101")
        result = runner.invoke(
            call_app,
            ["todos", "list", "20240201", "--facet", "personal", "--to", "20240101"],
        )
        assert result.exit_code == 1
        assert "Error" in result.output
