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


class TestTodosAdd:
    """Tests for 'sol call todos add' command."""

    def test_add_todo(self, todo_env):
        """Add a todo to a future day."""
        todo_env([], day="29991231")
        result = runner.invoke(
            call_app,
            [
                "todos",
                "add",
                "Ship feature",
                "--day",
                "29991231",
                "--facet",
                "personal",
            ],
        )
        assert result.exit_code == 0
        assert "Ship feature" in result.output

    def test_add_appends_to_existing(self, todo_env):
        """Add appends after existing items."""
        todo_env([{"text": "First"}], day="29991231")
        result = runner.invoke(
            call_app,
            ["todos", "add", "Second", "--day", "29991231", "--facet", "personal"],
        )
        assert result.exit_code == 0
        assert "First" in result.output
        assert "Second" in result.output

    def test_add_past_date_rejected(self, todo_env):
        """Adding to a past date fails."""
        todo_env([], day="20200101")
        result = runner.invoke(
            call_app,
            ["todos", "add", "Nope", "--day", "20200101", "--facet", "personal"],
        )
        assert result.exit_code == 1

    def test_add_empty_text_rejected(self, todo_env):
        """Adding empty text fails."""
        todo_env([], day="29991231")
        result = runner.invoke(
            call_app,
            ["todos", "add", "   ", "--day", "29991231", "--facet", "personal"],
        )
        assert result.exit_code == 1


class TestTodosDone:
    """Tests for 'sol call todos done' command."""

    def test_done_marks_complete(self, todo_env):
        """Mark a todo as done."""
        todo_env([{"text": "Buy milk"}], day="20240101")
        result = runner.invoke(
            call_app, ["todos", "done", "1", "--day", "20240101", "--facet", "personal"]
        )
        assert result.exit_code == 0
        assert "[x]" in result.output

    def test_done_invalid_line_number(self, todo_env):
        """Invalid line number fails."""
        todo_env([{"text": "Only one"}], day="20240101")
        result = runner.invoke(
            call_app, ["todos", "done", "5", "--day", "20240101", "--facet", "personal"]
        )
        assert result.exit_code == 1


class TestTodosCancel:
    """Tests for 'sol call todos cancel' command."""

    def test_cancel_entry(self, todo_env):
        """Cancel a todo."""
        todo_env([{"text": "Buy milk"}], day="20240101")
        result = runner.invoke(
            call_app,
            ["todos", "cancel", "1", "--day", "20240101", "--facet", "personal"],
        )
        assert result.exit_code == 0
        assert "cancelled" in result.output

    def test_cancel_invalid_line_number(self, todo_env):
        """Invalid line number fails."""
        todo_env([{"text": "Only one"}], day="20240101")
        result = runner.invoke(
            call_app,
            ["todos", "cancel", "5", "--day", "20240101", "--facet", "personal"],
        )
        assert result.exit_code == 1


class TestTodosUpcoming:
    """Tests for 'sol call todos upcoming' command."""

    def test_upcoming_shows_future(self, todo_env):
        """Upcoming shows future todos."""
        todo_env([{"text": "Future task"}], day="29991231")
        result = runner.invoke(call_app, ["todos", "upcoming"])
        assert result.exit_code == 0
        assert "Future task" in result.output

    def test_upcoming_with_facet_filter(self, todo_env):
        """Upcoming filters by facet."""
        todo_env([{"text": "Work task"}], day="29991231", facet="work")
        result = runner.invoke(call_app, ["todos", "upcoming", "--facet", "work"])
        assert result.exit_code == 0
        assert "Work task" in result.output

    def test_upcoming_no_future_todos(self, todo_env):
        """No future todos shows appropriate message."""
        todo_env([], day="20200101")
        result = runner.invoke(call_app, ["todos", "upcoming"])
        assert result.exit_code == 0
        assert "No upcoming todos" in result.output


class TestSolEnvResolution:
    """Tests for SOL_* env var resolution in todos commands."""

    def test_list_from_sol_day(self, todo_env, monkeypatch):
        """list with SOL_DAY env and no day arg works."""
        todo_env([{"text": "Env task"}], day="20240101")
        monkeypatch.setenv("SOL_DAY", "20240101")
        result = runner.invoke(call_app, ["todos", "list", "--facet", "personal"])
        assert result.exit_code == 0
        assert "Env task" in result.output

    def test_add_from_sol_day_and_facet(self, todo_env, monkeypatch):
        """add with SOL_DAY + SOL_FACET env works."""
        todo_env([], day="29991231")
        monkeypatch.setenv("SOL_DAY", "29991231")
        monkeypatch.setenv("SOL_FACET", "personal")
        result = runner.invoke(call_app, ["todos", "add", "Env todo"])
        assert result.exit_code == 0
        assert "Env todo" in result.output

    def test_done_from_sol_day_and_facet(self, todo_env, monkeypatch):
        """done with SOL_DAY + SOL_FACET env works."""
        todo_env([{"text": "Buy milk"}], day="20240101")
        monkeypatch.setenv("SOL_DAY", "20240101")
        monkeypatch.setenv("SOL_FACET", "personal")
        result = runner.invoke(call_app, ["todos", "done", "1"])
        assert result.exit_code == 0
        assert "[x]" in result.output
