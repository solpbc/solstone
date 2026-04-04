# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for sol call sol — identity directory read/write commands."""

import json

import pytest
from typer.testing import CliRunner

from think.tools.sol import app

runner = CliRunner()


@pytest.fixture
def journal_with_sol(tmp_path, monkeypatch):
    """Set up a journal with sol/ directory containing self.md and agency.md."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    # Provide minimal config for ensure_sol_directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "journal.json").write_text(
        json.dumps({"identity": {"name": "Test User"}})
    )

    sol_dir = tmp_path / "sol"
    sol_dir.mkdir()

    self_md = """\
# self

I am sol. this is a new journal — we're just getting started.

## my name
sol (default)

## who I'm here for
Test User

## our relationship
[forming]

## what I've noticed
[observing]

## what I find interesting
[discovering]
"""
    (sol_dir / "self.md").write_text(self_md)

    agency_md = """\
# agency

things I'm tracking, acting on, or watching.

## curation
[nothing yet]

## observations
[watching and learning]

## system
[monitoring]
"""
    (sol_dir / "agency.md").write_text(agency_md)

    return tmp_path


class TestSolSelfRead:
    def test_read_self(self, journal_with_sol):
        result = runner.invoke(app, ["self"])
        assert result.exit_code == 0
        assert "# self" in result.output
        assert "Test User" in result.output

    def test_read_self_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "journal.json").write_text(json.dumps({}))
        # ensure_sol_directory will create the file, so this tests the happy path
        result = runner.invoke(app, ["self"])
        assert result.exit_code == 0


class TestSolSelfWrite:
    def test_write_self(self, journal_with_sol):
        new_content = "# self\n\nI am sol. Jer's journal.\n\n## my name\nsol\n"
        result = runner.invoke(app, ["self", "--write"], input=new_content)
        assert result.exit_code == 0
        assert "self.md updated" in result.output

        # Verify file was written
        self_path = journal_with_sol / "sol" / "self.md"
        assert self_path.read_text() == new_content

    def test_write_self_empty_stdin(self, journal_with_sol):
        result = runner.invoke(app, ["self", "--write"], input="")
        assert result.exit_code == 1
        assert "no content" in result.output

    def test_write_self_whitespace_only(self, journal_with_sol):
        result = runner.invoke(app, ["self", "--write"], input="   \n\n  ")
        assert result.exit_code == 1
        assert "no content" in result.output


class TestSolSelfUpdateSection:
    def test_update_section_owner(self, journal_with_sol):
        result = runner.invoke(
            app,
            ["self", "--update-section", "who I'm here for"],
            input="Jer — goes by Jer, not Jeremie",
        )
        assert result.exit_code == 0
        assert "Updated ## who I'm here for" in result.output

        # Verify section was updated, other sections preserved
        self_path = journal_with_sol / "sol" / "self.md"
        content = self_path.read_text()
        assert "Jer — goes by Jer, not Jeremie" in content
        assert "## my name" in content
        assert "sol (default)" in content
        assert "## our relationship" in content

    def test_update_section_not_found(self, journal_with_sol):
        result = runner.invoke(
            app,
            ["self", "--update-section", "nonexistent"],
            input="content",
        )
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_update_section_empty_stdin(self, journal_with_sol):
        result = runner.invoke(
            app,
            ["self", "--update-section", "who I'm here for"],
            input="",
        )
        assert result.exit_code == 1
        assert "no content" in result.output


class TestSolAgencyRead:
    def test_read_agency(self, journal_with_sol):
        result = runner.invoke(app, ["agency"])
        assert result.exit_code == 0
        assert "# agency" in result.output
        assert "## curation" in result.output

    def test_read_agency_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "journal.json").write_text(json.dumps({}))
        # ensure_sol_directory creates agency.md
        result = runner.invoke(app, ["agency"])
        assert result.exit_code == 0


class TestSolAgencyWrite:
    def test_write_agency(self, journal_with_sol):
        new_content = "# agency\n\n## curation\n- review entity duplicates\n\n## system\n[clean]\n"
        result = runner.invoke(app, ["agency", "--write"], input=new_content)
        assert result.exit_code == 0
        assert "agency.md updated" in result.output

        # Verify file was written
        agency_path = journal_with_sol / "sol" / "agency.md"
        assert agency_path.read_text() == new_content

    def test_write_agency_empty_stdin(self, journal_with_sol):
        result = runner.invoke(app, ["agency", "--write"], input="")
        assert result.exit_code == 1
        assert "no content" in result.output


class TestSolPulseRead:
    def test_read_pulse(self, journal_with_sol):
        pulse_md = "---\nupdated: 2026-03-22T14:00:00\nsource: pulse-cogitate\n---\n\nTest narrative.\n"
        (journal_with_sol / "sol" / "pulse.md").write_text(pulse_md)
        result = runner.invoke(app, ["pulse"])
        assert result.exit_code == 0
        assert "Test narrative" in result.output

    def test_read_pulse_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "journal.json").write_text(json.dumps({}))
        result = runner.invoke(app, ["pulse"])
        assert result.exit_code == 1
        assert "not found" in result.output


class TestSolPulseWrite:
    def test_write_pulse(self, journal_with_sol):
        new_content = "---\nupdated: 2026-03-22T14:00:00\nsource: pulse-cogitate\n---\n\nNew narrative.\n"
        result = runner.invoke(app, ["pulse", "--write"], input=new_content)
        assert result.exit_code == 0
        assert "pulse.md updated" in result.output

        # Verify file was written
        pulse_path = journal_with_sol / "sol" / "pulse.md"
        assert pulse_path.read_text() == new_content

    def test_write_pulse_empty_stdin(self, journal_with_sol):
        result = runner.invoke(app, ["pulse", "--write"], input="")
        assert result.exit_code == 1
        assert "no content" in result.output


class TestSolWriteDoesNotEscapeSolDir:
    """Verify that sol call sol only writes to sol/ directory files."""

    def test_self_write_stays_in_sol_dir(self, journal_with_sol):
        """Write to self.md goes to sol/self.md, not anywhere else."""
        result = runner.invoke(app, ["self", "--write"], input="test content\n")
        assert result.exit_code == 0
        self_path = journal_with_sol / "sol" / "self.md"
        assert self_path.read_text() == "test content\n"
        # No files created outside sol/
        journal_files = set(f.name for f in journal_with_sol.iterdir() if f.is_file())
        assert "self.md" not in journal_files

    def test_agency_write_stays_in_sol_dir(self, journal_with_sol):
        """Write to agency.md goes to sol/agency.md, not anywhere else."""
        result = runner.invoke(app, ["agency", "--write"], input="test content\n")
        assert result.exit_code == 0
        agency_path = journal_with_sol / "sol" / "agency.md"
        assert agency_path.read_text() == "test content\n"
        # No files created outside sol/
        journal_files = set(f.name for f in journal_with_sol.iterdir() if f.is_file())
        assert "agency.md" not in journal_files

    def test_pulse_write_stays_in_sol_dir(self, journal_with_sol):
        """Write to pulse.md goes to sol/pulse.md, not anywhere else."""
        result = runner.invoke(app, ["pulse", "--write"], input="test content\n")
        assert result.exit_code == 0
        pulse_path = journal_with_sol / "sol" / "pulse.md"
        assert pulse_path.read_text() == "test content\n"
        # No files created outside sol/
        journal_files = set(f.name for f in journal_with_sol.iterdir() if f.is_file())
        assert "pulse.md" not in journal_files


class TestSolSelfValueOption:
    def test_write_self_with_value(self, journal_with_sol):
        new_content = "# self\n\nI am sol. Jer's journal.\n\n## my name\nsol\n"
        result = runner.invoke(app, ["self", "--write", "--value", new_content])
        assert result.exit_code == 0
        assert "self.md updated" in result.output
        self_path = journal_with_sol / "sol" / "self.md"
        assert self_path.read_text() == new_content

    def test_update_section_with_value(self, journal_with_sol):
        result = runner.invoke(
            app,
            [
                "self",
                "--update-section",
                "who I'm here for",
                "--value",
                "Jer — founder",
            ],
        )
        assert result.exit_code == 0
        assert "Updated ## who I'm here for" in result.output
        content = (journal_with_sol / "sol" / "self.md").read_text()
        assert "Jer — founder" in content

    def test_value_empty_string_errors(self, journal_with_sol):
        result = runner.invoke(app, ["self", "--write", "--value", "   "])
        assert result.exit_code == 1
        assert "no content" in result.output

    def test_value_takes_precedence_over_stdin(self, journal_with_sol):
        result = runner.invoke(
            app,
            ["self", "--write", "--value", "from value\n"],
            input="from stdin\n",
        )
        assert result.exit_code == 0
        self_path = journal_with_sol / "sol" / "self.md"
        assert self_path.read_text() == "from value\n"


class TestSolAgencyValueOption:
    def test_write_agency_with_value(self, journal_with_sol):
        new_content = "# agency\n\n## curation\n- item\n"
        result = runner.invoke(app, ["agency", "--write", "--value", new_content])
        assert result.exit_code == 0
        assert "agency.md updated" in result.output
        agency_path = journal_with_sol / "sol" / "agency.md"
        assert agency_path.read_text() == new_content

    def test_value_empty_string_errors(self, journal_with_sol):
        result = runner.invoke(app, ["agency", "--write", "--value", ""])
        assert result.exit_code == 1
        assert "no content" in result.output


class TestSolPulseValueOption:
    def test_write_pulse_with_value(self, journal_with_sol):
        new_content = "---\nupdated: 2026-03-22\n---\n\nNarrative.\n"
        result = runner.invoke(app, ["pulse", "--write", "--value", new_content])
        assert result.exit_code == 0
        assert "pulse.md updated" in result.output
        pulse_path = journal_with_sol / "sol" / "pulse.md"
        assert pulse_path.read_text() == new_content

    def test_value_empty_string_errors(self, journal_with_sol):
        result = runner.invoke(app, ["pulse", "--write", "--value", ""])
        assert result.exit_code == 1
        assert "no content" in result.output


class TestSolHistoryLogging:
    def test_self_write_logs_history(self, journal_with_sol):
        new_content = "# self\n\nUpdated.\n"
        runner.invoke(app, ["self", "--write", "--value", new_content])
        history = journal_with_sol / "sol" / "history.jsonl"
        assert history.exists()
        records = [json.loads(line) for line in history.read_text().strip().split("\n")]
        assert len(records) == 1
        assert records[0]["file"] == "self.md"
        assert records[0]["source"] == "cli"
        assert records[0]["section"] is None
        assert "ts" in records[0]
        assert "diff" in records[0]

    def test_agency_write_logs_history(self, journal_with_sol):
        runner.invoke(app, ["agency", "--write", "--value", "# agency\n\nNew.\n"])
        history = journal_with_sol / "sol" / "history.jsonl"
        assert history.exists()
        records = [json.loads(line) for line in history.read_text().strip().split("\n")]
        assert len(records) == 1
        assert records[0]["file"] == "agency.md"
        assert records[0]["source"] == "cli"

    def test_pulse_write_logs_history(self, journal_with_sol):
        runner.invoke(app, ["pulse", "--write", "--value", "---\n---\n\nPulse.\n"])
        history = journal_with_sol / "sol" / "history.jsonl"
        assert history.exists()
        records = [json.loads(line) for line in history.read_text().strip().split("\n")]
        assert len(records) == 1
        assert records[0]["file"] == "pulse.md"

    def test_update_section_logs_history(self, journal_with_sol):
        runner.invoke(
            app,
            ["self", "--update-section", "who I'm here for", "--value", "Jer"],
        )
        history = journal_with_sol / "sol" / "history.jsonl"
        assert history.exists()
        records = [json.loads(line) for line in history.read_text().strip().split("\n")]
        assert len(records) == 1
        assert records[0]["file"] == "self.md"
        assert records[0]["section"] == "who I'm here for"
        assert records[0]["source"] == "api"

    def test_multiple_writes_append(self, journal_with_sol):
        runner.invoke(app, ["self", "--write", "--value", "# self\n\nFirst.\n"])
        runner.invoke(app, ["self", "--write", "--value", "# self\n\nSecond.\n"])
        history = journal_with_sol / "sol" / "history.jsonl"
        records = [json.loads(line) for line in history.read_text().strip().split("\n")]
        assert len(records) == 2


class TestHeartbeatEnsureSolDirectory:
    """Verify the heartbeat bug fix — ensure_sol_directory() takes no args."""

    def test_ensure_sol_directory_no_args(self):
        """ensure_sol_directory accepts no positional args (heartbeat.py:32 fix)."""
        import inspect

        from think.awareness import ensure_sol_directory

        sig = inspect.signature(ensure_sol_directory)
        params = [
            p for p in sig.parameters.values() if p.default is inspect.Parameter.empty
        ]
        assert len(params) == 0, (
            "ensure_sol_directory should take no required arguments"
        )

    def test_heartbeat_calls_correctly(self):
        """heartbeat.py calls ensure_sol_directory() without arguments."""
        import ast
        from pathlib import Path

        heartbeat_path = Path(__file__).parent.parent / "think" / "heartbeat.py"
        tree = ast.parse(heartbeat_path.read_text())

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ensure_sol_directory"
            ):
                assert len(node.args) == 0, (
                    f"ensure_sol_directory() called with {len(node.args)} args at line {node.lineno}"
                )
