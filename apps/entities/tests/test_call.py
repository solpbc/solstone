# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for entities CLI commands (sol call entities ...)."""

from typer.testing import CliRunner

from think.call import call_app

runner = CliRunner()


class TestEntitiesList:
    def test_list_attached(self, entity_env):
        entity_env(
            attached=[
                {
                    "type": "Person",
                    "name": "Alice Johnson",
                    "description": "Friend",
                    "attached_at": 1000,
                    "updated_at": 1000,
                },
                {
                    "type": "Company",
                    "name": "Acme Corp",
                    "description": "Client",
                    "attached_at": 1001,
                    "updated_at": 1001,
                },
            ]
        )

        result = runner.invoke(call_app, ["entities", "list", "personal"])

        assert result.exit_code == 0
        assert "2 attached entities" in result.output
        assert "Alice Johnson" in result.output
        assert "Acme Corp" in result.output

    def test_list_detected(self, entity_env):
        entity_env(
            detected=[
                {
                    "type": "Person",
                    "name": "Alice",
                    "description": "Met at conference",
                },
                {
                    "type": "Tool",
                    "name": "pytest",
                    "description": "Testing tool",
                },
            ],
            day="20240101",
        )

        result = runner.invoke(
            call_app, ["entities", "list", "personal", "--day", "20240101"]
        )

        assert result.exit_code == 0
        assert "Alice" in result.output
        assert "pytest" in result.output

    def test_list_empty(self, entity_env):
        entity_env()

        result = runner.invoke(call_app, ["entities", "list", "personal"])

        assert result.exit_code == 0
        assert "No entities found" in result.output


class TestEntitiesDetect:
    def test_detect_new(self, entity_env):
        entity_env()

        result = runner.invoke(
            call_app,
            [
                "entities",
                "detect",
                "20240101",
                "personal",
                "Person",
                "Alice",
                "Met at conference",
            ],
        )

        assert result.exit_code == 0
        assert "detected" in result.output

    def test_detect_duplicate(self, entity_env):
        entity_env(
            detected=[
                {"type": "Person", "name": "Alice", "description": "First"},
            ],
            day="20240101",
        )

        result = runner.invoke(
            call_app,
            [
                "entities",
                "detect",
                "20240101",
                "personal",
                "Person",
                "Alice",
                "Second",
            ],
        )

        assert result.exit_code == 1
        assert "already detected" in result.output

    def test_detect_invalid_type(self, entity_env):
        entity_env()

        result = runner.invoke(
            call_app,
            [
                "entities",
                "detect",
                "20240101",
                "personal",
                "AB",
                "Alice",
                "Met at conference",
            ],
        )

        assert result.exit_code == 1
        assert "Invalid" in result.output


class TestEntitiesAttach:
    def test_attach_new(self, entity_env):
        entity_env()

        result = runner.invoke(
            call_app,
            ["entities", "attach", "personal", "Person", "Alice Johnson", "Friend"],
        )

        assert result.exit_code == 0
        assert "attached" in result.output

    def test_attach_existing(self, entity_env):
        entity_env(
            attached=[
                {
                    "type": "Person",
                    "name": "Alice Johnson",
                    "description": "Friend",
                    "attached_at": 1000,
                    "updated_at": 1000,
                }
            ]
        )

        result = runner.invoke(
            call_app,
            ["entities", "attach", "personal", "Person", "Alice Johnson", "Friend"],
        )

        assert result.exit_code == 0
        assert "already attached" in result.output

    def test_attach_invalid_type(self, entity_env):
        entity_env()

        result = runner.invoke(
            call_app,
            ["entities", "attach", "personal", "AB", "Alice Johnson", "Friend"],
        )

        assert result.exit_code == 1
        assert "Invalid" in result.output


class TestEntitiesUpdate:
    def test_update_attached(self, entity_env):
        entity_env(
            attached=[
                {
                    "type": "Person",
                    "name": "Alice Johnson",
                    "description": "Old",
                    "attached_at": 1000,
                    "updated_at": 1000,
                }
            ]
        )

        result = runner.invoke(
            call_app,
            ["entities", "update", "personal", "Alice Johnson", "New description"],
        )
        verify = runner.invoke(call_app, ["entities", "list", "personal"])

        assert result.exit_code == 0
        assert "updated" in result.output
        assert "New description" in verify.output

    def test_update_detected(self, entity_env):
        entity_env(
            detected=[
                {"type": "Person", "name": "Alice", "description": "Old"},
            ],
            day="20240101",
        )

        result = runner.invoke(
            call_app,
            [
                "entities",
                "update",
                "personal",
                "Alice",
                "New desc",
                "--day",
                "20240101",
            ],
        )

        assert result.exit_code == 0
        assert "updated" in result.output

    def test_update_not_found(self, entity_env):
        entity_env()

        result = runner.invoke(
            call_app,
            ["entities", "update", "personal", "Missing", "New description"],
        )

        assert result.exit_code == 1
        assert "not found" in result.output


class TestEntitiesAka:
    def test_add_aka(self, entity_env):
        entity_env(
            attached=[
                {
                    "type": "Person",
                    "name": "Alice Johnson",
                    "description": "Friend",
                    "attached_at": 1000,
                    "updated_at": 1000,
                }
            ]
        )

        result = runner.invoke(
            call_app,
            ["entities", "aka", "personal", "Alice Johnson", "Ali"],
        )

        assert result.exit_code == 0
        assert "Added alias" in result.output

    def test_aka_duplicate(self, entity_env):
        entity_env(
            attached=[
                {
                    "type": "Person",
                    "name": "Alice Johnson",
                    "description": "Friend",
                    "attached_at": 1000,
                    "updated_at": 1000,
                    "aka": ["Ali"],
                }
            ]
        )

        result = runner.invoke(
            call_app,
            ["entities", "aka", "personal", "Alice Johnson", "Ali"],
        )

        assert result.exit_code == 0
        assert "already exists" in result.output

    def test_aka_first_word(self, entity_env):
        entity_env(
            attached=[
                {
                    "type": "Person",
                    "name": "Alice Johnson",
                    "description": "Friend",
                    "attached_at": 1000,
                    "updated_at": 1000,
                }
            ]
        )

        result = runner.invoke(
            call_app,
            ["entities", "aka", "personal", "Alice Johnson", "Alice"],
        )

        assert result.exit_code == 0
        assert "first word" in result.output


class TestEntitiesObservations:
    def test_observations_empty(self, entity_env):
        entity_env(
            attached=[
                {
                    "type": "Person",
                    "name": "Alice Johnson",
                    "description": "Friend",
                    "attached_at": 1000,
                    "updated_at": 1000,
                }
            ]
        )

        result = runner.invoke(
            call_app,
            ["entities", "observations", "personal", "Alice Johnson"],
        )

        assert result.exit_code == 0
        assert "No observations" in result.output

    def test_observations_with_data(self, entity_env):
        entity_env(
            attached=[
                {
                    "type": "Person",
                    "name": "Alice Johnson",
                    "description": "Friend",
                    "attached_at": 1000,
                    "updated_at": 1000,
                }
            ],
            observations=["Likes coffee", "Expert in Python"],
            observation_entity="Alice Johnson",
        )

        result = runner.invoke(
            call_app,
            ["entities", "observations", "personal", "Alice Johnson"],
        )

        assert result.exit_code == 0
        assert "Likes coffee" in result.output
        assert "Expert in Python" in result.output


class TestEntitiesObserve:
    def test_observe_new(self, entity_env):
        entity_env(
            attached=[
                {
                    "type": "Person",
                    "name": "Alice Johnson",
                    "description": "Friend",
                    "attached_at": 1000,
                    "updated_at": 1000,
                }
            ]
        )

        result = runner.invoke(
            call_app,
            ["entities", "observe", "personal", "Alice Johnson", "Likes coffee"],
        )

        assert result.exit_code == 0
        assert "Observation added" in result.output

    def test_observe_not_found(self, entity_env):
        entity_env()

        result = runner.invoke(
            call_app,
            ["entities", "observe", "personal", "Missing", "Likes coffee"],
        )

        assert result.exit_code == 1
        assert "not found" in result.output
