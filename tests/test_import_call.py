# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest
from typer.testing import CliRunner

import convey.state
import think.utils
from think.call import call_app
from think.entities.journal import (
    clear_journal_entity_cache,
    load_journal_entity,
    save_journal_entity,
)
from think.entities.relationships import load_facet_relationship

journal_sources = import_module("apps.import.journal_sources")

create_state_directory = journal_sources.create_state_directory
generate_key = journal_sources.generate_key
get_state_directory = journal_sources.get_state_directory
save_journal_source = journal_sources.save_journal_source

runner = CliRunner()


def _source(name="test-source", key=None, **overrides):
    if key is None:
        key = generate_key()
    source = {
        "name": name,
        "key": key,
        "created_at": 1000,
        "enabled": True,
        "revoked": False,
        "revoked_at": None,
        "stats": {
            "segments_received": 0,
            "entities_received": 0,
            "facets_received": 0,
            "imports_received": 0,
            "config_received": 0,
        },
    }
    source.update(overrides)
    return source


@pytest.fixture
def import_env(tmp_path, monkeypatch):
    """Set up a temp journal with an import source and state directory."""

    monkeypatch.setattr(convey.state, "journal_root", str(tmp_path), raising=False)
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    think.utils._journal_path_cache = None
    clear_journal_entity_cache()
    (tmp_path / "apps" / "import" / "journal_sources").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)

    key = generate_key()
    source = _source(key=key)
    save_journal_source(source)
    key_prefix = key[:8]
    create_state_directory(tmp_path, key_prefix)

    return {
        "root": tmp_path,
        "key": key,
        "key_prefix": key_prefix,
        "source": source,
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_entity_state(key_prefix: str, state: dict) -> None:
    state_path = get_state_directory(key_prefix) / "entities" / "state.json"
    _write_json(state_path, state)


def test_list_staged_empty_state(import_env):
    result = runner.invoke(call_app, ["import", "list-staged", "--source", "test-source"])

    assert result.exit_code == 0
    assert result.stdout.strip() == ""


def test_list_staged_with_staged_entities(import_env):
    staged_path = (
        get_state_directory(import_env["key_prefix"])
        / "entities"
        / "staged"
        / "test-entity.json"
    )
    _write_json(
        staged_path,
        {
            "source_entity": {"id": "test-entity", "name": "Test Entity", "type": "Tool"},
            "match_candidates": [{"id": "target-id", "name": "Target Entity", "tier": 8}],
            "reason": "low_confidence_match",
            "staged_at": "2026-04-14T00:00:00+00:00",
        },
    )

    result = runner.invoke(
        call_app,
        ["import", "list-staged", "--source", "test-source", "--area", "entities"],
    )

    assert result.exit_code == 0
    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert lines == [
        {
            "area": "entities",
            "source_id": "test-entity",
            "reason": "low_confidence_match",
            "source_entity": {"id": "test-entity", "name": "Test Entity", "type": "Tool"},
            "match_candidates": [{"id": "target-id", "name": "Target Entity", "tier": 8}],
            "staged_at": "2026-04-14T00:00:00+00:00",
        }
    ]


def test_list_staged_with_config_diff(import_env):
    diff_path = get_state_directory(import_env["key_prefix"]) / "config" / "diff.json"
    _write_json(
        diff_path,
        {
            "identity.name": {
                "source": "Remote User",
                "target": "Local User",
                "category": "transferable",
            }
        },
    )

    result = runner.invoke(
        call_app,
        ["import", "list-staged", "--source", "test-source", "--area", "config"],
    )

    assert result.exit_code == 0
    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert lines == [
        {
            "area": "config",
            "diff": {
                "identity.name": {
                    "source": "Remote User",
                    "target": "Local User",
                    "category": "transferable",
                }
            },
        }
    ]


def test_list_staged_with_staged_facets(import_env):
    staged_file = "personal/entity_relationship/entities__source_entity__entity.json.staged.json"
    staged_path = get_state_directory(import_env["key_prefix"]) / "facets" / "staged" / staged_file
    _write_json(
        staged_path,
        {
            "reason": "unmapped_entity",
            "source_entity_id": "source_entity",
            "explanation": "Entity 'source_entity' has no mapping in entities/state.json id_map",
            "source_path": "entities/source_entity/entity.json",
            "source_data": json.dumps({"entity_id": "source_entity"}, ensure_ascii=False, indent=2)
            + "\n",
            "staged_at": "2026-04-14T00:00:00+00:00",
        },
    )

    result = runner.invoke(
        call_app,
        ["import", "list-staged", "--source", "test-source", "--area", "facets"],
    )

    assert result.exit_code == 0
    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert lines == [
        {
            "area": "facets",
            "staged_file": staged_file,
            "facet": "personal",
            "file_type": "entity_relationship",
            "reason": "unmapped_entity",
            "source_entity_id": "source_entity",
            "explanation": "Entity 'source_entity' has no mapping in entities/state.json id_map",
            "source_path": "entities/source_entity/entity.json",
            "source_data": json.dumps({"entity_id": "source_entity"}, ensure_ascii=False, indent=2)
            + "\n",
            "staged_at": "2026-04-14T00:00:00+00:00",
        }
    ]


def test_resolve_entity_merge(import_env):
    save_journal_entity(
        {
            "id": "target-id",
            "name": "Alice Johnson",
            "type": "Person",
            "aka": ["Ali"],
            "emails": ["alice@old.com"],
            "created_at": 2000,
        }
    )
    staged_path = (
        get_state_directory(import_env["key_prefix"])
        / "entities"
        / "staged"
        / "test-entity.json"
    )
    _write_json(
        staged_path,
        {
            "source_entity": {
                "id": "test-entity",
                "name": "Alice Johnson",
                "type": "Person",
                "aka": ["AJ"],
                "emails": ["alice@new.com"],
                "created_at": 1000,
            },
            "match_candidates": [{"id": "target-id", "name": "Alice Johnson", "tier": 8}],
            "reason": "low_confidence_match",
            "staged_at": "2026-04-14T00:00:00+00:00",
        },
    )

    result = runner.invoke(
        call_app,
        [
            "import",
            "resolve-entity",
            "test-entity",
            "merge",
            "--source",
            "test-source",
            "--target",
            "target-id",
        ],
    )

    assert result.exit_code == 0
    assert not staged_path.exists()
    merged = load_journal_entity("target-id")
    assert merged is not None
    assert merged["aka"] == ["AJ", "Ali"]
    assert merged["emails"] == ["alice@old.com", "alice@new.com"]
    assert merged["created_at"] == 1000

    state = _read_json(get_state_directory(import_env["key_prefix"]) / "entities" / "state.json")
    assert state["id_map"]["test-entity"] == "target-id"

    log_entries = _read_log(get_state_directory(import_env["key_prefix"]) / "entities" / "log.jsonl")
    assert log_entries[-1]["action"] == "resolved_merge"
    assert log_entries[-1]["resolved_by"] == "talent"


def test_resolve_entity_create(import_env):
    save_journal_entity({"id": "test-entity", "name": "Occupied Entity", "type": "Tool"})
    staged_path = (
        get_state_directory(import_env["key_prefix"])
        / "entities"
        / "staged"
        / "test-entity.json"
    )
    _write_json(
        staged_path,
        {
            "source_entity": {"id": "test-entity", "name": "Fresh Entity", "type": "Tool"},
            "match_candidates": [{"id": "test-entity", "name": "Occupied Entity", "tier": None}],
            "reason": "id_collision",
            "staged_at": "2026-04-14T00:00:00+00:00",
        },
    )

    result = runner.invoke(
        call_app,
        ["import", "resolve-entity", "test-entity", "create", "--source", "test-source"],
    )

    assert result.exit_code == 0
    assert not staged_path.exists()
    created = load_journal_entity("fresh_entity")
    assert created is not None
    assert created["name"] == "Fresh Entity"

    state = _read_json(get_state_directory(import_env["key_prefix"]) / "entities" / "state.json")
    assert state["id_map"]["test-entity"] == "fresh_entity"


def test_resolve_entity_create_principal_conflict(import_env):
    save_journal_entity(
        {
            "id": "existing-principal",
            "name": "Existing Principal",
            "type": "Person",
            "is_principal": True,
        }
    )
    staged_path = (
        get_state_directory(import_env["key_prefix"])
        / "entities"
        / "staged"
        / "new-principal.json"
    )
    _write_json(
        staged_path,
        {
            "source_entity": {
                "id": "new-principal",
                "name": "New Principal",
                "type": "Person",
                "is_principal": True,
            },
            "match_candidates": [],
            "reason": "principal_conflict",
            "staged_at": "2026-04-14T00:00:00+00:00",
        },
    )

    result = runner.invoke(
        call_app,
        ["import", "resolve-entity", "new-principal", "create", "--source", "test-source"],
    )

    assert result.exit_code == 0
    created = load_journal_entity("new-principal")
    assert created is not None
    assert created["is_principal"] is False


def test_resolve_entity_skip(import_env):
    staged_path = (
        get_state_directory(import_env["key_prefix"])
        / "entities"
        / "staged"
        / "test-entity.json"
    )
    _write_json(
        staged_path,
        {
            "source_entity": {"id": "test-entity", "name": "Skip Entity", "type": "Tool"},
            "match_candidates": [],
            "reason": "principal_conflict",
            "staged_at": "2026-04-14T00:00:00+00:00",
        },
    )

    result = runner.invoke(
        call_app,
        ["import", "resolve-entity", "test-entity", "skip", "--source", "test-source"],
    )

    assert result.exit_code == 0
    assert not staged_path.exists()
    assert load_journal_entity("test-entity") is None

    log_entries = _read_log(get_state_directory(import_env["key_prefix"]) / "entities" / "log.jsonl")
    assert log_entries[-1]["action"] == "resolved_skip"
    assert log_entries[-1]["resolved_by"] == "talent"


def test_resolve_config_apply(import_env):
    diff_path = get_state_directory(import_env["key_prefix"]) / "config" / "diff.json"
    _write_json(
        diff_path,
        {
            "identity.name": {
                "source": "Remote User",
                "target": "Local User",
                "category": "transferable",
            }
        },
    )
    _write_json(
        get_state_directory(import_env["key_prefix"]) / "config" / "source_config.json",
        {"identity": {"name": "Remote User"}},
    )
    _write_json(import_env["root"] / "config" / "journal.json", {"identity": {"name": "Local User"}})

    result = runner.invoke(
        call_app,
        ["import", "resolve-config", "identity.name", "apply", "--source", "test-source"],
    )

    assert result.exit_code == 0
    journal_config = _read_json(import_env["root"] / "config" / "journal.json")
    assert journal_config["identity"]["name"] == "Remote User"
    assert not diff_path.exists()

    log_entries = _read_log(get_state_directory(import_env["key_prefix"]) / "config" / "log.jsonl")
    assert log_entries[-1]["action"] == "config_field_applied"
    assert log_entries[-1]["resolved_by"] == "talent"


def test_resolve_config_keep(import_env):
    diff_path = get_state_directory(import_env["key_prefix"]) / "config" / "diff.json"
    _write_json(
        diff_path,
        {
            "retention.days": {
                "source": 30,
                "target": 90,
                "category": "preference",
            }
        },
    )
    _write_json(
        get_state_directory(import_env["key_prefix"]) / "config" / "source_config.json",
        {"retention": {"days": 30}},
    )
    _write_json(import_env["root"] / "config" / "journal.json", {"retention": {"days": 90}})

    result = runner.invoke(
        call_app,
        ["import", "resolve-config", "retention.days", "keep", "--source", "test-source"],
    )

    assert result.exit_code == 0
    journal_config = _read_json(import_env["root"] / "config" / "journal.json")
    assert journal_config["retention"]["days"] == 90
    assert not diff_path.exists()


def test_resolve_config_all_transferable(import_env):
    diff_path = get_state_directory(import_env["key_prefix"]) / "config" / "diff.json"
    _write_json(
        diff_path,
        {
            "identity.name": {
                "source": "Remote User",
                "target": "Local User",
                "category": "transferable",
            },
            "retention.days": {
                "source": 30,
                "target": 90,
                "category": "preference",
            },
        },
    )
    _write_json(
        get_state_directory(import_env["key_prefix"]) / "config" / "source_config.json",
        {
            "identity": {"name": "Remote User"},
            "retention": {"days": 30},
        },
    )
    _write_json(
        import_env["root"] / "config" / "journal.json",
        {"identity": {"name": "Local User"}, "retention": {"days": 90}},
    )

    result = runner.invoke(
        call_app,
        [
            "import",
            "resolve-config-all",
            "--source",
            "test-source",
            "--category",
            "transferable",
        ],
    )

    assert result.exit_code == 0
    journal_config = _read_json(import_env["root"] / "config" / "journal.json")
    assert journal_config["identity"]["name"] == "Remote User"
    assert journal_config["retention"]["days"] == 90

    remaining_diff = _read_json(diff_path)
    assert list(remaining_diff) == ["retention.days"]


def test_resolve_facet_apply_unmapped_entity(import_env):
    _write_entity_state(
        import_env["key_prefix"],
        {"id_map": {"source_entity": "target_entity"}, "received": {}},
    )
    staged_file = "personal/entity_relationship/entities__source_entity__entity.json.staged.json"
    staged_path = get_state_directory(import_env["key_prefix"]) / "facets" / "staged" / staged_file
    _write_json(
        staged_path,
        {
            "reason": "unmapped_entity",
            "source_entity_id": "source_entity",
            "explanation": "Entity 'source_entity' has no mapping in entities/state.json id_map",
            "source_path": "entities/source_entity/entity.json",
            "source_data": json.dumps(
                {"entity_id": "source_entity", "description": "imported relationship"},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            "staged_at": "2026-04-14T00:00:00+00:00",
        },
    )

    result = runner.invoke(
        call_app,
        ["import", "resolve-facet", staged_file, "apply", "--source", "test-source"],
    )

    assert result.exit_code == 0
    assert not staged_path.exists()
    relationship = load_facet_relationship("personal", "target_entity")
    assert relationship is not None
    assert relationship["entity_id"] == "target_entity"
    assert relationship["description"] == "imported relationship"

    log_entries = _read_log(get_state_directory(import_env["key_prefix"]) / "facets" / "log.jsonl")
    assert log_entries[-1]["action"] == "resolved_apply"
    assert log_entries[-1]["resolved_by"] == "talent"


def test_resolve_facet_apply_facet_json_conflict(import_env):
    target_path = import_env["root"] / "facets" / "personal" / "facet.json"
    _write_json(target_path, {"title": "Local"})
    staged_file = "personal/facet_json/facet.json.staged.json"
    staged_path = get_state_directory(import_env["key_prefix"]) / "facets" / "staged" / staged_file
    _write_json(
        staged_path,
        {
            "reason": "facet_json_conflict",
            "source_content": {"title": "Remote"},
            "target_content": {"title": "Local"},
            "staged_at": "2026-04-14T00:00:00+00:00",
        },
    )

    result = runner.invoke(
        call_app,
        ["import", "resolve-facet", staged_file, "apply", "--source", "test-source"],
    )

    assert result.exit_code == 0
    assert not staged_path.exists()
    assert _read_json(target_path) == {"title": "Remote"}

    log_entries = _read_log(get_state_directory(import_env["key_prefix"]) / "facets" / "log.jsonl")
    assert log_entries[-1]["action"] == "resolved_apply"
    assert log_entries[-1]["item_id"] == "personal/facet.json"
    assert log_entries[-1]["resolved_by"] == "talent"


def test_resolve_facet_unmapped_entity_fails_without_mapping(import_env):
    staged_file = "personal/entity_relationship/entities__source_entity__entity.json.staged.json"
    staged_path = get_state_directory(import_env["key_prefix"]) / "facets" / "staged" / staged_file
    _write_json(
        staged_path,
        {
            "reason": "unmapped_entity",
            "source_entity_id": "source_entity",
            "explanation": "Entity 'source_entity' has no mapping in entities/state.json id_map",
            "source_path": "entities/source_entity/entity.json",
            "source_data": json.dumps({"entity_id": "source_entity"}, ensure_ascii=False, indent=2)
            + "\n",
            "staged_at": "2026-04-14T00:00:00+00:00",
        },
    )

    result = runner.invoke(
        call_app,
        ["import", "resolve-facet", staged_file, "apply", "--source", "test-source"],
    )

    assert result.exit_code == 1
    assert "Entity source_entity has no mapping yet. Run entity review first." in result.stderr
    assert staged_path.exists()


def test_resolve_facet_skip(import_env):
    staged_file = "personal/entity_relationship/entities__source_entity__entity.json.staged.json"
    staged_path = get_state_directory(import_env["key_prefix"]) / "facets" / "staged" / staged_file
    _write_json(
        staged_path,
        {
            "reason": "unmapped_entity",
            "source_entity_id": "source_entity",
            "explanation": "Entity 'source_entity' has no mapping in entities/state.json id_map",
            "source_path": "entities/source_entity/entity.json",
            "source_data": json.dumps({"entity_id": "source_entity"}, ensure_ascii=False, indent=2)
            + "\n",
            "staged_at": "2026-04-14T00:00:00+00:00",
        },
    )

    result = runner.invoke(
        call_app,
        ["import", "resolve-facet", staged_file, "skip", "--source", "test-source"],
    )

    assert result.exit_code == 0
    assert not staged_path.exists()
    assert load_facet_relationship("personal", "source_entity") is None


def test_resolve_source_not_found(import_env):
    result = runner.invoke(
        call_app,
        ["import", "list-staged", "--source", "nonexistent"],
    )

    assert result.exit_code == 1
    assert (
        "Import source 'nonexistent' not found. Check available sources in "
        "~/.local/share/solstone/app-storage/import/journal_sources/."
    ) in result.stderr
