# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for journal merge command."""

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from think.call import call_app

runner = CliRunner()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in items),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _find_merge_artifact_root(target: Path) -> Path:
    merge_dir = target.parent / f"{target.name}.merge"
    runs = sorted(path for path in merge_dir.iterdir() if path.is_dir())
    assert len(runs) >= 1
    return runs[-1]


def _mock_indexer(monkeypatch):
    import think.tools.call as call_module

    calls = []

    def _run(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(call_module.subprocess, "run", _run)
    return calls


@pytest.fixture
def merge_journals_fixture(tmp_path, monkeypatch):
    target = tmp_path / "target"
    source = tmp_path / "source"

    (source / "20260101" / "143022_300").mkdir(parents=True)
    (source / "20260101" / "143022_300" / "audio.jsonl").write_text(
        '{"audio": "source-segment"}\n',
        encoding="utf-8",
    )
    (source / "20260101" / "120000_60").mkdir(parents=True)
    (source / "20260101" / "120000_60" / "audio.jsonl").write_text(
        '{"audio": "source-existing-segment"}\n',
        encoding="utf-8",
    )

    (target / "chronicle" / "20260101" / "120000_60").mkdir(parents=True)
    (target / "chronicle" / "20260101" / "120000_60" / "audio.jsonl").write_text(
        '{"audio": "target-existing-segment"}\n',
        encoding="utf-8",
    )

    _write_json(
        source / "entities" / "alice_johnson" / "entity.json",
        {
            "id": "alice_johnson",
            "name": "Alice Johnson",
            "type": "person",
            "aka": ["Ali"],
            "emails": ["alice@example.com"],
            "is_principal": False,
            "created_at": 1000,
        },
    )
    _write_json(
        target / "entities" / "alice_johnson" / "entity.json",
        {
            "id": "alice_johnson",
            "name": "Alice Johnson",
            "type": "person",
            "aka": ["AJ"],
            "emails": ["aj@work.com"],
            "is_principal": False,
            "created_at": 500,
        },
    )

    _write_json(
        source / "facets" / "work" / "facet.json",
        {"title": "Work"},
    )

    (source / "identity").mkdir(parents=True)
    (source / "identity" / "self.md").write_text("source identity\n", encoding="utf-8")
    (source / "config").mkdir(parents=True)
    (source / "config" / "source-only.json").write_text("{}", encoding="utf-8")

    (source / "imports" / "20260101_120000").mkdir(parents=True)
    (source / "imports" / "20260101_120000" / "manifest.json").write_text(
        '{"manifest": "source"}\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(target))
    import think.utils

    think.utils._journal_path_cache = None
    yield {"target": target, "source": source}
    think.utils._journal_path_cache = None


def test_segment_copy(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    assert (
        paths["target"] / "chronicle" / "20260101" / "143022_300" / "audio.jsonl"
    ).exists()


def test_segment_skip(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    assert (
        paths["target"] / "chronicle" / "20260101" / "120000_60" / "audio.jsonl"
    ).read_text(encoding="utf-8") == '{"audio": "target-existing-segment"}\n'


def test_entity_create(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)
    _write_json(
        paths["source"] / "entities" / "bob_smith" / "entity.json",
        {
            "id": "bob_smith",
            "name": "Bob Smith",
            "type": "person",
            "created_at": 2000,
        },
    )

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    merged = _read_json(paths["target"] / "entities" / "bob_smith" / "entity.json")
    assert merged["name"] == "Bob Smith"


def test_entity_merge_aka_emails(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    merged = _read_json(paths["target"] / "entities" / "alice_johnson" / "entity.json")
    assert merged["name"] == "Alice Johnson"
    assert merged["type"] == "person"
    assert merged["aka"] == ["AJ", "Ali"]
    assert merged["emails"] == ["aj@work.com", "alice@example.com"]


def test_entity_principal_dedup(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)
    _write_json(
        paths["target"] / "entities" / "jer" / "entity.json",
        {
            "id": "jer",
            "name": "Jer",
            "type": "person",
            "is_principal": True,
            "created_at": 1,
        },
    )
    _write_json(
        paths["source"] / "entities" / "principal_person" / "entity.json",
        {
            "id": "principal_person",
            "name": "Principal Person",
            "type": "person",
            "is_principal": True,
            "created_at": 2,
        },
    )

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    merged = _read_json(
        paths["target"] / "entities" / "principal_person" / "entity.json"
    )
    assert merged["is_principal"] is False


def test_entity_id_collision(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)
    _write_json(
        paths["source"] / "entities" / "alice_johnson" / "entity.json",
        {
            "id": "alice_johnson",
            "name": "Alice Cooper",
            "type": "person",
            "created_at": 3000,
        },
    )

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    assert not (
        paths["target"] / "entities" / "alice_johnson_2" / "entity.json"
    ).exists()
    artifact_root = _find_merge_artifact_root(paths["target"])
    staged = _read_json(artifact_root / "staging" / "alice_johnson" / "entity.json")
    assert staged["id"] == "alice_johnson"
    assert staged["name"] == "Alice Cooper"
    assert "staged" in result.output


def test_facet_copy_new(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)
    (paths["source"] / "facets" / "work" / "logs").mkdir(parents=True)
    (paths["source"] / "facets" / "work" / "logs" / "20260101.jsonl").write_text(
        '{"log": "copied"}\n',
        encoding="utf-8",
    )

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    assert (paths["target"] / "facets" / "work" / "facet.json").exists()
    assert (paths["target"] / "facets" / "work" / "logs" / "20260101.jsonl").exists()


def test_facet_merge_overlapping(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)

    _write_json(paths["target"] / "facets" / "work" / "facet.json", {"title": "Work"})
    _write_json(
        paths["source"]
        / "facets"
        / "work"
        / "entities"
        / "alice_johnson"
        / "entity.json",
        {
            "entity_id": "alice_johnson",
            "description": "Source description",
            "source_only": "keep-me",
        },
    )
    _write_json(
        paths["target"]
        / "facets"
        / "work"
        / "entities"
        / "alice_johnson"
        / "entity.json",
        {
            "entity_id": "alice_johnson",
            "description": "Target description",
            "target_only": "wins",
        },
    )
    _write_jsonl(
        paths["source"]
        / "facets"
        / "work"
        / "entities"
        / "alice_johnson"
        / "observations.jsonl",
        [
            {"content": "Shared fact", "observed_at": 100},
            {"content": "Source fact", "observed_at": 200},
        ],
    )
    _write_jsonl(
        paths["target"]
        / "facets"
        / "work"
        / "entities"
        / "alice_johnson"
        / "observations.jsonl",
        [
            {"content": "Shared fact", "observed_at": 100},
            {"content": "Target fact", "observed_at": 300},
        ],
    )
    _write_jsonl(
        paths["source"] / "facets" / "work" / "todos" / "20260101.jsonl",
        [
            {"text": "Duplicate todo", "created_at": 10},
            {"text": "Source todo", "created_at": 11},
        ],
    )
    _write_jsonl(
        paths["target"] / "facets" / "work" / "todos" / "20260101.jsonl",
        [
            {"text": "Duplicate todo", "created_at": 10},
            {"text": "Target todo", "created_at": 12},
        ],
    )
    (paths["source"] / "facets" / "work" / "news").mkdir(parents=True)
    (paths["target"] / "facets" / "work" / "news").mkdir(parents=True)
    (paths["source"] / "facets" / "work" / "news" / "20260101.md").write_text(
        "source duplicate\n",
        encoding="utf-8",
    )
    (paths["source"] / "facets" / "work" / "news" / "20260102.md").write_text(
        "source new\n",
        encoding="utf-8",
    )
    (paths["target"] / "facets" / "work" / "news" / "20260101.md").write_text(
        "target duplicate\n",
        encoding="utf-8",
    )
    _write_jsonl(
        paths["source"] / "facets" / "work" / "activities" / "activities.jsonl",
        [
            {"id": "coding", "name": "Coding"},
            {"id": "meeting", "name": "Meeting"},
        ],
    )
    _write_jsonl(
        paths["target"] / "facets" / "work" / "activities" / "activities.jsonl",
        [
            {"id": "coding", "name": "Coding"},
            {"id": "email", "name": "Email"},
        ],
    )
    _write_jsonl(
        paths["source"] / "facets" / "work" / "activities" / "20260101.jsonl",
        [
            {"id": "coding_100000_300", "activity": "coding"},
            {"id": "meeting_110000_300", "activity": "meeting"},
        ],
    )
    _write_jsonl(
        paths["target"] / "facets" / "work" / "activities" / "20260101.jsonl",
        [
            {"id": "coding_100000_300", "activity": "coding"},
        ],
    )
    source_output = (
        paths["source"]
        / "facets"
        / "work"
        / "activities"
        / "20260101"
        / "coding_100000_300"
    )
    source_output.mkdir(parents=True)
    (source_output / "session_review.md").write_text(
        "source review\n",
        encoding="utf-8",
    )
    _write_jsonl(
        paths["source"] / "facets" / "work" / "logs" / "20260101.jsonl",
        [{"action": "test_action", "ts": 1000}],
    )
    _write_jsonl(
        paths["source"] / "facets" / "work" / "entities" / "20260101.jsonl",
        [
            {"id": "alice_johnson", "type": "Person", "name": "Alice Johnson"},
            {"id": "bob_smith", "type": "Person", "name": "Bob Smith"},
        ],
    )
    _write_jsonl(
        paths["target"] / "facets" / "work" / "entities" / "20260101.jsonl",
        [
            {"id": "alice_johnson", "type": "Person", "name": "Alice Johnson"},
        ],
    )
    (paths["source"] / "facets" / "work" / "entities.jsonl").write_text(
        '{"skip": true}\n',
        encoding="utf-8",
    )

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    relationship = _read_json(
        paths["target"]
        / "facets"
        / "work"
        / "entities"
        / "alice_johnson"
        / "entity.json"
    )
    assert relationship["description"] == "Target description"
    assert relationship["target_only"] == "wins"
    assert relationship["source_only"] == "keep-me"

    observations = _read_jsonl(
        paths["target"]
        / "facets"
        / "work"
        / "entities"
        / "alice_johnson"
        / "observations.jsonl"
    )
    assert len(observations) == 3
    assert {item["content"] for item in observations} == {
        "Shared fact",
        "Source fact",
        "Target fact",
    }

    todos = _read_jsonl(
        paths["target"] / "facets" / "work" / "todos" / "20260101.jsonl"
    )
    assert {item["text"] for item in todos} == {
        "Duplicate todo",
        "Source todo",
        "Target todo",
    }

    assert (paths["target"] / "facets" / "work" / "news" / "20260102.md").read_text(
        encoding="utf-8"
    ) == "source new\n"
    assert (paths["target"] / "facets" / "work" / "news" / "20260101.md").read_text(
        encoding="utf-8"
    ) == "target duplicate\n"

    activities_config = _read_jsonl(
        paths["target"] / "facets" / "work" / "activities" / "activities.jsonl"
    )
    config_ids = {item["id"] for item in activities_config}
    assert config_ids == {"coding", "email", "meeting"}

    activity_records = _read_jsonl(
        paths["target"] / "facets" / "work" / "activities" / "20260101.jsonl"
    )
    record_ids = {item["id"] for item in activity_records}
    assert record_ids == {"coding_100000_300", "meeting_110000_300"}

    assert (
        paths["target"]
        / "facets"
        / "work"
        / "activities"
        / "20260101"
        / "coding_100000_300"
        / "session_review.md"
    ).exists()

    logs = _read_jsonl(paths["target"] / "facets" / "work" / "logs" / "20260101.jsonl")
    assert any(item.get("action") == "test_action" for item in logs)

    detected = _read_jsonl(
        paths["target"] / "facets" / "work" / "entities" / "20260101.jsonl"
    )
    detected_ids = {item["id"] for item in detected}
    assert detected_ids == {"alice_johnson", "bob_smith"}

    assert not (paths["target"] / "facets" / "work" / "entities.jsonl").exists()


def test_import_copy(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    assert (paths["target"] / "imports" / "20260101_120000" / "manifest.json").exists()


def test_import_skip(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)
    (paths["target"] / "imports" / "20260101_120000").mkdir(parents=True)
    (paths["target"] / "imports" / "20260101_120000" / "manifest.json").write_text(
        '{"manifest": "target"}\n',
        encoding="utf-8",
    )

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    assert (
        paths["target"] / "imports" / "20260101_120000" / "manifest.json"
    ).read_text(encoding="utf-8") == '{"manifest": "target"}\n'


def test_source_identity_skipped(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    assert not (paths["target"] / "identity" / "self.md").exists()


def test_source_config_skipped(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    assert not (paths["target"] / "config" / "source-only.json").exists()


def test_dry_run(merge_journals_fixture):
    paths = merge_journals_fixture
    _write_json(
        paths["source"] / "entities" / "bob_smith" / "entity.json",
        {
            "id": "bob_smith",
            "name": "Bob Smith",
            "type": "person",
            "created_at": 2000,
        },
    )

    result = runner.invoke(
        call_app,
        ["journal", "merge", str(paths["source"]), "--dry-run"],
    )

    assert result.exit_code == 0
    assert "Would merge:" in result.output
    assert not (paths["target"] / "chronicle" / "20260101" / "143022_300").exists()
    assert not (paths["target"] / "entities" / "bob_smith").exists()
    assert not (paths["target"] / "facets" / "work").exists()
    assert not (paths["target"] / "imports" / "20260101_120000").exists()


def test_invalid_source(tmp_path):
    result = runner.invoke(call_app, ["journal", "merge", str(tmp_path / "missing")])

    assert result.exit_code == 1
    assert "is not a directory" in result.output


def test_invalid_source_no_days(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    result = runner.invoke(call_app, ["journal", "merge", str(source)])

    assert result.exit_code == 1
    assert "does not appear to be a journal" in result.output


def test_error_resilience(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)
    _write_json(
        paths["source"] / "entities" / "bob_smith" / "entity.json",
        {
            "id": "bob_smith",
            "name": "Bob Smith",
            "type": "person",
            "created_at": 2000,
        },
    )
    (paths["source"] / "20260101" / "150000_60").mkdir(parents=True)
    (paths["source"] / "20260101" / "150000_60" / "audio.jsonl").write_text(
        '{"audio": "bad"}\n',
        encoding="utf-8",
    )

    import think.merge as journal_merge_module

    real_copytree = shutil.copytree
    bad_segment = paths["source"] / "20260101" / "150000_60"

    def failing_copytree(src, dst, *args, **kwargs):
        if Path(src) == bad_segment:
            raise OSError("boom")
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(journal_merge_module.shutil, "copytree", failing_copytree)

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    assert (
        paths["target"] / "chronicle" / "20260101" / "143022_300" / "audio.jsonl"
    ).exists()
    assert (paths["target"] / "entities" / "bob_smith" / "entity.json").exists()
    assert "1 errors:" in result.output


def test_decision_log_written(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    artifact_root = _find_merge_artifact_root(paths["target"])
    decision_log = artifact_root / "decisions.jsonl"
    assert decision_log.exists()
    entries = _read_jsonl(decision_log)
    assert entries
    for entry in entries:
        assert {"ts", "action", "item_type", "item_id", "reason"} <= set(entry)


def test_decision_log_entity_merge_snapshots(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    artifact_root = _find_merge_artifact_root(paths["target"])
    entries = _read_jsonl(artifact_root / "decisions.jsonl")
    entity_merged = next(
        entry for entry in entries if entry["action"] == "entity_merged"
    )
    assert "source" in entity_merged
    assert "target" in entity_merged
    assert "fields_changed" in entity_merged


def test_entity_staged_count_in_output(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)
    _write_json(
        paths["source"] / "entities" / "alice_johnson" / "entity.json",
        {
            "id": "alice_johnson",
            "name": "Alice Cooper",
            "type": "person",
            "created_at": 3000,
        },
    )

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    assert "staged" in result.output
