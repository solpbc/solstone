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

    (target / "20260101" / "120000_60").mkdir(parents=True)
    (target / "20260101" / "120000_60" / "audio.jsonl").write_text(
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

    (source / "sol").mkdir(parents=True)
    (source / "sol" / "self.md").write_text("source sol\n", encoding="utf-8")
    (source / "config").mkdir(parents=True)
    (source / "config" / "source-only.json").write_text("{}", encoding="utf-8")

    (source / "imports" / "20260101_120000").mkdir(parents=True)
    (source / "imports" / "20260101_120000" / "manifest.json").write_text(
        '{"manifest": "source"}\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(target))
    import think.utils

    think.utils._journal_path_cache = None
    yield {"target": target, "source": source}
    think.utils._journal_path_cache = None


def test_segment_copy(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    assert (paths["target"] / "20260101" / "143022_300" / "audio.jsonl").exists()


def test_segment_skip(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    assert (paths["target"] / "20260101" / "120000_60" / "audio.jsonl").read_text(
        encoding="utf-8"
    ) == '{"audio": "target-existing-segment"}\n'


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
    merged = _read_json(
        paths["target"] / "entities" / "alice_johnson_2" / "entity.json"
    )
    assert merged["id"] == "alice_johnson_2"
    assert merged["name"] == "Alice Cooper"


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
    _write_jsonl(
        paths["source"] / "facets" / "work" / "calendar" / "20260101.jsonl",
        [
            {"title": "Duplicate event", "start": "09:00"},
            {"title": "Source event", "start": "10:00"},
        ],
    )
    _write_jsonl(
        paths["target"] / "facets" / "work" / "calendar" / "20260101.jsonl",
        [
            {"title": "Duplicate event", "start": "09:00"},
            {"title": "Target event", "start": "11:00"},
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
    (paths["source"] / "facets" / "work" / "activities").mkdir(parents=True)
    (paths["source"] / "facets" / "work" / "activities" / "skip.txt").write_text(
        "skip\n",
        encoding="utf-8",
    )
    (paths["source"] / "facets" / "work" / "logs").mkdir(parents=True)
    (paths["source"] / "facets" / "work" / "logs" / "20260101.jsonl").write_text(
        '{"log": "skip"}\n',
        encoding="utf-8",
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

    events = _read_jsonl(
        paths["target"] / "facets" / "work" / "calendar" / "20260101.jsonl"
    )
    assert {(item["title"], item["start"]) for item in events} == {
        ("Duplicate event", "09:00"),
        ("Source event", "10:00"),
        ("Target event", "11:00"),
    }

    assert (paths["target"] / "facets" / "work" / "news" / "20260102.md").read_text(
        encoding="utf-8"
    ) == "source new\n"
    assert (paths["target"] / "facets" / "work" / "news" / "20260101.md").read_text(
        encoding="utf-8"
    ) == "target duplicate\n"
    assert not (paths["target"] / "facets" / "work" / "activities").exists()
    assert not (paths["target"] / "facets" / "work" / "logs").exists()
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


def test_source_sol_skipped(merge_journals_fixture, monkeypatch):
    paths = merge_journals_fixture
    _mock_indexer(monkeypatch)

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    assert not (paths["target"] / "sol" / "self.md").exists()


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
    assert not (paths["target"] / "20260101" / "143022_300").exists()
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

    import think.tools.journal_merge as journal_merge_module

    real_copytree = shutil.copytree
    bad_segment = paths["source"] / "20260101" / "150000_60"

    def failing_copytree(src, dst, *args, **kwargs):
        if Path(src) == bad_segment:
            raise OSError("boom")
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(journal_merge_module.shutil, "copytree", failing_copytree)

    result = runner.invoke(call_app, ["journal", "merge", str(paths["source"])])

    assert result.exit_code == 0
    assert (paths["target"] / "20260101" / "143022_300" / "audio.jsonl").exists()
    assert (paths["target"] / "entities" / "bob_smith" / "entity.json").exists()
    assert "1 errors:" in result.output
