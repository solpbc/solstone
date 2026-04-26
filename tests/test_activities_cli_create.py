# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json

from typer.testing import CliRunner

from apps.activities.call import app

runner = CliRunner()


def _write_detected_entities(tmp_path, facet: str, day: str, rows: list[dict]) -> None:
    entities_path = tmp_path / "facets" / facet / "entities" / f"{day}.jsonl"
    entities_path.parent.mkdir(parents=True, exist_ok=True)
    entities_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _configure_cli_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    monkeypatch.setenv("SOL_SKIP_SUPERVISOR_CHECK", "1")

    import think.utils

    think.utils._journal_path_cache = None

    from think.entities.loading import clear_entity_loading_cache

    clear_entity_loading_cache()


def _base_payload() -> dict:
    return {
        "title": "Team sync",
        "activity": "meeting",
    }


def _valid_participation_entry(**overrides) -> dict:
    entry = {
        "name": "JB",
        "role": "attendee",
        "source": "voice",
        "confidence": 0.98,
        "context": "Spoke during the meeting",
    }
    entry.update(overrides)
    return entry


def _invoke_create(payload: dict, *, day: str = "20260418", facet: str = "work"):
    return runner.invoke(
        app,
        [
            "create",
            "--facet",
            facet,
            "--day",
            day,
            "--since-segment",
            "090000_300",
            "--source",
            "cogitate",
        ],
        input=json.dumps(payload),
    )


def _read_written_record(
    tmp_path, *, day: str = "20260418", facet: str = "work"
) -> dict:
    records_path = tmp_path / "facets" / facet / "activities" / f"{day}.jsonl"
    lines = records_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_create_resolves_participation_entity_ids(tmp_path, monkeypatch):
    _configure_cli_env(tmp_path, monkeypatch)

    _write_detected_entities(
        tmp_path,
        "work",
        "20260418",
        [
            {
                "id": "john_borthwick",
                "type": "Person",
                "name": "John Borthwick",
                "aka": ["JB"],
            }
        ],
    )

    payload = _base_payload()
    payload["participation"] = [
        _valid_participation_entry(entity_id="fake_id", extra="keep-me"),
        _valid_participation_entry(
            name="Alex",
            role="mentioned",
            source="transcript",
            confidence=0.55,
            context="Mentioned as a follow-up owner",
            entity_id="fake_id",
        ),
    ]

    result = _invoke_create(payload)

    assert result.exit_code == 0
    record = _read_written_record(tmp_path)
    assert record["participation"][0]["entity_id"] == "john_borthwick"
    assert record["participation"][1]["entity_id"] is None
    assert record["participation"][0]["extra"] == "keep-me"
    assert "participation" in record["edits"][0]["fields"]

    import think.utils

    think.utils._journal_path_cache = None


def test_create_omits_participation_when_not_provided(tmp_path, monkeypatch):
    _configure_cli_env(tmp_path, monkeypatch)

    result = _invoke_create(_base_payload())

    assert result.exit_code == 0
    record = _read_written_record(tmp_path)
    assert "participation" not in record
    assert "participation" not in record["edits"][0]["fields"]

    import think.utils

    think.utils._journal_path_cache = None


def test_create_persists_empty_participation_array(tmp_path, monkeypatch):
    _configure_cli_env(tmp_path, monkeypatch)

    payload = _base_payload()
    payload["participation"] = []

    result = _invoke_create(payload)

    assert result.exit_code == 0
    record = _read_written_record(tmp_path)
    assert record["participation"] == []
    assert "participation" in record["edits"][0]["fields"]

    import think.utils

    think.utils._journal_path_cache = None


def test_create_rejects_non_list_participation(tmp_path, monkeypatch):
    _configure_cli_env(tmp_path, monkeypatch)

    payload = _base_payload()
    payload["participation"] = {"name": "JB"}
    activities_path = tmp_path / "facets" / "work" / "activities" / "20260418.jsonl"

    result = _invoke_create(payload)

    assert result.exit_code == 1
    assert "Error: participation must be an array" in result.output
    assert not activities_path.exists()

    import think.utils

    think.utils._journal_path_cache = None


def test_create_rejects_non_object_participation_entry(tmp_path, monkeypatch):
    _configure_cli_env(tmp_path, monkeypatch)

    payload = _base_payload()
    payload["participation"] = ["JB"]
    activities_path = tmp_path / "facets" / "work" / "activities" / "20260418.jsonl"

    result = _invoke_create(payload)

    assert result.exit_code == 1
    assert "Error: participation[0] must be an object" in result.output
    assert not activities_path.exists()

    import think.utils

    think.utils._journal_path_cache = None


def test_create_participation_resolver_is_read_only(tmp_path, monkeypatch):
    _configure_cli_env(tmp_path, monkeypatch)

    _write_detected_entities(
        tmp_path,
        "work",
        "20260418",
        [
            {
                "id": "john_borthwick",
                "type": "Person",
                "name": "John Borthwick",
                "aka": ["JB"],
            }
        ],
    )
    _write_detected_entities(
        tmp_path,
        "work",
        "20260417",
        [{"id": "other_person", "type": "Person", "name": "Other Person"}],
    )

    entities_dir = tmp_path / "facets" / "work" / "entities"
    snapshot_before = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in entities_dir.iterdir()
    }

    payload = _base_payload()
    payload["participation"] = [_valid_participation_entry()]

    result = _invoke_create(payload)

    snapshot_after = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in entities_dir.iterdir()
    }
    assert result.exit_code == 0
    assert snapshot_after == snapshot_before

    import think.utils

    think.utils._journal_path_cache = None


def test_create_rejects_bad_participation_name(tmp_path, monkeypatch):
    _configure_cli_env(tmp_path, monkeypatch)
    activities_path = tmp_path / "facets" / "work" / "activities" / "20260418.jsonl"

    cases = [
        (
            _valid_participation_entry(name=""),
            "Error: participation[0] requires a non-empty string 'name'",
        ),
        (
            _valid_participation_entry(name=7),
            "Error: participation[0] requires a non-empty string 'name'",
        ),
        (
            {
                key: value
                for key, value in _valid_participation_entry().items()
                if key != "name"
            },
            "Error: participation[0] requires a non-empty string 'name'",
        ),
    ]

    for entry, message in cases:
        payload = _base_payload()
        payload["participation"] = [entry]

        result = _invoke_create(payload)

        assert result.exit_code == 1
        assert message in result.output
        assert not activities_path.exists()

    import think.utils

    think.utils._journal_path_cache = None


def test_create_rejects_invalid_participation_fields(tmp_path, monkeypatch):
    _configure_cli_env(tmp_path, monkeypatch)
    activities_path = tmp_path / "facets" / "work" / "activities" / "20260418.jsonl"

    cases = [
        (
            _valid_participation_entry(role="observer"),
            "Error: participation[0] has invalid role 'observer' (must be one of ['attendee', 'mentioned'])",
        ),
        (
            _valid_participation_entry(source="calendar"),
            "Error: participation[0] has invalid source 'calendar' (must be one of ['other', 'screen', 'speaker_label', 'transcript', 'voice'])",
        ),
        (
            _valid_participation_entry(confidence="high"),
            "Error: participation[0] 'confidence' must be a number",
        ),
        (
            _valid_participation_entry(context=7),
            "Error: participation[0] 'context' must be a string",
        ),
    ]

    for entry, message in cases:
        payload = _base_payload()
        payload["participation"] = [entry]

        result = _invoke_create(payload)

        assert result.exit_code == 1
        assert message in result.output
        assert not activities_path.exists()

    import think.utils

    think.utils._journal_path_cache = None
