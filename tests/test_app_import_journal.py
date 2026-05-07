# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
from pathlib import Path

import pytest

from solstone.convey import create_app


@pytest.fixture(autouse=True)
def _temp_journal(monkeypatch, tmp_path):
    journal_root = tmp_path / "journal"
    journal_root.mkdir()
    config_dir = journal_root / "config"
    config_dir.mkdir()
    (config_dir / "journal.json").write_text(
        json.dumps(
            {
                "setup": {"completed_at": "2026-04-26T00:00:00Z"},
                "convey": {"trust_localhost": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal_root))
    return journal_root


@pytest.fixture
def client(_temp_journal):
    app = create_app(str(_temp_journal))
    return app.test_client()


def _write_import_detail(
    journal_root: Path,
    timestamp: str,
    *,
    imported_json: dict | None = None,
    import_json: dict | None = None,
) -> Path:
    import_dir = journal_root / "imports" / timestamp
    import_dir.mkdir(parents=True)
    if import_json is not None:
        (import_dir / "import.json").write_text(
            json.dumps(import_json),
            encoding="utf-8",
        )
    if imported_json is not None:
        (import_dir / "imported.json").write_text(
            json.dumps(imported_json),
            encoding="utf-8",
        )
    return import_dir


def test_import_sources_include_journal_archive(client):
    response = client.get("/app/import/api/sources")

    assert response.status_code == 200
    data = response.get_json()
    journal_source = next(item for item in data if item["name"] == "journal_archive")
    assert journal_source["display_name"] == "Journal"
    assert journal_source["emoji"] == "📓"
    assert journal_source["accept"] == ".zip"
    assert journal_source["has_guide"] is True
    assert journal_source["input_type"] == "file"


def test_import_detail_api_includes_merge_fields(client, _temp_journal):
    timestamp = "20260426_120000"
    merge_root = (
        _temp_journal.parent / f"{_temp_journal.name}.merge" / "20260426T120000Z"
    )
    decisions_path = merge_root / "decisions.jsonl"
    staging_path = merge_root / "staging"
    decisions_path.parent.mkdir(parents=True)
    decisions_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "action": "entity_staged",
                        "item_id": "source_person",
                        "source": {"name": "Source Person"},
                        "target": {"name": "Target Person"},
                        "staging_path": str(
                            staging_path / "source_person" / "entity.json"
                        ),
                    }
                ),
                json.dumps(
                    {
                        "action": "segment_errored",
                        "item_id": "20260101/default/090000_300",
                        "reason": "segment copy failed",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_import_detail(
        _temp_journal,
        timestamp,
        import_json={"original_filename": "journal.zip"},
        imported_json={
            "source_type": "journal_archive",
            "merge_summary": {"segments_copied": 1},
            "merge_log_path": str(decisions_path),
            "merge_staging_path": str(staging_path),
            "summary_errors": [
                "segment 20260101/default/090000_300: segment copy failed"
            ],
            "principal_collision": {
                "source_name": "Source Person",
                "target_name": "Target Person",
            },
        },
    )

    response = client.get(f"/app/import/api/{timestamp}")

    assert response.status_code == 200
    data = response.get_json()
    assert data["merge_artifact_paths"] == {
        "decisions": str(decisions_path),
        "staging": str(staging_path),
    }
    assert data["decision_highlights"] == {
        "staged_entities": [
            {
                "source_name": "Source Person",
                "target_name": "Target Person",
                "staging_path": str(staging_path / "source_person" / "entity.json"),
            }
        ],
        "errored_segments": [
            {
                "item_id": "20260101/default/090000_300",
                "reason": "segment copy failed",
            }
        ],
    }
    assert data["summary_errors"] == [
        "segment 20260101/default/090000_300: segment copy failed"
    ]
    assert data["imported_json"]["principal_collision"] == {
        "source_name": "Source Person",
        "target_name": "Target Person",
    }


def test_import_detail_api_omits_decision_highlights_without_qualifying_rows(
    client, _temp_journal
):
    timestamp = "20260426_120001"
    merge_root = (
        _temp_journal.parent / f"{_temp_journal.name}.merge" / "20260426T120001Z"
    )
    decisions_path = merge_root / "decisions.jsonl"
    staging_path = merge_root / "staging"
    decisions_path.parent.mkdir(parents=True)
    decisions_path.write_text(
        json.dumps(
            {
                "action": "segment_copied",
                "item_id": "20260101/default/090000_300",
                "reason": "new",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_import_detail(
        _temp_journal,
        timestamp,
        imported_json={
            "source_type": "journal_archive",
            "merge_summary": {"segments_copied": 1},
            "merge_log_path": str(decisions_path),
            "merge_staging_path": str(staging_path),
        },
    )

    response = client.get(f"/app/import/api/{timestamp}")

    assert response.status_code == 200
    data = response.get_json()
    assert "decision_highlights" not in data


def test_import_detail_api_omits_merge_fields_for_non_merge_import(
    client, _temp_journal
):
    timestamp = "20260426_120002"
    _write_import_detail(
        _temp_journal,
        timestamp,
        imported_json={
            "source_type": "chatgpt",
            "total_files_created": 2,
        },
    )

    response = client.get(f"/app/import/api/{timestamp}")

    assert response.status_code == 200
    data = response.get_json()
    assert "merge_artifact_paths" not in data
    assert "decision_highlights" not in data
    assert "summary_errors" not in data


def test_import_guide_for_journal_archive(client):
    response = client.get("/app/import/api/guide/journal_archive")

    assert response.status_code == 200
    assert "# Exporting Your Journal" in response.get_data(as_text=True)
