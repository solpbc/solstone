# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
from pathlib import Path

import pytest

from solstone.convey import create_app


@pytest.fixture
def content_client():
    journal = Path(__file__).resolve().parent / "fixtures" / "journal"
    app = create_app(str(journal))
    return app.test_client()


def test_content_list_endpoint(content_client):
    response = content_client.get("/app/import/api/20260101_090000/content")

    assert response.status_code == 200
    data = response.get_json()
    assert data["total"] == 5
    assert data["source_type"] == "ics"
    assert data["source_display"] == "Calendar"
    assert data["months"] == {"202601": 5}


def test_content_list_pagination(content_client):
    response = content_client.get(
        "/app/import/api/20260101_090000/content?page=2&per_page=2"
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["page"] == 2
    assert data["per_page"] == 2
    assert data["pages"] == 3
    assert len(data["items"]) == 2


def test_content_list_search_filter(content_client):
    response = content_client.get("/app/import/api/20260101_090000/content?q=betaworks")

    assert response.status_code == 200
    data = response.get_json()
    assert data["total"] == 2
    assert all(
        "betaworks" in (item["title"] + item["preview"]).lower()
        for item in data["items"]
    )


def test_content_list_month_filter(content_client):
    response = content_client.get(
        "/app/import/api/20260101_100000/content?month=202601"
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["total"] == 3
    assert all(item["date"].startswith("202601") for item in data["items"])


def test_content_detail_endpoint(content_client):
    response = content_client.get("/app/import/api/20260101_090000/content/event-0")

    assert response.status_code == 200
    data = response.get_json()
    assert data["item"]["title"] == "Weekly Engineering Standup"
    assert data["content"][0]["type"] == "markdown"
    assert "Weekly Engineering Standup" in data["content"][0]["content"]


def test_content_endpoint_404_for_missing_import(content_client):
    response = content_client.get("/app/import/api/20990101_000000/content")

    assert response.status_code == 404
    assert response.get_json()["error"] == "Import not found"


def test_content_detail_404_for_missing_item(content_client):
    response = content_client.get(
        "/app/import/api/20260101_090000/content/missing-item"
    )

    assert response.status_code == 404
    assert response.get_json()["error"] == "Item not found"


def test_content_lazy_backfill(tmp_path):
    journal_root = tmp_path
    import_dir = journal_root / "imports" / "20260101_120000"
    seg_dir = journal_root / "chronicle" / "20260101" / "import.chatgpt" / "120000_300"
    import_dir.mkdir(parents=True)
    seg_dir.mkdir(parents=True)

    (seg_dir / "conversation_transcript.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"topics": "planning"}),
                json.dumps({"speaker": "Human", "text": "hello"}),
                json.dumps({"speaker": "Assistant", "text": "hi"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (import_dir / "imported.json").write_text(
        json.dumps(
            {
                "source_type": "chatgpt",
                "all_created_files": [
                    "20260101/import.chatgpt/120000_300/conversation_transcript.jsonl",
                ],
            }
        ),
        encoding="utf-8",
    )

    app = create_app(str(journal_root))
    client = app.test_client()

    response = client.get("/app/import/api/20260101_120000/content")

    assert response.status_code == 200
    data = response.get_json()
    assert data["total"] == 1
    assert (import_dir / "content_manifest.jsonl").exists()
