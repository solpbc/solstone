# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import re
import time
from datetime import datetime

from solstone.think.entities.journal import save_journal_entity


def _action_log_rows(journal_root, day):
    log_path = journal_root / "config" / "actions" / f"{day}.jsonl"
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _create_journal_entity(entity_id, *, is_principal=False):
    save_journal_entity(
        {
            "id": entity_id,
            "name": entity_id.title(),
            "type": "Person",
            "is_principal": is_principal,
        }
    )


def test_delete_journal_entity_route_rejects_principal(
    client, journal_copy, monkeypatch
):
    entity_id = "principal-delete-test"
    today = datetime.now().strftime("%Y%m%d")
    _create_journal_entity(entity_id, is_principal=True)

    response = client.delete(f"/app/entities/api/journal/entity/{entity_id}")

    assert response.status_code == 400
    assert response.get_json() == {"error": "Cannot delete the principal (self) entity"}
    assert (journal_copy / "entities" / entity_id).exists()
    rows = _action_log_rows(journal_copy, today)
    assert not any(
        row["action"] == "journal_entity_delete"
        and row["params"].get("entity_id") == entity_id
        for row in rows
    )


def test_delete_journal_entity_route_rejects_missing_entity(client):
    response = client.delete("/app/entities/api/journal/entity/missing-entity")

    assert response.status_code == 400
    assert response.get_json() == {"error": "Entity 'missing-entity' not found"}


def test_delete_journal_entity_route_returns_pending_response_shape(
    client, journal_copy, monkeypatch
):
    entity_id = "pending-delete-test"
    today = datetime.now().strftime("%Y%m%d")
    _create_journal_entity(entity_id)
    monkeypatch.setattr("solstone.apps.entities.routes.ENTITY_DELETE_TTL", 0.05)
    before_ms = int(time.time() * 1000)

    response = client.delete(f"/app/entities/api/journal/entity/{entity_id}")

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert re.fullmatch(r"[0-9a-f]{32}", data["pending"])
    assert data["ttl_seconds"] == 0.05
    assert data["commit_at_ms"] >= before_ms
    assert (journal_copy / "entities" / entity_id).exists()
    rows = _action_log_rows(journal_copy, today)
    assert any(
        row["action"] == "journal_entity_delete"
        and row["params"].get("entity_id") == entity_id
        and row["params"].get("phase") == "pending"
        for row in rows
    )
    time.sleep(0.2)


def test_cancel_delete_journal_entity_within_window_keeps_entity(
    client, journal_copy, monkeypatch
):
    entity_id = "cancel-delete-test"
    _create_journal_entity(entity_id)
    monkeypatch.setattr("solstone.apps.entities.routes.ENTITY_DELETE_TTL", 0.2)

    delete_response = client.delete(f"/app/entities/api/journal/entity/{entity_id}")
    pending_id = delete_response.get_json()["pending"]

    cancel_response = client.post(f"/app/entities/api/cancel-delete/{pending_id}")

    assert cancel_response.status_code == 200
    assert cancel_response.get_json() == {"cancelled": pending_id}
    time.sleep(0.3)
    assert (journal_copy / "entities" / entity_id).exists()


def test_cancel_delete_journal_entity_too_late_after_commit(
    client, journal_copy, monkeypatch
):
    entity_id = "late-delete-test"
    today = datetime.now().strftime("%Y%m%d")
    _create_journal_entity(entity_id)
    monkeypatch.setattr("solstone.apps.entities.routes.ENTITY_DELETE_TTL", 0.05)

    delete_response = client.delete(f"/app/entities/api/journal/entity/{entity_id}")
    pending_id = delete_response.get_json()["pending"]

    time.sleep(0.2)
    cancel_response = client.post(f"/app/entities/api/cancel-delete/{pending_id}")

    assert cancel_response.status_code == 410
    assert cancel_response.get_json() == {"error": "already committed or unknown"}
    assert not (journal_copy / "entities" / entity_id).exists()
    rows = _action_log_rows(journal_copy, today)
    assert any(
        row["action"] == "journal_entity_delete"
        and row["params"].get("entity_id") == entity_id
        and row["params"].get("phase") == "committed"
        for row in rows
    )


def test_cancel_delete_journal_entity_unknown_pending_id_returns_410(client):
    response = client.post(f"/app/entities/api/cancel-delete/{'b' * 32}")

    assert response.status_code == 410
    assert response.get_json() == {"error": "already committed or unknown"}
