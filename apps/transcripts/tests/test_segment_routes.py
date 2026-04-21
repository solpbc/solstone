# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
import re
import time
from datetime import datetime

import pytest

FIXTURE_DAY = "20260304"
FIXTURE_STREAM = "default"
FIXTURE_SEGMENT = "090000_300"


def _action_log_rows(journal_root, day):
    log_path = journal_root / "config" / "actions" / f"{day}.jsonl"
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize("stream", ["-bad", "Upper", "..bad"])
def test_segment_content_rejects_invalid_stream(client, stream):
    response = client.get(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{stream}/{FIXTURE_SEGMENT}"
    )

    assert response.status_code == 404
    assert response.get_json() == {"error": "Invalid stream format"}


@pytest.mark.parametrize("stream", ["-bad", "Upper", "..bad"])
def test_delete_segment_rejects_invalid_stream(client, stream):
    response = client.delete(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{stream}/{FIXTURE_SEGMENT}"
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "Invalid stream format"}


def test_segment_content_missing_segment_does_not_create_phantom_directory(
    client, journal_copy
):
    response = client.get("/app/transcripts/api/segment/29990101/default/090000_300")

    assert response.status_code == 404
    assert response.get_json() == {"error": "Segment directory not found"}
    assert not (journal_copy / "chronicle" / "29990101").exists()
    assert not (
        journal_copy / "chronicle" / "29990101" / "default" / "090000_300"
    ).exists()


def test_delete_missing_segment_does_not_create_phantom_directory(client, journal_copy):
    response = client.delete("/app/transcripts/api/segment/29990101/default/090000_300")

    assert response.status_code == 404
    assert response.get_json() == {"error": "Segment not found"}
    assert not (journal_copy / "chronicle" / "29990101").exists()
    assert not (
        journal_copy / "chronicle" / "29990101" / "default" / "090000_300"
    ).exists()


def test_segment_content_happy_path_returns_segment_payload(client):
    response = client.get(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{FIXTURE_STREAM}/{FIXTURE_SEGMENT}"
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["segment_key"] == FIXTURE_SEGMENT
    assert data["chunks"]
    assert "media_sizes" in data


def test_delete_segment_happy_path_removes_segment_directory(
    client, journal_copy, monkeypatch
):
    monkeypatch.setattr("apps.transcripts.routes.is_supervisor_up", lambda: True)
    monkeypatch.setattr("apps.transcripts.routes.SEGMENT_DELETE_TTL", 0.05)
    segment_dir = (
        journal_copy / "chronicle" / FIXTURE_DAY / FIXTURE_STREAM / FIXTURE_SEGMENT
    )

    response = client.delete(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{FIXTURE_STREAM}/{FIXTURE_SEGMENT}"
    )

    assert response.status_code == 200
    assert response.get_json()["deleted"] == FIXTURE_SEGMENT
    time.sleep(0.2)
    assert not segment_dir.exists()


def test_delete_segment_includes_search_index_warning_when_supervisor_is_down(
    client, monkeypatch
):
    monkeypatch.setattr("apps.transcripts.routes.SEGMENT_DELETE_TTL", 0.05)
    response = client.delete(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{FIXTURE_STREAM}/{FIXTURE_SEGMENT}"
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["deleted"] == FIXTURE_SEGMENT
    assert data["search_index_warning"] is True
    time.sleep(0.2)


def test_delete_segment_omits_search_index_warning_when_supervisor_is_up(
    client, monkeypatch
):
    monkeypatch.setattr("apps.transcripts.routes.is_supervisor_up", lambda: True)
    monkeypatch.setattr("apps.transcripts.routes.SEGMENT_DELETE_TTL", 0.05)

    response = client.delete(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{FIXTURE_STREAM}/{FIXTURE_SEGMENT}"
    )

    assert response.status_code == 200
    assert response.get_json()["deleted"] == FIXTURE_SEGMENT
    time.sleep(0.2)


def test_delete_segment_returns_pending_response_shape(client, monkeypatch):
    monkeypatch.setattr("apps.transcripts.routes.is_supervisor_up", lambda: True)
    monkeypatch.setattr("apps.transcripts.routes.SEGMENT_DELETE_TTL", 0.05)
    before_ms = int(time.time() * 1000)

    response = client.delete(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{FIXTURE_STREAM}/{FIXTURE_SEGMENT}"
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["deleted"] == FIXTURE_SEGMENT
    assert re.fullmatch(r"[0-9a-f]{32}", data["pending"])
    assert data["ttl_seconds"] == 0.05
    assert data["commit_at_ms"] >= before_ms
    time.sleep(0.2)


def test_cancel_delete_segment_within_window_keeps_directory(
    client, journal_copy, monkeypatch
):
    monkeypatch.setattr("apps.transcripts.routes.SEGMENT_DELETE_TTL", 0.2)
    segment_dir = (
        journal_copy / "chronicle" / FIXTURE_DAY / FIXTURE_STREAM / FIXTURE_SEGMENT
    )

    delete_response = client.delete(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{FIXTURE_STREAM}/{FIXTURE_SEGMENT}"
    )
    pending_id = delete_response.get_json()["pending"]

    cancel_response = client.post(f"/app/transcripts/api/cancel-delete/{pending_id}")

    assert cancel_response.status_code == 200
    assert cancel_response.get_json() == {"cancelled": pending_id}
    time.sleep(0.3)
    assert segment_dir.exists()


def test_cancel_delete_segment_too_late_after_commit(client, journal_copy, monkeypatch):
    monkeypatch.setattr("apps.transcripts.routes.SEGMENT_DELETE_TTL", 0.05)
    segment_dir = (
        journal_copy / "chronicle" / FIXTURE_DAY / FIXTURE_STREAM / FIXTURE_SEGMENT
    )

    delete_response = client.delete(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{FIXTURE_STREAM}/{FIXTURE_SEGMENT}"
    )
    pending_id = delete_response.get_json()["pending"]

    time.sleep(0.2)
    cancel_response = client.post(f"/app/transcripts/api/cancel-delete/{pending_id}")

    assert cancel_response.status_code == 410
    assert cancel_response.get_json() == {"error": "already committed or unknown"}
    assert not segment_dir.exists()


def test_cancel_delete_segment_unknown_pending_id_returns_410(client):
    response = client.post(f"/app/transcripts/api/cancel-delete/{'a' * 32}")

    assert response.status_code == 410
    assert response.get_json() == {"error": "already committed or unknown"}


def test_cancel_delete_segment_malformed_pending_id_returns_410(client):
    response = client.post("/app/transcripts/api/cancel-delete/not-hex")

    assert response.status_code == 410
    assert response.get_json() == {"error": "already committed or unknown"}


def test_delete_segment_writes_pending_and_committed_audit_rows(
    client, journal_copy, monkeypatch
):
    monkeypatch.setattr("apps.transcripts.routes.is_supervisor_up", lambda: True)
    monkeypatch.setattr("apps.transcripts.routes.SEGMENT_DELETE_TTL", 0.05)

    delete_response = client.delete(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{FIXTURE_STREAM}/{FIXTURE_SEGMENT}"
    )
    pending_id = delete_response.get_json()["pending"]

    day_rows = _action_log_rows(journal_copy, FIXTURE_DAY)
    assert any(
        row["action"] == "segment_delete"
        and row["params"].get("pending_id") == pending_id
        and row["params"].get("phase") == "pending"
        for row in day_rows
    )

    time.sleep(0.2)
    day_rows = _action_log_rows(journal_copy, FIXTURE_DAY)
    assert any(
        row["action"] == "segment_delete"
        and row["params"].get("pending_id") == pending_id
        and row["params"].get("phase") == "committed"
        for row in day_rows
    )


def test_cancel_delete_segment_writes_cancelled_audit_row(
    client, journal_copy, monkeypatch
):
    monkeypatch.setattr("apps.transcripts.routes.SEGMENT_DELETE_TTL", 0.2)
    cancel_response = client.delete(
        f"/app/transcripts/api/segment/{FIXTURE_DAY}/{FIXTURE_STREAM}/{FIXTURE_SEGMENT}"
    )
    cancel_pending_id = cancel_response.get_json()["pending"]
    cancel_result = client.post(
        f"/app/transcripts/api/cancel-delete/{cancel_pending_id}"
    )

    assert cancel_result.status_code == 200
    cancel_day = datetime.now().strftime("%Y%m%d")
    cancel_rows = _action_log_rows(journal_copy, cancel_day)
    assert any(
        row["action"] == "segment_delete"
        and row["params"].get("pending_id") == cancel_pending_id
        and row["params"].get("phase") == "cancelled"
        for row in cancel_rows
    )
