# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json

import pytest

from solstone.think.models import record_provider_failure


def _read_health(tmp_path):
    return json.loads((tmp_path / "health" / "talents.json").read_text())


def test_record_provider_failure_appends_new_row(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    record_provider_failure(
        "google",
        "flash",
        "gemini-3-flash-preview",
        "cogitate",
        12345,
    )

    payload = _read_health(tmp_path)
    assert payload["summary"] == {"total": 1, "passed": 0, "skipped": 0, "failed": 1}
    row = payload["results"][0]
    assert row["provider"] == "google"
    assert row["tier"] == "flash"
    assert row["model"] == "gemini-3-flash-preview"
    assert row["interface"] == "cogitate"
    assert row["ok"] is False
    assert row["status"] == "quota_exhausted"
    assert row["message"] == "Quota exhausted; retry after 12345"
    assert row["elapsed_s"] == 0.0
    assert row["reset_at_ms"] == 12345
    assert isinstance(row["recorded_at"], str)


def test_record_provider_failure_updates_duplicate_key(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    record_provider_failure("google", "flash", "gemini", "cogitate", 100)
    record_provider_failure("google", "flash", "gemini", "cogitate", 200)

    payload = _read_health(tmp_path)
    assert len(payload["results"]) == 1
    row = payload["results"][0]
    assert row["reset_at_ms"] == 200
    assert row["message"] == "Quota exhausted; retry after 200"
    assert row["ok"] is False
    assert row["status"] == "quota_exhausted"
    assert row["elapsed_s"] == 0.0


def test_record_provider_failure_recomputes_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    health_dir = tmp_path / "health"
    health_dir.mkdir()
    (health_dir / "talents.json").write_text(
        json.dumps(
            {
                "results": [
                    {"provider": "openai", "status": "ok", "ok": True},
                    {"provider": "anthropic", "status": "skip", "ok": True},
                ],
                "summary": {"total": 999, "passed": 999, "skipped": 0, "failed": 0},
                "checked_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )

    record_provider_failure("google", "flash", "gemini", "cogitate", 300)

    payload = _read_health(tmp_path)
    assert payload["summary"] == {"total": 3, "passed": 1, "skipped": 1, "failed": 1}
    assert payload["checked_at"] == "2026-01-01T00:00:00+00:00"


def test_record_provider_failure_atomic_replace_failure_preserves_file(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    health_dir = tmp_path / "health"
    health_dir.mkdir()
    health_path = health_dir / "talents.json"
    original = {
        "results": [{"provider": "openai", "status": "ok", "ok": True}],
        "summary": {"total": 1, "passed": 1, "skipped": 0, "failed": 0},
        "checked_at": "2026-01-01T00:00:00+00:00",
    }
    health_path.write_text(json.dumps(original), encoding="utf-8")

    def fail_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr("solstone.think.models.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        record_provider_failure("google", "flash", "gemini", "cogitate", 400)

    assert json.loads(health_path.read_text()) == original
