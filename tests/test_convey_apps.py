# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for convey app placeholder and attention behavior."""

from unittest.mock import patch

import pytest
from flask import Flask


@pytest.fixture(autouse=True)
def _temp_journal(monkeypatch, tmp_path):
    """Ensure journaling defaults remain isolated from developer data."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))


def _run_triage():
    """Run the triage endpoint with mocked state."""
    app = Flask(__name__)
    with (
        patch("convey.utils.spawn_agent", return_value="agent-1") as mock_spawn,
        patch("think.cortex_client.wait_for_agents", return_value=({}, [])),
        patch(
            "think.cortex_client.read_agent_events",
            return_value=[{"event": "finish", "result": "ok"}],
        ),
    ):
        from convey.triage import triage

        with app.test_request_context("/", method="POST", json={"message": "hello"}):
            response = triage()

    assert response.status_code == 200
    return mock_spawn

# --- Placeholder resolution ---


class TestPlaceholderResolution:
    def test_no_imports_young(self):
        from convey.apps import _resolve_placeholder

        result = _resolve_placeholder({}, 0)
        assert "Bring in past conversations" in result

    def test_no_daily(self):
        from convey.apps import _resolve_placeholder

        current = {"imports": {"has_imported": True}}
        result = _resolve_placeholder(current, 0)
        assert "Capture is running" in result

    def test_first_daily_young(self):
        from convey.apps import _resolve_placeholder

        current = {
            "imports": {"has_imported": True},
            "journal": {"first_daily_ready": True},
        }
        result = _resolve_placeholder(current, 1)
        assert "first daily analysis is ready" in result

    def test_first_daily_mid(self):
        from convey.apps import _resolve_placeholder

        current = {"journal": {"first_daily_ready": True}}
        result = _resolve_placeholder(current, 3)
        assert "daily analysis is ready" in result
        assert "first" not in result

    def test_first_daily_mature(self):
        from convey.apps import _resolve_placeholder

        current = {"journal": {"first_daily_ready": True}}
        result = _resolve_placeholder(current, 10)
        assert "Ask me about your day" in result

    def test_default_fallback(self):
        from convey.apps import _resolve_placeholder

        result = _resolve_placeholder({}, 5)
        assert "Capture is running" in result


class TestAttentionResolution:
    """Tests for _resolve_attention() and attention-aware placeholder resolution."""

    def test_no_attention_returns_none(self):
        from convey.apps import _resolve_attention

        assert _resolve_attention({}) is None

    def test_no_attention_empty_sections(self):
        from convey.apps import _resolve_attention

        current = {"imports": {"has_imported": True}, "journal": {}}
        assert _resolve_attention(current) is None

    def test_p1_capture_stale(self):
        from convey.apps import _resolve_attention

        current = {"capture": {"status": "stale", "last_seen": 1000.0}}
        result = _resolve_attention(current)
        assert result is not None
        assert (
            "offline" in result.placeholder_text.lower()
            or "stale" in result.placeholder_text.lower()
        )
        assert len(result.placeholder_text) <= 90
        assert any("capture" in line.lower() for line in result.context_lines)

    def test_p1_capture_ok_no_attention(self):
        from convey.apps import _resolve_attention

        current = {"capture": {"status": "ok", "last_seen": 1000.0}}
        assert _resolve_attention(current) is None

    def test_p2_recent_import(self):
        from datetime import datetime

        from convey.apps import _resolve_attention

        current = {
            "imports": {
                "has_imported": True,
                "last_completed": datetime.now().isoformat(),
                "last_result_summary": "142 Calendar events",
            }
        }
        result = _resolve_attention(current)
        assert result is not None
        assert "import" in result.placeholder_text.lower()
        assert len(result.placeholder_text) <= 90

    def test_p2_old_import_no_attention(self):
        from datetime import datetime, timedelta

        from convey.apps import _resolve_attention

        old_time = (datetime.now() - timedelta(hours=2)).isoformat()
        current = {
            "imports": {
                "has_imported": True,
                "last_completed": old_time,
                "last_result_summary": "142 Calendar events",
            }
        }
        assert _resolve_attention(current) is None

    def test_p0_cortex_errors(self, tmp_path, monkeypatch):
        """Cortex errors are P0 — highest priority."""
        import json
        from datetime import datetime

        from convey.apps import _resolve_attention

        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

        today = datetime.now().strftime("%Y%m%d")
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        day_index = agents_dir / f"{today}.jsonl"
        day_index.write_text(
            json.dumps(
                {
                    "agent_id": "1",
                    "name": "flow",
                    "day": today,
                    "ts": 1000,
                    "status": "error",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "agent_id": "2",
                    "name": "meetings",
                    "day": today,
                    "ts": 1001,
                    "status": "completed",
                }
            )
            + "\n"
        )

        result = _resolve_attention({})
        assert result is not None
        assert "error" in result.placeholder_text.lower()
        assert "1" in result.placeholder_text
        assert len(result.placeholder_text) <= 90

    def test_p0_self_healing(self, tmp_path, monkeypatch):
        """An error followed by a success for the same agent is resolved."""
        import json
        from datetime import datetime

        from convey.apps import _resolve_attention

        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

        today = datetime.now().strftime("%Y%m%d")
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        day_index = agents_dir / f"{today}.jsonl"
        day_index.write_text(
            json.dumps(
                {
                    "agent_id": "1",
                    "name": "flow",
                    "day": today,
                    "ts": 1000,
                    "status": "error",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "agent_id": "3",
                    "name": "flow",
                    "day": today,
                    "ts": 2000,
                    "status": "completed",
                }
            )
            + "\n"
        )

        result = _resolve_attention({})
        assert result is None

    def test_priority_p0_over_p1(self, tmp_path, monkeypatch):
        """P0 (cortex errors) takes priority over P1 (capture stale)."""
        import json
        from datetime import datetime

        from convey.apps import _resolve_attention

        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

        today = datetime.now().strftime("%Y%m%d")
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        day_index = agents_dir / f"{today}.jsonl"
        day_index.write_text(
            json.dumps(
                {
                    "agent_id": "1",
                    "name": "flow",
                    "day": today,
                    "ts": 1000,
                    "status": "error",
                }
            )
            + "\n"
        )

        current = {"capture": {"status": "stale", "last_seen": 1000.0}}
        result = _resolve_attention(current)
        assert result is not None
        assert "error" in result.placeholder_text.lower()

    def test_priority_p1_over_p2(self):
        """P1 (capture stale) takes priority over P2 (recent import)."""
        from datetime import datetime

        from convey.apps import _resolve_attention

        current = {
            "capture": {"status": "stale"},
            "imports": {
                "has_imported": True,
                "last_completed": datetime.now().isoformat(),
                "last_result_summary": "10 items",
            },
        }
        result = _resolve_attention(current)
        assert result is not None
        assert (
            "offline" in result.placeholder_text.lower()
            or "capture" in result.placeholder_text.lower()
        )

    def test_placeholder_with_attention_overrides_daily(self, tmp_path, monkeypatch):
        """Attention items override regular daily analysis placeholders."""
        from convey.apps import _resolve_placeholder

        current = {
            "capture": {"status": "stale"},
            "journal": {"first_daily_ready": True},
        }
        result = _resolve_placeholder(current, 10)
        assert "offline" in result.lower() or "capture" in result.lower()

    def test_placeholder_no_attention_preserves_behavior(self):
        """When no attention items, existing placeholder logic unchanged."""
        from convey.apps import _resolve_placeholder

        current = {"journal": {"first_daily_ready": True}}
        result = _resolve_placeholder(current, 10)
        assert "Ask me about your day" in result

    def test_all_placeholder_texts_under_90_chars(self, tmp_path, monkeypatch):
        """All attention placeholder texts must be <=90 characters."""
        import json
        from datetime import datetime

        from convey.apps import _resolve_attention

        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

        today = datetime.now().strftime("%Y%m%d")
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        day_index = agents_dir / f"{today}.jsonl"
        day_index.write_text(
            json.dumps({"agent_id": "1", "name": "flow", "ts": 1000, "status": "error"})
            + "\n"
        )
        result = _resolve_attention({})
        assert result is not None
        assert len(result.placeholder_text) <= 90

        day_index.unlink()
        agents_dir.rmdir()
        result = _resolve_attention({"capture": {"status": "stale"}})
        assert result is not None
        assert len(result.placeholder_text) <= 90

        result = _resolve_attention(
            {
                "imports": {
                    "last_completed": datetime.now().isoformat(),
                    "last_result_summary": "142 Calendar events",
                }
            }
        )
        assert result is not None
        assert len(result.placeholder_text) <= 90

    def test_p3_daily_analysis(self, tmp_path, monkeypatch):
        """P3: daily analysis outputs available."""
        from datetime import datetime

        from convey.apps import _resolve_attention

        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

        today = datetime.now().strftime("%Y%m%d")
        agents_dir = tmp_path / today / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "flow.md").write_text("# Flow")
        (agents_dir / "meetings.md").write_text("# Meetings")

        current = {"journal": {"first_daily_ready": True}}
        result = _resolve_attention(current)
        assert result is not None
        assert "2" in result.placeholder_text
        assert "report" in result.placeholder_text.lower()
        assert len(result.placeholder_text) <= 90


class TestTriageSystemHealth:
    """Tests for system health context injection in triage."""

    def test_triage_injects_health_context_when_capture_stale(
        self, tmp_path, monkeypatch
    ):
        """System health context is added when attention items exist."""
        from think.awareness import update_state

        update_state("capture", {"status": "stale", "last_seen": 1000.0})

        mock = _run_triage()
        prompt = mock.call_args.kwargs["prompt"]
        assert "System health" in prompt
        assert "capture" in prompt.lower()

    def test_triage_no_health_context_when_healthy(self):
        """No system health context when nothing needs attention."""
        mock = _run_triage()
        prompt = mock.call_args.kwargs["prompt"]
        assert "System health" not in prompt
