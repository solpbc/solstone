# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for convey app placeholder and attention behavior."""

import pytest


@pytest.fixture(autouse=True)
def _temp_journal(monkeypatch, tmp_path):
    """Ensure journaling defaults remain isolated from developer data."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))


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

    def test_p1_recent_import(self):
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

        monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

        today = datetime.now().strftime("%Y%m%d")
        agents_dir = tmp_path / "talents"
        agents_dir.mkdir()
        day_index = agents_dir / f"{today}.jsonl"
        day_index.write_text(
            json.dumps(
                {
                    "use_id": "1",
                    "name": "flow",
                    "day": today,
                    "ts": 1000,
                    "status": "error",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "use_id": "2",
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

        monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

        today = datetime.now().strftime("%Y%m%d")
        agents_dir = tmp_path / "talents"
        agents_dir.mkdir()
        day_index = agents_dir / f"{today}.jsonl"
        day_index.write_text(
            json.dumps(
                {
                    "use_id": "1",
                    "name": "flow",
                    "day": today,
                    "ts": 1000,
                    "status": "error",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "use_id": "3",
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

    def test_priority_p0_over_p1_imports(self, tmp_path, monkeypatch):
        """P0 (cortex errors) takes priority over P1 (recent import)."""
        import json
        from datetime import datetime

        from convey.apps import _resolve_attention

        monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

        today = datetime.now().strftime("%Y%m%d")
        agents_dir = tmp_path / "talents"
        agents_dir.mkdir()
        day_index = agents_dir / f"{today}.jsonl"
        day_index.write_text(
            json.dumps(
                {
                    "use_id": "1",
                    "name": "flow",
                    "day": today,
                    "ts": 1000,
                    "status": "error",
                }
            )
            + "\n"
        )

        current = {
            "imports": {
                "has_imported": True,
                "last_completed": datetime.now().isoformat(),
                "last_result_summary": "10 items",
            }
        }
        result = _resolve_attention(current)
        assert result is not None
        assert "error" in result.placeholder_text.lower()

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

        monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

        today = datetime.now().strftime("%Y%m%d")
        agents_dir = tmp_path / "talents"
        agents_dir.mkdir()
        day_index = agents_dir / f"{today}.jsonl"
        day_index.write_text(
            json.dumps({"use_id": "1", "name": "flow", "ts": 1000, "status": "error"})
            + "\n"
        )
        result = _resolve_attention({})
        assert result is not None
        assert len(result.placeholder_text) <= 90

        day_index.unlink()
        agents_dir.rmdir()
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

        monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

        today = datetime.now().strftime("%Y%m%d")
        agents_dir = tmp_path / today / "talents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "flow.md").write_text("# Flow")
        (agents_dir / "meetings.md").write_text("# Meetings")

        current = {"journal": {"first_daily_ready": True}}
        result = _resolve_attention(current)
        assert result is not None
        assert "2" in result.placeholder_text
        assert "report" in result.placeholder_text.lower()
        assert len(result.placeholder_text) <= 90
