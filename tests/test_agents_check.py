# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import argparse
import asyncio
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


def test_run_check_writes_health_file(tmp_path, monkeypatch):
    """_run_check writes agents health results to JOURNAL_PATH/health/agents.json."""
    import think.agents as agents

    fake_registry = {"fake": object()}
    fake_defaults = {
        "fake": {
            1: "fake-pro-model",
            2: "fake-flash-model",
            3: "fake-lite-model",
        }
    }

    monkeypatch.setattr("think.providers.PROVIDER_REGISTRY", fake_registry)
    monkeypatch.setattr("think.models.PROVIDER_DEFAULTS", fake_defaults)
    monkeypatch.setattr(agents, "get_journal", lambda: str(tmp_path))
    monkeypatch.setattr(agents, "_check_generate", lambda *_args: (True, "ok"))

    async def mock_check_cogitate(*_args):
        return True, "ok"

    monkeypatch.setattr(agents, "_check_cogitate", mock_check_cogitate)

    args = argparse.Namespace(
        provider=None,
        interface=None,
        tier=None,
        json=False,
        timeout=1,
    )

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(agents._run_check(args))

    assert exc_info.value.code == 0

    health_file = tmp_path / "health" / "agents.json"
    assert health_file.exists()

    payload = json.loads(health_file.read_text())
    assert "results" in payload
    assert "summary" in payload
    assert "checked_at" in payload
    assert datetime.fromisoformat(payload["checked_at"]).tzinfo is not None
    assert payload["summary"]["passed"] > 0


def test_run_check_dedup_same_model(tmp_path, monkeypatch):
    """_run_check deduplicates checks when tiers resolve to the same model."""
    import think.agents as agents

    fake_registry = {"fake": object()}
    fake_defaults = {
        "fake": {
            1: "fake-same-model",
            2: "fake-same-model",
            3: "fake-same-model",
        }
    }

    monkeypatch.setattr("think.providers.PROVIDER_REGISTRY", fake_registry)
    monkeypatch.setattr("think.models.PROVIDER_DEFAULTS", fake_defaults)
    monkeypatch.setattr(agents, "get_journal", lambda: str(tmp_path))

    gen_mock = MagicMock(return_value=(True, "ok"))
    monkeypatch.setattr(agents, "_check_generate", gen_mock)

    cog_inner = MagicMock(return_value=(True, "ok"))

    async def mock_check_cogitate(*args):
        return cog_inner(*args)

    monkeypatch.setattr(agents, "_check_cogitate", mock_check_cogitate)

    args = argparse.Namespace(
        provider=None,
        interface=None,
        tier=None,
        json=False,
        timeout=1,
    )

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(agents._run_check(args))

    assert exc_info.value.code == 0
    assert gen_mock.call_count == 1
    assert cog_inner.call_count == 1

    health_file = tmp_path / "health" / "agents.json"
    assert health_file.exists()

    payload = json.loads(health_file.read_text())
    results = payload["results"]
    assert len(results) == 6
    assert payload["summary"]["total"] == 6
    assert payload["summary"]["passed"] == 6

    non_reused = [result for result in results if "reused_from" not in result]
    reused = [result for result in results if "reused_from" in result]
    assert len(non_reused) == 2
    assert len(reused) == 4
    assert all(result["reused_from"] == "pro" for result in reused)
    assert all(result["elapsed_s"] == 0.0 for result in reused)


def test_cortex_start_emits_agents_check(tmp_path):
    """Cortex startup requests an agents health check via supervisor."""
    from think.cortex import CortexService

    cortex = CortexService(journal_path=str(tmp_path))
    cortex.callosum = MagicMock()
    cortex.callosum.start.return_value = None
    cortex.shutdown_requested.set()

    with patch("think.cortex.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        with patch("think.cortex.time.sleep", return_value=None):
            cortex.start()

    cortex.callosum.emit.assert_any_call(
        "supervisor", "request", cmd=["sol", "agents", "check"]
    )
