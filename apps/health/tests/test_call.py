# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for health app call commands."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from typer.testing import CliRunner

from apps.health.call import app

runner = CliRunner()


def test_pipeline_default_is_today(health_env):
    health_env()

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["day"] == datetime.now().strftime("%Y%m%d")


def test_pipeline_day_option(health_env):
    health_env()

    result = runner.invoke(app, ["--day", "20250101"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["day"] == "20250101"


def test_pipeline_yesterday_option(health_env):
    health_env()

    result = runner.invoke(app, ["--yesterday"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["day"] == (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")


def test_mutual_exclusion_error(health_env):
    health_env()

    result = runner.invoke(app, ["--day", "20250101", "--yesterday"])

    assert result.exit_code == 1
    assert "mutually exclusive" in result.stderr


def test_pipeline_with_real_fixture(health_env):
    env = health_env()
    day = "20260101"
    health_path = env.journal / day / "health" / "123_segment_dream.jsonl"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.write_text(
        "\n".join(
            [
                json.dumps({"event": "run.start", "mode": "segment"}),
                json.dumps({"event": "agent.dispatch", "mode": "segment"}),
                json.dumps({"event": "agent.complete", "mode": "segment"}),
                json.dumps(
                    {"event": "run.complete", "mode": "segment", "duration_ms": 42}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["--day", day])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["runs"]["segment"]["count"] == 1
    assert payload["agents"]["dispatched"] >= 1
