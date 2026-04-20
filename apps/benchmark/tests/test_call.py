# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for benchmark CLI commands (``sol call benchmark ...``)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

from typer.testing import CliRunner

from apps.benchmark import call as benchmark_call
from think.call import call_app

runner = CliRunner()


def _fake_smi_result(stdout: str):
    class _R:
        pass

    r = _R()
    r.returncode = 0
    r.stdout = stdout
    r.stderr = ""
    return r


class TestProfile:
    def test_profile_writes_health_file(self, journal_override):
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            result = runner.invoke(call_app, ["benchmark", "profile", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["hardware_class"] == "cpu-only"
        assert payload["gpus"] == []

        health_file = journal_override / "health" / "hardware.json"
        assert health_file.exists()

    def test_profile_detects_nvidia_gpu(self, journal_override):
        stdout = "NVIDIA GeForce RTX 4090, 24564, 550.144.03\n"
        with patch.object(subprocess, "run", return_value=_fake_smi_result(stdout)):
            result = runner.invoke(call_app, ["benchmark", "profile", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["hardware_class"] == "rtx-4090"
        assert len(payload["gpus"]) == 1
        assert payload["gpus"][0]["name"] == "NVIDIA GeForce RTX 4090"

    def test_profile_text_output_is_human_readable(self, journal_override):
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            result = runner.invoke(call_app, ["benchmark", "profile"])
        assert result.exit_code == 0
        assert "Platform:" in result.output
        assert "CPU:" in result.output
        assert "Hardware class:" in result.output


class TestListModels:
    def test_lists_without_hardware_probe(self, journal_override):
        with patch.object(benchmark_call, "_list_installed_models", return_value=set()):
            result = runner.invoke(call_app, ["benchmark", "list-models", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["hardware_probed"] is False
        assert len(payload["models"]) > 0
        for row in payload["models"]:
            assert row["installed"] is False

    def test_marks_installed_models(self, journal_override):
        # Seed with a probed-hardware file so we don't get the "unprobed" note.
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            runner.invoke(call_app, ["benchmark", "profile"])

        fake_installed = {"ollama-local/qwen3.5:9b"}
        with patch.object(
            benchmark_call, "_list_installed_models", return_value=fake_installed
        ):
            result = runner.invoke(call_app, ["benchmark", "list-models", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        by_id = {row["model_id"]: row for row in payload["models"]}
        assert by_id["ollama-local/qwen3.5:9b"]["installed"] is True
        assert by_id["ollama-local/qwen3.5:2b"]["installed"] is False


class TestEstimate:
    def test_requires_probed_hardware(self, journal_override):
        result = runner.invoke(
            call_app, ["benchmark", "estimate", "ollama-local/qwen3.5:9b"]
        )
        assert result.exit_code == 1
        assert (
            "probe" in result.output.lower() or "probe" in (result.stderr or "").lower()
        )

    def test_rejects_unknown_model(self, journal_override):
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            runner.invoke(call_app, ["benchmark", "profile"])
        result = runner.invoke(
            call_app, ["benchmark", "estimate", "ollama-local/not-a-real-model:1b"]
        )
        assert result.exit_code == 1

    def test_estimates_known_model(self, journal_override):
        stdout = "NVIDIA GeForce RTX 4090, 24564, 550.144.03\n"
        with patch.object(subprocess, "run", return_value=_fake_smi_result(stdout)):
            runner.invoke(call_app, ["benchmark", "profile"])
        result = runner.invoke(
            call_app,
            ["benchmark", "estimate", "ollama-local/qwen3.5:9b", "--json"],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["model_id"] == "ollama-local/qwen3.5:9b"
        assert payload["hardware_class"] == "rtx-4090"
        # Without seed measurements, confidence is unknown for now.
        assert payload["confidence"] in ("unknown", "measured", "interpolated")
