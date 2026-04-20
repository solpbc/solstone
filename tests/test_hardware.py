# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.hardware — the host hardware probe."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from think import hardware


@pytest.fixture
def journal_override(tmp_path, monkeypatch):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    return tmp_path


def _mock_smi(returncode: int, stdout: str, stderr: str = "") -> object:
    class _Result:
        pass

    r = _Result()
    r.returncode = returncode
    r.stdout = stdout
    r.stderr = stderr
    return r


class TestProbeNvidia:
    def test_parses_csv_output(self):
        stdout = (
            "NVIDIA GeForce RTX 4090, 24564, 550.144.03\n"
            "NVIDIA GeForce RTX 3090, 24576, 550.144.03\n"
        )
        with patch.object(
            subprocess, "run", return_value=_mock_smi(0, stdout)
        ) as mock_run:
            gpus = hardware._probe_nvidia()
        assert mock_run.called
        assert len(gpus) == 2
        assert gpus[0] == {
            "name": "NVIDIA GeForce RTX 4090",
            "vram_gb": 24.0,
            "driver": "550.144.03",
            "unified_memory": False,
        }
        assert gpus[1]["name"] == "NVIDIA GeForce RTX 3090"
        assert gpus[1]["unified_memory"] is False

    def test_unified_memory_gpu_kept_with_none_vram(self):
        """DGX Spark / Jetson report [N/A] for memory.total — keep the GPU."""
        stdout = "NVIDIA GB10, [N/A], 580.142\n"
        with patch.object(subprocess, "run", return_value=_mock_smi(0, stdout)):
            gpus = hardware._probe_nvidia()
        assert len(gpus) == 1
        assert gpus[0]["name"] == "NVIDIA GB10"
        assert gpus[0]["vram_gb"] is None
        assert gpus[0]["unified_memory"] is True

    def test_handles_missing_smi(self):
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            gpus = hardware._probe_nvidia()
        assert gpus == []

    def test_handles_timeout(self):
        with patch.object(
            subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0),
        ):
            gpus = hardware._probe_nvidia()
        assert gpus == []

    def test_handles_nonzero_exit(self):
        with patch.object(
            subprocess, "run", return_value=_mock_smi(9, "", "driver error")
        ):
            gpus = hardware._probe_nvidia()
        assert gpus == []

    def test_skips_blank_and_malformed_lines(self):
        stdout = (
            "NVIDIA GeForce RTX 4090, 24564, 550.144.03\n"
            "\n"
            "malformed-line\n"
            "NVIDIA GeForce RTX 3080, not-a-number, 550.144.03\n"
        )
        with patch.object(subprocess, "run", return_value=_mock_smi(0, stdout)):
            gpus = hardware._probe_nvidia()
        assert len(gpus) == 1
        assert gpus[0]["name"] == "NVIDIA GeForce RTX 4090"


class TestProbeHardwareWrites:
    def test_writes_health_file(self, journal_override):
        with patch.object(hardware, "_probe_nvidia", return_value=[]):
            payload = hardware.probe_hardware()
        health_file = journal_override / "health" / "hardware.json"
        assert health_file.exists()
        on_disk = json.loads(health_file.read_text())
        assert on_disk == payload
        assert "probed_at" in payload
        assert "cpu" in payload
        assert "ram_gb" in payload
        assert payload["gpus"] == []

    def test_load_hardware_returns_none_when_missing(self, journal_override):
        assert hardware.load_hardware() is None

    def test_load_hardware_roundtrip(self, journal_override):
        with patch.object(hardware, "_probe_nvidia", return_value=[]):
            payload = hardware.probe_hardware()
        assert hardware.load_hardware() == payload

    def test_load_hardware_handles_corrupt_file(self, journal_override):
        health_dir = journal_override / "health"
        health_dir.mkdir(parents=True, exist_ok=True)
        (health_dir / "hardware.json").write_text("{ not json")
        assert hardware.load_hardware() is None


class TestProbeRam:
    def test_reads_linux_meminfo(self, tmp_path, monkeypatch):
        fake = tmp_path / "meminfo"
        fake.write_text("MemTotal:       32925156 kB\nMemFree:         100 kB\n")
        original = Path.read_text

        def fake_read_text(self, *args, **kwargs):
            if str(self) == "/proc/meminfo":
                return fake.read_text()
            return original(self, *args, **kwargs)

        with patch("think.hardware.platform.system", return_value="Linux"):
            monkeypatch.setattr(Path, "read_text", fake_read_text)
            ram_gb = hardware._probe_ram_gb()
        # 32925156 kB ≈ 31.4 GB
        assert 30 <= ram_gb <= 32


class TestProbeCpu:
    def test_cpu_info_has_minimum_fields(self):
        info = hardware._probe_cpu()
        assert "model" in info
        assert "cores" in info
        assert "threads" in info
        assert isinstance(info["threads"], int)
