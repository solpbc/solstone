# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Host hardware probe.

Detects CPU, RAM, and NVIDIA GPU state for use by the benchmark heuristic
(``think.benchmark``). Raw probe data is cached at
``journal/health/hardware.json`` so downstream estimators can run without
re-probing.

This module owns the hardware cache file per CLAUDE.md § 7 L2.
It does **not** resolve the hardware to a benchmark-reference class —
that's the estimator's job (``think.benchmark.estimate``) so that updates
to the reference table take effect without needing to re-probe.

Linux + NVIDIA is the Phase-1 target. Missing ``nvidia-smi`` returns an
empty GPU list rather than failing; non-Linux hosts still get CPU and
RAM probes via the ``platform`` fallbacks.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from think.utils import get_journal

logger = logging.getLogger(__name__)

_HEALTH_FILE = "hardware.json"
_NVIDIA_SMI_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def probe_hardware() -> dict[str, Any]:
    """Probe the host CPU, RAM, and NVIDIA GPUs and cache the result.

    Returns the probe payload and writes it to
    ``<journal>/health/hardware.json``. Safe to call repeatedly — each
    invocation overwrites the previous cache with fresh data.
    """
    payload: dict[str, Any] = {
        "probed_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.system(),
        "cpu": _probe_cpu(),
        "ram_gb": _probe_ram_gb(),
        "gpus": _probe_nvidia(),
    }

    health_dir = Path(get_journal()) / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    (health_dir / _HEALTH_FILE).write_text(json.dumps(payload, indent=2))

    return payload


def load_hardware() -> dict[str, Any] | None:
    """Return the cached hardware probe, or ``None`` if not yet probed."""
    path = Path(get_journal()) / "health" / _HEALTH_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("hardware cache at %s unreadable: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _probe_nvidia() -> list[dict[str, Any]]:
    """Return one dict per detected NVIDIA GPU, or ``[]`` if none found.

    Uses ``nvidia-smi --query-gpu`` for CSV output. A missing or failing
    ``nvidia-smi`` is treated as "no NVIDIA GPU" — not an error.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("nvidia-smi unavailable: %s", exc)
        return []

    if result.returncode != 0:
        logger.debug(
            "nvidia-smi exited %s: %s", result.returncode, result.stderr.strip()
        )
        return []

    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, mem_mib, driver = parts[0], parts[1], parts[2]
        # Unified-memory systems (DGX Spark, Jetson) return "[N/A]" because
        # VRAM isn't a discrete resource. Keep the GPU; mark vram_gb as None
        # and let the estimator fall back to system RAM for fit checks.
        vram_gb: float | None
        if mem_mib == "[N/A]" or not mem_mib:
            vram_gb = None
            unified = True
        else:
            try:
                vram_gb = round(int(mem_mib) / 1024, 1)
            except ValueError:
                continue
            unified = False
        gpus.append(
            {
                "name": name,
                "vram_gb": vram_gb,
                "driver": driver,
                "unified_memory": unified,
            }
        )
    return gpus


def _probe_cpu() -> dict[str, Any]:
    """Return CPU model + core/thread counts, best-effort across platforms."""
    info: dict[str, Any] = {
        "model": platform.processor() or platform.machine() or "unknown",
        "cores": os.cpu_count() or 0,
        "threads": os.cpu_count() or 0,
    }

    if platform.system() == "Linux":
        lscpu = _read_lscpu()
        if lscpu.get("model"):
            info["model"] = lscpu["model"]
        if lscpu.get("cores"):
            info["cores"] = lscpu["cores"]
        if lscpu.get("threads"):
            info["threads"] = lscpu["threads"]

        # Fall back to /proc/cpuinfo only if lscpu didn't give us a model.
        if info["model"] in ("", "unknown", platform.machine()):
            model, cores = _read_proc_cpuinfo()
            if model:
                info["model"] = model
            if cores:
                info["cores"] = cores

    return info


def _read_lscpu() -> dict[str, Any]:
    """Parse ``lscpu`` for model name + socket/core/thread layout."""
    try:
        result = subprocess.run(
            ["lscpu"],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}

    model: str | None = None
    sockets: int | None = None
    cores_per_socket: int | None = None
    threads_per_core: int | None = None
    cpus_total: int | None = None
    for line in result.stdout.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "Model name" and model is None:
            model = value
        elif key == "Socket(s)":
            sockets = _safe_int(value)
        elif key == "Core(s) per socket":
            cores_per_socket = _safe_int(value)
        elif key == "Thread(s) per core":
            threads_per_core = _safe_int(value)
        elif key == "CPU(s)" and cpus_total is None:
            cpus_total = _safe_int(value)

    out: dict[str, Any] = {}
    if model:
        out["model"] = model
    # Prefer the total CPU(s) reported by lscpu — on hybrid-layout ARM chips
    # the per-socket math understates the actual core count.
    if cpus_total:
        out["threads"] = cpus_total
    if sockets and cores_per_socket:
        out["cores"] = sockets * cores_per_socket
        if threads_per_core and not cpus_total:
            out["threads"] = sockets * cores_per_socket * threads_per_core
    return out


def _safe_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _read_proc_cpuinfo() -> tuple[str | None, int | None]:
    """Fallback: parse ``/proc/cpuinfo`` when lscpu is unavailable."""
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return None, None

    model: str | None = None
    core_ids_by_socket: dict[str, set[str]] = {}
    current_physical: str | None = None

    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key in ("model name", "Model name") and model is None:
            model = value
        elif key == "physical id":
            current_physical = value
        elif key == "core id" and current_physical is not None:
            core_ids_by_socket.setdefault(current_physical, set()).add(value)

    cores: int | None = None
    if core_ids_by_socket:
        cores = sum(len(ids) for ids in core_ids_by_socket.values())
    return model, cores


def _probe_ram_gb() -> float:
    """Return total system RAM in GB (rounded to 1 decimal)."""
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    kib = int(line.split()[1])
                    return round(kib / (1024 * 1024), 1)
        except (OSError, ValueError, IndexError):
            pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return round((page_size * pages) / (1024**3), 1)
    except (ValueError, OSError):
        return 0.0
