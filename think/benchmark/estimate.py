# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Estimator: map (hardware class, model) -> expected output tok/s.

When a measurement exists for the user's exact hardware class, that wins.
Otherwise the estimator picks the closest class in the model's benchmark
table (by ``fp16_tflops * mem_bandwidth_gbs``) and scales the measured
tok/s by the ratio. If the model has no measurements at all — or the
user is ``cpu-only`` and the model requires VRAM — the estimate is
returned with ``confidence="unknown"``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent
_REFERENCE_FILE = _DATA_DIR / "reference.json"
_REGISTRY_FILE = _DATA_DIR / "models.json"


Confidence = Literal["measured", "interpolated", "unknown"]


@dataclass(frozen=True)
class Estimate:
    """Single-model speed estimate."""

    model_id: str
    hardware_class: str
    tok_s: float | None
    confidence: Confidence
    source_class: str | None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_reference() -> dict[str, Any]:
    """Load ``reference.json`` (cached)."""
    return json.loads(_REFERENCE_FILE.read_text())


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    """Load ``models.json`` (cached)."""
    return json.loads(_REGISTRY_FILE.read_text())


# ---------------------------------------------------------------------------
# Hardware class resolution
# ---------------------------------------------------------------------------


def resolve_hardware_class(gpu_name: str | None) -> str:
    """Map an ``nvidia-smi`` GPU name to a canonical hardware class.

    Tries (1) exact alias match, (2) case-insensitive substring match
    against class labels, (3) fallback to ``"cpu-only"``.
    """
    if not gpu_name:
        return "cpu-only"

    ref = load_reference()
    aliases: dict[str, str] = ref.get("aliases", {})
    classes: dict[str, dict[str, Any]] = ref.get("classes", {})

    if gpu_name in aliases:
        return aliases[gpu_name]

    needle = gpu_name.lower()
    for alias_name, class_key in aliases.items():
        if alias_name.lower() in needle or needle in alias_name.lower():
            return class_key
    for class_key, class_spec in classes.items():
        label = str(class_spec.get("label", "")).lower()
        if label and (label in needle or needle in label):
            return class_key

    return "cpu-only"


def _class_throughput(class_key: str) -> float:
    """Return the TFLOPs × bandwidth product used as the interpolation proxy."""
    classes = load_reference().get("classes", {})
    spec = classes.get(class_key, {})
    return float(spec.get("fp16_tflops", 0.0)) * float(
        spec.get("mem_bandwidth_gbs", 0.0)
    )


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def estimate_output_tok_s(model_id: str, hardware_class: str) -> Estimate:
    """Estimate output tok/s for ``model_id`` on ``hardware_class``."""
    registry = load_registry()
    model = registry.get("models", {}).get(model_id)
    if model is None:
        return Estimate(
            model_id=model_id,
            hardware_class=hardware_class,
            tok_s=None,
            confidence="unknown",
            source_class=None,
        )

    benchmarks: dict[str, dict[str, Any]] = model.get("benchmarks", {}) or {}

    if hardware_class in benchmarks:
        tok_s = benchmarks[hardware_class].get("output_tok_s")
        if isinstance(tok_s, (int, float)):
            return Estimate(
                model_id=model_id,
                hardware_class=hardware_class,
                tok_s=float(tok_s),
                confidence="measured",
                source_class=hardware_class,
            )

    if hardware_class == "cpu-only" or not benchmarks:
        return Estimate(
            model_id=model_id,
            hardware_class=hardware_class,
            tok_s=None,
            confidence="unknown",
            source_class=None,
        )

    target = _class_throughput(hardware_class)
    if target <= 0:
        return Estimate(
            model_id=model_id,
            hardware_class=hardware_class,
            tok_s=None,
            confidence="unknown",
            source_class=None,
        )

    best_source: str | None = None
    best_source_throughput: float = 0.0
    best_distance: float = float("inf")
    for source_class, bench in benchmarks.items():
        tok_s = bench.get("output_tok_s")
        if not isinstance(tok_s, (int, float)):
            continue
        source_throughput = _class_throughput(source_class)
        if source_throughput <= 0:
            continue
        distance = abs(source_throughput - target)
        if distance < best_distance:
            best_distance = distance
            best_source = source_class
            best_source_throughput = source_throughput

    if best_source is None:
        return Estimate(
            model_id=model_id,
            hardware_class=hardware_class,
            tok_s=None,
            confidence="unknown",
            source_class=None,
        )

    source_tok_s = float(benchmarks[best_source]["output_tok_s"])
    scaled = source_tok_s * (target / best_source_throughput)
    return Estimate(
        model_id=model_id,
        hardware_class=hardware_class,
        tok_s=round(scaled, 1),
        confidence="interpolated",
        source_class=best_source,
    )


# ---------------------------------------------------------------------------
# Registry listing
# ---------------------------------------------------------------------------


def list_prevetted_models(hardware: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return each pre-vetted model with an estimate + VRAM fit flag.

    ``hardware`` is the cached probe from ``think.hardware.load_hardware()``;
    pass ``None`` for a hardware-agnostic listing (estimates all unknown).
    """
    hardware_class, user_vram_gb = _user_hardware(hardware)
    registry = load_registry()
    rows: list[dict[str, Any]] = []

    for model_id, spec in registry.get("models", {}).items():
        vram_required = float(spec.get("vram_required_gb") or 0)
        estimate = estimate_output_tok_s(model_id, hardware_class)
        rows.append(
            {
                "model_id": model_id,
                "label": spec.get("label"),
                "tier_hint": spec.get("tier_hint"),
                "size_gb": spec.get("size_gb"),
                "capabilities": spec.get("capabilities", []),
                "vram_required_gb": vram_required,
                "fits_in_vram": (user_vram_gb is None or user_vram_gb >= vram_required),
                "notes": spec.get("notes"),
                "estimate": {
                    "tok_s": estimate.tok_s,
                    "confidence": estimate.confidence,
                    "source_class": estimate.source_class,
                    "hardware_class": estimate.hardware_class,
                },
            }
        )
    return rows


def _user_hardware(
    hardware: dict[str, Any] | None,
) -> tuple[str, float | None]:
    """Resolve (hardware_class, effective_vram_gb) from a probe payload.

    On unified-memory systems (Spark, Jetson) the GPU reports no discrete
    VRAM; effective VRAM for fit checks is the system RAM.
    """
    if not hardware:
        return "cpu-only", None
    gpus = hardware.get("gpus") or []
    if not gpus:
        return "cpu-only", 0.0
    primary = gpus[0]
    hardware_class = resolve_hardware_class(primary.get("name"))

    has_unified = any(g.get("unified_memory") for g in gpus)
    if has_unified:
        # Use system RAM as the memory ceiling; keep a small reserve for the OS.
        ram_gb = float(hardware.get("ram_gb") or 0)
        effective_vram = max(ram_gb - 8.0, 0.0) if ram_gb else None
        return hardware_class, effective_vram

    total_vram = sum(float(g.get("vram_gb") or 0) for g in gpus)
    return hardware_class, total_vram
