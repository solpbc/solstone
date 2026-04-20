# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Benchmark CLI — local-model speed heuristics for the Ollama provider.

Verbs:

- ``sol call benchmark profile`` — probe host hardware, cache result.
- ``sol call benchmark list-models`` — pre-vetted + installed models with
  estimated output tok/s.
- ``sol call benchmark estimate <model-id>`` — single-model estimate.

Writes only to ``journal/health/hardware.json`` (via ``think.hardware``);
the pre-vetted registry and reference tables are in-repo static data.
"""

from __future__ import annotations

import json as jsonlib
import logging
from typing import Any

import typer

from think.benchmark import (
    estimate_output_tok_s,
    list_prevetted_models,
    load_registry,
    resolve_hardware_class,
)
from think.hardware import load_hardware, probe_hardware

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="benchmark",
    help="Estimate local-model performance without running the models.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("profile")
def profile(
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
) -> None:
    """Probe CPU / RAM / NVIDIA GPUs and cache the result.

    Writes to ``journal/health/hardware.json``. Safe to re-run.
    """
    payload = probe_hardware()
    hardware_class = _resolved_class(payload)
    payload_out = dict(payload)
    payload_out["hardware_class"] = hardware_class

    if json:
        typer.echo(jsonlib.dumps(payload_out, indent=2))
        return

    cpu = payload.get("cpu", {}) or {}
    typer.echo(f"Platform:       {payload.get('platform', 'unknown')}")
    typer.echo(f"CPU:            {cpu.get('model', 'unknown')}")
    typer.echo(f"  cores/threads: {cpu.get('cores', 0)} / {cpu.get('threads', 0)}")
    typer.echo(f"RAM:            {payload.get('ram_gb', 0)} GB")

    gpus = payload.get("gpus") or []
    if gpus:
        typer.echo("GPUs:")
        for gpu in gpus:
            vram = gpu.get("vram_gb")
            if vram is None or gpu.get("unified_memory"):
                vram_str = "unified memory"
            else:
                vram_str = f"{vram} GB VRAM"
            typer.echo(f"  - {gpu.get('name')}  {vram_str}  driver {gpu.get('driver')}")
    else:
        typer.echo("GPUs:           none detected (CPU-only)")

    typer.echo(f"Hardware class: {hardware_class}")


@app.command("list-models")
def list_models(
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
) -> None:
    """List pre-vetted models with installed status + speed estimates."""
    hardware = load_hardware()
    installed = _list_installed_models()

    rows = list_prevetted_models(hardware)
    for row in rows:
        row["installed"] = row["model_id"] in installed

    if json:
        typer.echo(
            jsonlib.dumps(
                {
                    "hardware_probed": hardware is not None,
                    "hardware_class": (
                        rows[0]["estimate"]["hardware_class"] if rows else "cpu-only"
                    ),
                    "models": rows,
                },
                indent=2,
            )
        )
        return

    if hardware is None:
        typer.echo(
            "Note: hardware not yet probed — run 'sol call benchmark profile' "
            "for accurate estimates. Showing registry with unknown confidence."
        )
        typer.echo("")

    header = (
        f"{'MODEL':40} {'INSTALLED':10} {'TIER':5} {'SIZE':>6} "
        f"{'VRAM':>5} {'FIT':4} {'TOK/S':>7} {'CONF':12}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))
    for row in rows:
        est = row["estimate"]
        tok_s = "?" if est["tok_s"] is None else f"{est['tok_s']:.1f}"
        fit = "ok" if row["fits_in_vram"] else "NO"
        typer.echo(
            f"{_truncate(row['model_id'], 40):40} "
            f"{'yes' if row['installed'] else 'no':10} "
            f"{row.get('tier_hint') or '-':<5} "
            f"{row.get('size_gb') or '?':>6} "
            f"{row.get('vram_required_gb') or 0:>5} "
            f"{fit:4} "
            f"{tok_s:>7} "
            f"{est['confidence']:12}"
        )


@app.command("estimate")
def estimate(
    model_id: str = typer.Argument(..., help="Model ID, e.g. ollama-local/qwen3.5:9b"),
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
) -> None:
    """Estimate output tok/s for a single model on the probed hardware."""
    hardware = load_hardware()
    if hardware is None:
        typer.echo(
            "Hardware not yet probed. Run 'sol call benchmark profile' first.",
            err=True,
        )
        raise typer.Exit(1)

    registry = load_registry()
    if model_id not in registry.get("models", {}):
        typer.echo(
            f"Model '{model_id}' is not in the pre-vetted registry "
            f"(think/benchmark/models.json).",
            err=True,
        )
        raise typer.Exit(1)

    hardware_class = _resolved_class(hardware)
    est = estimate_output_tok_s(model_id, hardware_class)

    if json:
        typer.echo(
            jsonlib.dumps(
                {
                    "model_id": est.model_id,
                    "hardware_class": est.hardware_class,
                    "tok_s": est.tok_s,
                    "confidence": est.confidence,
                    "source_class": est.source_class,
                },
                indent=2,
            )
        )
        return

    tok_s = "unknown" if est.tok_s is None else f"{est.tok_s:.1f} tok/s"
    typer.echo(f"Model:          {model_id}")
    typer.echo(f"Hardware class: {hardware_class}")
    typer.echo(f"Estimate:       {tok_s}")
    typer.echo(f"Confidence:     {est.confidence}")
    if est.source_class and est.source_class != hardware_class:
        typer.echo(f"Source class:   {est.source_class} (interpolated)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolved_class(hardware: dict[str, Any]) -> str:
    """Resolve the hardware-class key from a probe payload."""
    gpus = hardware.get("gpus") or []
    if not gpus:
        return "cpu-only"
    return resolve_hardware_class(gpus[0].get("name"))


def _list_installed_models() -> set[str]:
    """Query Ollama ``/api/tags`` and return installed model IDs with prefix.

    Returns an empty set if Ollama is unreachable (not a hard error — the
    CLI should still work without Ollama running).
    """
    try:
        from think.providers.ollama import _OLLAMA_LOCAL_PREFIX, _get_client
    except ImportError:
        return set()

    try:
        client = _get_client()
        response = client.get("/api/tags", timeout=5.0)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.debug("Ollama /api/tags unreachable: %s", exc)
        return set()

    installed: set[str] = set()
    for entry in data.get("models", []) or []:
        name = entry.get("name")
        if name:
            installed.add(f"{_OLLAMA_LOCAL_PREFIX}{name}")
    return installed


def _truncate(text: str, width: int) -> str:
    """Truncate with ellipsis so long model IDs don't break the table."""
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"
