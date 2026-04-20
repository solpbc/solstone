# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Local-model benchmark heuristic.

Given a probed host (see ``think.hardware``), estimates the expected
output tok/s for each pre-vetted Ollama model without requiring the
model to be pulled. Uses a reference table of measured tok/s per
canonical hardware class, interpolating by FP16 throughput × memory
bandwidth when the user's exact hardware isn't in the table.

Public API:

- ``load_reference()`` — load ``reference.json``
- ``load_registry()`` — load ``models.json``
- ``resolve_hardware_class(gpu_name)`` — alias lookup + fuzzy match
- ``estimate_output_tok_s(model_id, hardware_class)`` — single-model estimate
- ``list_prevetted_models(hardware)`` — full registry with estimates attached
"""

from think.benchmark.estimate import (
    Estimate,
    estimate_output_tok_s,
    list_prevetted_models,
    load_reference,
    load_registry,
    resolve_hardware_class,
)

__all__ = [
    "Estimate",
    "estimate_output_tok_s",
    "list_prevetted_models",
    "load_reference",
    "load_registry",
    "resolve_hardware_class",
]
