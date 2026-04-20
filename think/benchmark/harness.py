# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Benchmark harness — measure a model's output tok/s on this host.

Maintainer-only. Run by hand to seed ``models.json`` with real
measurements; not invoked by the live pipeline.

Usage::

    python -m think.benchmark.harness --model ollama-local/qwen3.5:9b \\
        --class rtx-4090

The script sends a fixed prompt to the local Ollama instance, records
output tok/s using Ollama's ``eval_count`` / ``eval_duration`` fields,
and prints a JSON snippet ready to paste into
``models.json -> models.<model_id>.benchmarks.<class>``.

It deliberately does **not** write ``models.json`` itself — the
maintainer reviews the output before committing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from typing import Any

# Fixed prompt used for every benchmark run so numbers are comparable
# across runs and across hardware. ~150-200 tokens of input, asking for
# ~200 tokens of output.
_PROMPT = (
    "You are a benchmarking fixture. Write a focused, concrete 200-word "
    "technical paragraph explaining the tradeoffs between dense and "
    "sparse-mixture-of-experts transformer architectures for on-device "
    "inference. Discuss parameter count, active-parameter count, memory "
    "bandwidth, and typical quantization strategies. Do not include "
    "headings, lists, or citations — a single paragraph of prose."
)
_MAX_OUTPUT_TOKENS = 256
_WARMUP_RUNS = 1
_MEASURE_RUNS = 3


def run_once(model: str) -> dict[str, Any]:
    """Send one completion request; return the raw Ollama response body."""
    from think.providers.ollama import _get_client, _strip_model_prefix

    bare_model = _strip_model_prefix(model)
    client = _get_client()
    response = client.post(
        "/api/chat",
        json={
            "model": bare_model,
            "messages": [{"role": "user", "content": _PROMPT}],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.2,
                "num_predict": _MAX_OUTPUT_TOKENS,
            },
        },
        timeout=600.0,
    )
    response.raise_for_status()
    return response.json()


def ensure_installed(model: str, *, allow_pull: bool) -> None:
    """Verify the model is installed locally; optionally trigger a pull."""
    from think.providers.ollama import _get_client, _strip_model_prefix

    bare_model = _strip_model_prefix(model)
    client = _get_client()
    response = client.get("/api/tags", timeout=10.0)
    response.raise_for_status()
    installed = {m.get("name") for m in response.json().get("models", [])}

    if bare_model in installed:
        return

    if not allow_pull:
        raise SystemExit(
            f"Model '{bare_model}' not installed. Run `ollama pull {bare_model}` "
            f"first, or pass --pull."
        )

    print(f"Pulling {bare_model}…", file=sys.stderr)
    with client.stream(
        "POST",
        "/api/pull",
        json={"name": bare_model},
        timeout=None,
    ) as stream:
        stream.raise_for_status()
        for line in stream.iter_lines():
            if line:
                print(line, file=sys.stderr)


def tok_s_from_response(body: dict[str, Any]) -> tuple[float, float]:
    """Compute (output_tok_s, prompt_tok_s) from a single Ollama response.

    Ollama reports durations in nanoseconds.
    """
    eval_count = body.get("eval_count") or 0
    eval_duration_ns = body.get("eval_duration") or 0
    prompt_eval_count = body.get("prompt_eval_count") or 0
    prompt_eval_duration_ns = body.get("prompt_eval_duration") or 0

    output_tok_s = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns else 0.0
    prompt_tok_s = (
        (prompt_eval_count / (prompt_eval_duration_ns / 1e9))
        if prompt_eval_duration_ns
        else 0.0
    )
    return output_tok_s, prompt_tok_s


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a local Ollama model.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID, e.g. ollama-local/qwen3.5:9b",
    )
    parser.add_argument(
        "--class",
        dest="hw_class",
        required=True,
        help="Hardware class key for this host (see think/benchmark/reference.json)",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull the model via Ollama API if not already installed.",
    )
    args = parser.parse_args()

    ensure_installed(args.model, allow_pull=args.pull)

    print(
        f"Benchmarking {args.model} on class '{args.hw_class}' "
        f"({_WARMUP_RUNS} warmup + {_MEASURE_RUNS} measured runs)…",
        file=sys.stderr,
    )

    for i in range(_WARMUP_RUNS):
        print(f"  warmup {i + 1}/{_WARMUP_RUNS}…", file=sys.stderr)
        run_once(args.model)

    output_rates: list[float] = []
    prompt_rates: list[float] = []
    for i in range(_MEASURE_RUNS):
        print(f"  run {i + 1}/{_MEASURE_RUNS}…", file=sys.stderr)
        body = run_once(args.model)
        out_rate, prompt_rate = tok_s_from_response(body)
        output_rates.append(out_rate)
        prompt_rates.append(prompt_rate)
        print(
            f"    output {out_rate:.1f} tok/s, prompt {prompt_rate:.1f} tok/s",
            file=sys.stderr,
        )

    median_output = sorted(output_rates)[len(output_rates) // 2]
    median_prompt = sorted(prompt_rates)[len(prompt_rates) // 2]

    snippet = {
        args.hw_class: {
            "output_tok_s": round(median_output, 1),
            "prompt_tok_s": round(median_prompt, 1),
            "measured_at": date.today().isoformat(),
        }
    }
    print(
        f"\n# Paste into think/benchmark/models.json "
        f"-> models['{args.model}'].benchmarks:",
        file=sys.stderr,
    )
    print(json.dumps(snippet, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
