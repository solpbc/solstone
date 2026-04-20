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
import base64
import io
import json
import sys
from datetime import date
from typing import Any

# Fixed text prompt used for text-mode runs. ~150-200 tokens of input,
# asks for ~200 tokens of output.
_TEXT_PROMPT = (
    "You are a benchmarking fixture. Write a focused, concrete 200-word "
    "technical paragraph explaining the tradeoffs between dense and "
    "sparse-mixture-of-experts transformer architectures for on-device "
    "inference. Discuss parameter count, active-parameter count, memory "
    "bandwidth, and typical quantization strategies. Do not include "
    "headings, lists, or citations — a single paragraph of prose."
)

# Prompt used in vision mode alongside the canned image. Asks for enough
# output to make the output-tok/s number stable.
_VISION_PROMPT = (
    "Describe this image in a focused 200-word paragraph. Cover the "
    "geometric shapes, colors, layout, and any text visible. Do not "
    "include headings, lists, or citations — a single paragraph of prose."
)

_MAX_OUTPUT_TOKENS = 256
_WARMUP_RUNS = 1
_MEASURE_RUNS = 3

# Cap context window to keep the compute graph tractable. Ollama otherwise
# defaults to a very large context (256K for recent Qwen builds), which
# inflates the KV cache + compute graph enough to OOM big models on
# unified-memory systems. 8K is plenty for the fixed benchmark prompt +
# image tokens + 256-token completion.
_BENCHMARK_NUM_CTX = 8192


def _build_canned_image_b64() -> str:
    """Generate a deterministic 512x512 JPEG and return it as base64.

    Uses PIL to draw a mix of shapes, a gradient, and text so the
    vision encoder has non-trivial content to process. Same image every
    run so prompt-eval numbers are comparable.
    """
    from PIL import Image, ImageDraw, ImageFont

    size = 512
    img = Image.new("RGB", (size, size), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Gradient band
    for y in range(0, 128):
        shade = int(80 + (y / 128) * 120)
        draw.line([(0, y), (size, y)], fill=(shade, shade // 2, 200 - shade // 2))

    # Geometric shapes
    draw.rectangle([40, 180, 200, 340], fill=(200, 60, 60), outline=(0, 0, 0), width=3)
    draw.ellipse([280, 180, 460, 360], fill=(60, 140, 200), outline=(0, 0, 0), width=3)
    draw.polygon(
        [(256, 380), (400, 480), (112, 480)],
        fill=(80, 180, 90),
        outline=(0, 0, 0),
    )

    # Text so the encoder sees character content
    try:
        font = ImageFont.load_default()
        draw.text((20, 20), "solstone benchmark fixture", fill=(20, 20, 20), font=font)
        draw.text(
            (20, 490), "shapes: square, circle, triangle", fill=(20, 20, 20), font=font
        )
    except OSError:
        # If default font missing in some minimal env, skip text.
        pass

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_once(model: str, *, vision: bool = False) -> dict[str, Any]:
    """Send one completion request; return the raw Ollama response body.

    When ``vision=True``, include a canned image in the user message so
    the prompt-eval count captures image-encoder cost.
    """
    from think.providers.ollama import _get_client, _strip_model_prefix

    bare_model = _strip_model_prefix(model)
    client = _get_client()

    if vision:
        message: dict[str, Any] = {
            "role": "user",
            "content": _VISION_PROMPT,
            "images": [_build_canned_image_b64()],
        }
    else:
        message = {"role": "user", "content": _TEXT_PROMPT}

    response = client.post(
        "/api/chat",
        json={
            "model": bare_model,
            "messages": [message],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.2,
                "num_predict": _MAX_OUTPUT_TOKENS,
                "num_ctx": _BENCHMARK_NUM_CTX,
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
    parser.add_argument(
        "--vision",
        action="store_true",
        help=(
            "Vision mode: include a canned image in the prompt. Use for VLMs "
            "so prompt-eval captures image-encoder cost."
        ),
    )
    args = parser.parse_args()

    ensure_installed(args.model, allow_pull=args.pull)

    mode = "vision" if args.vision else "text_only"
    print(
        f"Benchmarking {args.model} on class '{args.hw_class}' in {mode} mode "
        f"({_WARMUP_RUNS} warmup + {_MEASURE_RUNS} measured runs)…",
        file=sys.stderr,
    )

    for i in range(_WARMUP_RUNS):
        print(f"  warmup {i + 1}/{_WARMUP_RUNS}…", file=sys.stderr)
        run_once(args.model, vision=args.vision)

    output_rates: list[float] = []
    prompt_rates: list[float] = []
    for i in range(_MEASURE_RUNS):
        print(f"  run {i + 1}/{_MEASURE_RUNS}…", file=sys.stderr)
        body = run_once(args.model, vision=args.vision)
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
            "mode": mode,
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
