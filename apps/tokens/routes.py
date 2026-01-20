# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Token usage and cost tracking app."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict

from flask import Blueprint, jsonify, render_template, request

from convey import state
from convey.utils import DATE_RE
from muse.models import calc_token_cost, get_model_provider, iter_token_log

tokens_bp = Blueprint(
    "app:tokens",
    __name__,
    url_prefix="/app/tokens",
)


def _parse_context_prefix(context: str) -> str:
    """Return full context string for grouping."""
    if not context:
        return "unknown"
    return context


def _aggregate_token_data(day: str) -> Dict[str, Any]:
    """
    Read and aggregate token usage data for a given day.

    Returns dict with daily summary and breakdowns by provider, model, token type, context, and segment.
    """
    # Accumulators
    by_provider: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "requests": 0,
            "tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "cost": 0.0,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "models": set(),
        }
    )

    by_model: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "requests": 0,
            "tokens": 0,
            "cost": 0.0,
            "provider": "",
        }
    )

    by_context: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "requests": 0,
            "tokens": 0,
            "cost": 0.0,
            "models": set(),
        }
    )

    by_segment: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "requests": 0,
            "tokens": 0,
            "cost": 0.0,
            "models": set(),
        }
    )

    # Token type totals
    token_types = {
        "input": {"tokens": 0, "cost": 0.0},
        "output": {"tokens": 0, "cost": 0.0},
        "cached": {"tokens": 0, "cost": 0.0},
        "reasoning": {"tokens": 0, "cost": 0.0},
    }

    total_requests = 0
    total_tokens = 0
    total_cost = 0.0

    for entry in iter_token_log(day):
        # Extract fields
        model = entry.get("model", "unknown")
        usage = entry.get("usage", {})
        context = entry.get("context", "unknown")

        # Get token counts (handle both missing keys and explicit None values)
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        cached_tokens = usage.get("cached_tokens", 0) or 0
        reasoning_tokens = usage.get("reasoning_tokens", 0) or 0
        total_entry_tokens = usage.get("total_tokens", 0) or 0

        # Calculate cost
        cost_data = calc_token_cost(entry)
        if cost_data:
            entry_cost = cost_data["total_cost"]
            entry_input_cost = cost_data["input_cost"]
            entry_output_cost = cost_data["output_cost"]
        else:
            entry_cost = 0.0
            entry_input_cost = 0.0
            entry_output_cost = 0.0

        # Get provider
        provider = get_model_provider(model)
        if provider == "unknown":
            continue  # Skip unknown providers

        # Update totals
        total_requests += 1
        total_tokens += total_entry_tokens
        total_cost += entry_cost

        # Update provider breakdown
        by_provider[provider]["requests"] += 1
        by_provider[provider]["tokens"] += total_entry_tokens
        by_provider[provider]["input_tokens"] += input_tokens
        by_provider[provider]["output_tokens"] += output_tokens
        by_provider[provider]["cached_tokens"] += cached_tokens
        by_provider[provider]["reasoning_tokens"] += reasoning_tokens
        by_provider[provider]["cost"] += entry_cost
        by_provider[provider]["input_cost"] += entry_input_cost
        by_provider[provider]["output_cost"] += entry_output_cost
        by_provider[provider]["models"].add(model)

        # Update model breakdown
        by_model[model]["requests"] += 1
        by_model[model]["tokens"] += total_entry_tokens
        by_model[model]["cost"] += entry_cost
        by_model[model]["provider"] = provider

        # Update context breakdown
        context_prefix = _parse_context_prefix(context)
        by_context[context_prefix]["requests"] += 1
        by_context[context_prefix]["tokens"] += total_entry_tokens
        by_context[context_prefix]["cost"] += entry_cost
        by_context[context_prefix]["models"].add(model)

        # Update segment breakdown
        segment = entry.get("segment") or "[unattributed]"
        by_segment[segment]["requests"] += 1
        by_segment[segment]["tokens"] += total_entry_tokens
        by_segment[segment]["cost"] += entry_cost
        by_segment[segment]["models"].add(model)

        # Update token type breakdown
        token_types["input"]["tokens"] += input_tokens
        token_types["input"]["cost"] += entry_input_cost
        # Output tokens include reasoning (they're billed together)
        token_types["output"]["tokens"] += output_tokens + reasoning_tokens
        token_types["output"]["cost"] += entry_output_cost
        token_types["cached"]["tokens"] += cached_tokens
        # Track reasoning separately for display annotation
        token_types["reasoning"]["tokens"] += reasoning_tokens

    # Convert to lists and sort
    provider_list = [
        {
            "provider": prov,
            "requests": data["requests"],
            "tokens": data["tokens"],
            "input_tokens": data["input_tokens"],
            "output_tokens": data["output_tokens"],
            "cached_tokens": data["cached_tokens"],
            "reasoning_tokens": data["reasoning_tokens"],
            "cost": round(data["cost"], 6),
            "input_cost": round(data["input_cost"], 6),
            "output_cost": round(data["output_cost"], 6),
            "models_used": sorted(list(data["models"])),
            "percent": round(
                (data["cost"] / total_cost * 100) if total_cost > 0 else 0, 1
            ),
        }
        for prov, data in by_provider.items()
    ]
    provider_list.sort(key=lambda x: x["cost"], reverse=True)

    model_list = [
        {
            "model": mdl,
            "provider": data["provider"],
            "requests": data["requests"],
            "tokens": data["tokens"],
            "cost": round(data["cost"], 6),
            "avg_cost_per_request": round(
                data["cost"] / data["requests"] if data["requests"] > 0 else 0, 6
            ),
            "percent": round(
                (data["cost"] / total_cost * 100) if total_cost > 0 else 0, 1
            ),
        }
        for mdl, data in by_model.items()
    ]
    model_list.sort(key=lambda x: x["cost"], reverse=True)

    context_list = [
        {
            "context": ctx,
            "requests": data["requests"],
            "tokens": data["tokens"],
            "cost": round(data["cost"], 6),
            "models_used": sorted(list(data["models"])),
            "percent": round(
                (data["cost"] / total_cost * 100) if total_cost > 0 else 0, 1
            ),
        }
        for ctx, data in by_context.items()
    ]
    context_list.sort(key=lambda x: x["cost"], reverse=True)

    # Build segment list (exclude [unattributed] from avg calculation)
    segment_list = [
        {
            "segment": seg,
            "requests": data["requests"],
            "tokens": data["tokens"],
            "cost": round(data["cost"], 6),
            "models_used": sorted(list(data["models"])),
            "percent": round(
                (data["cost"] / total_cost * 100) if total_cost > 0 else 0, 1
            ),
        }
        for seg, data in by_segment.items()
    ]
    segment_list.sort(key=lambda x: x["cost"], reverse=True)

    # Calculate segment average (excluding unattributed)
    attributed_segments = [s for s in segment_list if s["segment"] != "[unattributed]"]
    segment_count = len(attributed_segments)
    segment_total_cost = sum(s["cost"] for s in attributed_segments)
    segment_avg_cost = (
        round(segment_total_cost / segment_count, 6) if segment_count > 0 else 0.0
    )

    # Calculate cached/reasoning percentages for display annotations
    # - cached_tokens are a subset of input_tokens (reduce cost)
    # - reasoning_tokens are part of output_tokens (billed as output)
    input_tokens_total = token_types["input"]["tokens"]
    output_tokens_total = token_types["output"]["tokens"]  # Already includes reasoning
    cached_tokens_total = token_types["cached"]["tokens"]
    reasoning_tokens_total = token_types["reasoning"]["tokens"]

    cached_pct = (
        round((cached_tokens_total / input_tokens_total * 100), 1)
        if input_tokens_total > 0
        else 0.0
    )
    reasoning_pct = (
        round((reasoning_tokens_total / output_tokens_total * 100), 1)
        if output_tokens_total > 0
        else 0.0
    )

    token_types["input"]["cached_pct"] = cached_pct
    token_types["output"]["reasoning_pct"] = reasoning_pct

    # Calculate average rates for token types (only for input/output, not cached/reasoning)
    for ttype_name in ["input", "output"]:
        ttype = token_types[ttype_name]
        if ttype["tokens"] > 0:
            ttype["avg_rate"] = round(
                ttype["cost"] / ttype["tokens"] * 1000, 4
            )  # per 1K tokens
        else:
            ttype["avg_rate"] = 0.0
        ttype["percent"] = round(
            (ttype["cost"] / total_cost * 100) if total_cost > 0 else 0, 1
        )
        ttype["cost"] = round(ttype["cost"], 6)

    return {
        "day": day,
        "total": {
            "requests": total_requests,
            "tokens": total_tokens,
            "cost": round(total_cost, 6),
            "segment_avg_cost": segment_avg_cost,
        },
        "by_provider": provider_list,
        "by_model": model_list,
        "by_token_type": token_types,
        "by_context": context_list,
        "by_segment": segment_list,
    }


@tokens_bp.route("/")
def index():
    """Redirect to today's token usage."""
    from flask import redirect, url_for

    today = date.today().strftime("%Y%m%d")
    return redirect(url_for("app:tokens.day_view", day=today))


@tokens_bp.route("/<day>")
def day_view(day: str):
    """Token usage dashboard for specific day."""
    if not DATE_RE.fullmatch(day):
        return "", 404
    return render_template("app.html")


@tokens_bp.route("/api/usage")
def api_usage():
    """API endpoint to get token usage data for a day."""
    day = request.args.get("day", date.today().strftime("%Y%m%d"))

    if not DATE_RE.fullmatch(day):
        return jsonify({"error": "Invalid day format"}), 400

    data = _aggregate_token_data(day)
    return jsonify(data)


@tokens_bp.route("/api/stats/<month>")
def api_stats(month: str):
    """Return token cost for each day in a specific month.

    Args:
        month: YYYYMM format month string

    Returns:
        JSON dict mapping day (YYYYMMDD) to cost in dollars.
        Tokens app is not facet-aware, so returns simple {day: cost} mapping.
    """
    import re

    if not re.fullmatch(r"\d{6}", month):
        return jsonify({"error": "Invalid month format, expected YYYYMM"}), 400

    tokens_dir = Path(state.journal_root) / "tokens"
    if not tokens_dir.exists():
        return jsonify({})

    stats: dict[str, float] = {}

    for log_file in tokens_dir.glob(f"{month}*.jsonl"):
        day = log_file.stem
        if not DATE_RE.fullmatch(day):
            continue

        # Get aggregated data for this day
        data = _aggregate_token_data(day)
        cost = data["total"]["cost"]
        if cost > 0:
            stats[day] = cost

    return jsonify(stats)
