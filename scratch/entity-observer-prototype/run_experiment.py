#!/usr/bin/env python3
"""Run entity_observer generate prototype experiments.

Tests different prompt strategies, output formats, and model tiers against
real pre-computed context from the journal. READ-ONLY on journal data;
writes results to scratch/entity-observer-prototype/results/.

Usage:
    cd /home/jer/projects/solstone
    python3 scratch/entity-observer-prototype/run_experiment.py \
        --facet solstone --day 20260414 \
        --prompt structured_json \
        --model gemini-2.5-flash-lite \
        --label lite-structured-json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("SOL_JOURNAL", str(Path(__file__).resolve().parents[2] / "journal"))

from assemble_context import assemble_full_context, format_prompt_context

RESULTS_DIR = Path(__file__).parent / "results"

# --- Prompt templates ---

SYSTEM_PROMPTS = {
    "observer_v1": """You are an entity observation agent for a personal knowledge system called Solstone.
Your task: extract durable factoids about entities from today's journal content.

An observation is a lasting fact about WHO or WHAT an entity IS — not what happened today.

Good observations:
- "Advocates for Socratic questioning in mentorship"
- "Based in Seattle, previously worked at Google"
- "Prefers async communication over meetings"

NOT observations (these are activity logs):
- "Discussed migration today"
- "Sent contract for review"
- "Uses v2.1.50" (expires)

Rules:
1. One fact per observation — no compound sentences
2. Must pass BOTH litmus tests:
   a) "Would this be true and useful 6 months from now?"
   b) "Would this help someone who's never met this entity?"
3. Check for semantic duplicates against existing observations
4. If existing observations are already rich, restraint is correct — zero new observations is valid
5. Skip entities where today's content reveals nothing durable""",

    "observer_v2_terse": """Entity observation agent. Extract durable factoids from today's journal content.

Observation = lasting fact about WHO/WHAT an entity IS. NOT activity logs, NOT ephemeral state.

Litmus: (1) true in 6 months? (2) useful to a stranger? Both must be yes.
One fact per observation. No duplicates of existing observations. Zero new is valid.""",
}

OUTPUT_FORMAT_INSTRUCTIONS = {
    "json_array": """Output format: JSON array of observation objects.
```json
[
  {
    "entity_id": "entity_slug",
    "entity_name": "Full Name",
    "content": "The durable observation text",
    "reasoning": "Why this qualifies as a durable observation (1 sentence)"
  }
]
```
Output ONLY the JSON array. No markdown, no commentary.""",

    "jsonl": """Output format: one JSON object per line (JSONL), no wrapping array.
{"entity_id": "entity_slug", "entity_name": "Full Name", "content": "The observation", "reasoning": "Why"}
{"entity_id": "other_entity", "entity_name": "Other Name", "content": "Another observation", "reasoning": "Why"}

Output ONLY the JSONL lines. No markdown, no commentary, no blank lines between entries.""",

    "markdown_structured": """Output format: Markdown with one section per entity that has new observations.

## Entity Name (entity_id)
- **Observation:** The durable factoid
- **Reasoning:** Why this is durable (1 sentence)

## Another Entity (another_id)
- **Observation:** Another factoid
- **Reasoning:** Why

Skip entities with no new observations entirely. End with a summary line:
"Observed X entities, Y new observations total."
""",

    "json_grouped": """Output format: JSON object grouped by entity_id.
```json
{
  "observations": {
    "entity_slug": [
      {"content": "The observation", "reasoning": "Why"}
    ],
    "other_entity": [
      {"content": "Another observation", "reasoning": "Why"}
    ]
  },
  "skipped": ["entity_ids_with_no_new_observations"],
  "summary": "Observed X entities, Y new observations total."
}
```
Output ONLY the JSON. No markdown wrapping.""",
}


def build_prompt(
    context: dict,
    *,
    system_key: str = "observer_v1",
    format_key: str = "json_array",
    context_style: str = "structured",
) -> tuple[str, str]:
    """Build system + user prompt for an experiment.

    Returns (system_prompt, user_prompt).
    """
    system = SYSTEM_PROMPTS[system_key]
    format_inst = OUTPUT_FORMAT_INSTRUCTIONS[format_key]

    # Build user prompt with pre-computed context
    context_text = format_prompt_context(context, style=context_style)

    user_prompt = f"""{format_inst}

---

{context_text}"""

    return system, user_prompt


def call_gemini(
    system_prompt: str,
    user_prompt: str,
    model: str,
    *,
    max_output_tokens: int = 8192,
    temperature: float = 0.3,
    thinking_budget: int | None = None,
) -> dict:
    """Call Gemini API and return result dict."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Try loading from vault
        vault_path = Path(__file__).resolve().parents[2].parent / "extro" / "cso" / "vault" / "api-keys" / "google-ai-studio.json"
        if vault_path.exists():
            vault_data = json.loads(vault_path.read_text())
            api_key = vault_data.get("api_key", "")
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key

    if not api_key:
        raise RuntimeError("No GOOGLE_API_KEY found")

    client = genai.Client(api_key=api_key)

    config_kwargs = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
    }
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        **config_kwargs,
    )

    start = time.time()
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=config,
    )
    elapsed = time.time() - start

    # Extract usage
    usage = {}
    if response.usage_metadata:
        um = response.usage_metadata
        usage = {
            "prompt_tokens": getattr(um, "prompt_token_count", 0),
            "output_tokens": getattr(um, "candidates_token_count", 0),
            "total_tokens": getattr(um, "total_token_count", 0),
            "thinking_tokens": getattr(um, "thoughts_token_count", 0),
        }

    return {
        "text": response.text or "",
        "usage": usage,
        "elapsed_seconds": round(elapsed, 2),
        "model": model,
        "finish_reason": str(getattr(response.candidates[0], "finish_reason", "")) if response.candidates else "",
    }


def parse_output(text: str, format_key: str) -> dict:
    """Attempt to parse the model output and validate structure."""
    result = {
        "raw_text": text,
        "parse_success": False,
        "observation_count": 0,
        "entity_count": 0,
        "observations": [],
        "errors": [],
    }

    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (fences)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        if format_key == "json_array":
            observations = json.loads(cleaned)
            if isinstance(observations, list):
                result["parse_success"] = True
                result["observations"] = observations
                result["observation_count"] = len(observations)
                result["entity_count"] = len(set(o.get("entity_id", "") for o in observations))

        elif format_key == "jsonl":
            observations = []
            parse_errors = 0
            for line in cleaned.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    observations.append(json.loads(line))
                except json.JSONDecodeError:
                    parse_errors += 1
            result["parse_success"] = len(observations) > 0
            result["observations"] = observations
            result["observation_count"] = len(observations)
            result["entity_count"] = len(set(o.get("entity_id", "") for o in observations))
            if parse_errors:
                result["errors"].append(f"{parse_errors} lines failed to parse (likely truncation)")

        elif format_key == "json_grouped":
            data = json.loads(cleaned)
            if isinstance(data, dict) and "observations" in data:
                result["parse_success"] = True
                all_obs = []
                for entity_id, obs_list in data["observations"].items():
                    for obs in obs_list:
                        all_obs.append({"entity_id": entity_id, **obs})
                result["observations"] = all_obs
                result["observation_count"] = len(all_obs)
                result["entity_count"] = len(data["observations"])
                result["skipped_count"] = len(data.get("skipped", []))

        elif format_key == "markdown_structured":
            # Count ## headers and **Observation:** lines
            import re
            entities = re.findall(r"^## (.+?)(?:\s*\(|$)", cleaned, re.MULTILINE)
            observations = re.findall(r"\*\*Observation:\*\*\s*(.+)", cleaned)
            result["parse_success"] = len(observations) > 0 or "0 new observations" in cleaned.lower()
            result["observation_count"] = len(observations)
            result["entity_count"] = len(entities)
            result["observations"] = [
                {"content": o, "entity_name": entities[i] if i < len(entities) else "?"}
                for i, o in enumerate(observations)
            ]

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        result["errors"].append(str(e))

    return result


def evaluate_observations(parsed: dict, context: dict) -> dict:
    """Evaluate observation quality against the pre-computed context."""
    eval_result = {
        "total_observations": parsed["observation_count"],
        "entities_with_observations": parsed["entity_count"],
        "duplicates": 0,
        "quality_flags": [],
    }

    # Build existing observation index
    existing_obs = {}
    for ec in context["entity_contexts"]:
        existing_obs[ec["id"]] = set(
            o.lower().strip() for o in ec.get("observations", ec.get("recent_observations", []))
        )

    # Check each observation
    for obs in parsed.get("observations", []):
        entity_id = obs.get("entity_id", "")
        content = obs.get("content", "")
        content_lower = content.lower().strip()

        # Check for exact duplicates
        if entity_id in existing_obs:
            for existing in existing_obs[entity_id]:
                if content_lower == existing or content_lower in existing or existing in content_lower:
                    eval_result["duplicates"] += 1
                    eval_result["quality_flags"].append(
                        f"DUPLICATE: {entity_id}: '{content[:60]}...'"
                    )
                    break

        # Check for temporal language (not durable)
        temporal_markers = ["today", "currently", "as of", "this week", "yesterday", "right now"]
        for marker in temporal_markers:
            if marker in content_lower:
                eval_result["quality_flags"].append(
                    f"TEMPORAL: {entity_id}: '{content[:60]}...' (contains '{marker}')"
                )
                break

        # Check for activity-log patterns
        activity_markers = ["discussed", "sent", "reviewed", "filed", "submitted", "scheduled"]
        for marker in activity_markers:
            if content_lower.startswith(marker):
                eval_result["quality_flags"].append(
                    f"ACTIVITY_LOG: {entity_id}: '{content[:60]}...' (starts with '{marker}')"
                )
                break

    eval_result["quality_score"] = max(0, 1.0 - (
        eval_result["duplicates"] * 0.15 +
        len(eval_result["quality_flags"]) * 0.05
    ))

    return eval_result


def run_experiment(
    facet: str,
    day: str,
    *,
    strategy: str = "focused",
    context_style: str = "structured",
    system_key: str = "observer_v1",
    format_key: str = "json_array",
    model: str = "gemini-2.5-flash-lite",
    thinking_budget: int | None = None,
    label: str = "",
) -> dict:
    """Run a single experiment and return full results."""
    print(f"\n{'='*60}")
    print(f"Experiment: {label or 'unnamed'}")
    print(f"  Model: {model}")
    print(f"  Strategy: {strategy}, Style: {context_style}")
    print(f"  System: {system_key}, Format: {format_key}")
    if thinking_budget:
        print(f"  Thinking budget: {thinking_budget}")
    print(f"{'='*60}")

    # Assemble context
    print("Assembling context...")
    context = assemble_full_context(facet, day, strategy=strategy)
    print(f"  Active entities: {context['active_count']}, Est tokens: {context['estimated_tokens']:,}")

    # Build prompt
    system_prompt, user_prompt = build_prompt(
        context,
        system_key=system_key,
        format_key=format_key,
        context_style=context_style,
    )
    print(f"  System prompt: {len(system_prompt):,} chars")
    print(f"  User prompt: {len(user_prompt):,} chars")

    # Call model
    print(f"Calling {model}...")
    gen_result = call_gemini(
        system_prompt,
        user_prompt,
        model,
        thinking_budget=thinking_budget,
    )
    print(f"  Elapsed: {gen_result['elapsed_seconds']}s")
    print(f"  Usage: {gen_result['usage']}")
    print(f"  Output length: {len(gen_result['text']):,} chars")

    # Parse output
    print("Parsing output...")
    parsed = parse_output(gen_result["text"], format_key)
    print(f"  Parse success: {parsed['parse_success']}")
    print(f"  Observations: {parsed['observation_count']}")
    print(f"  Entities: {parsed['entity_count']}")
    if parsed["errors"]:
        print(f"  Errors: {parsed['errors']}")

    # Evaluate quality
    print("Evaluating quality...")
    evaluation = evaluate_observations(parsed, context)
    print(f"  Duplicates: {evaluation['duplicates']}")
    print(f"  Quality flags: {len(evaluation['quality_flags'])}")
    print(f"  Quality score: {evaluation['quality_score']:.2f}")
    if evaluation["quality_flags"][:5]:
        for flag in evaluation["quality_flags"][:5]:
            print(f"    - {flag}")

    # Compile result
    result = {
        "label": label,
        "config": {
            "facet": facet,
            "day": day,
            "strategy": strategy,
            "context_style": context_style,
            "system_key": system_key,
            "format_key": format_key,
            "model": model,
            "thinking_budget": thinking_budget,
        },
        "context_stats": {
            "total_attached": context["total_attached"],
            "active_count": context["active_count"],
            "estimated_tokens": context["estimated_tokens"],
            "system_prompt_chars": len(system_prompt),
            "user_prompt_chars": len(user_prompt),
        },
        "generation": {
            "elapsed_seconds": gen_result["elapsed_seconds"],
            "usage": gen_result["usage"],
            "output_chars": len(gen_result["text"]),
            "finish_reason": gen_result["finish_reason"],
        },
        "parsing": {
            "parse_success": parsed["parse_success"],
            "observation_count": parsed["observation_count"],
            "entity_count": parsed["entity_count"],
            "errors": parsed["errors"],
        },
        "evaluation": evaluation,
        "raw_output": gen_result["text"],
        "observations": parsed["observations"],
    }

    # Save result
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_file = RESULTS_DIR / f"{label or 'unnamed'}.json"
    result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nResult saved to {result_file}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--facet", default="solstone")
    parser.add_argument("--day", default="20260414")
    parser.add_argument("--strategy", default="focused")
    parser.add_argument("--context-style", default="structured")
    parser.add_argument("--system", default="observer_v1")
    parser.add_argument("--format", default="json_array")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    run_experiment(
        args.facet,
        args.day,
        strategy=args.strategy,
        context_style=args.context_style,
        system_key=args.system,
        format_key=args.format,
        model=args.model,
        thinking_budget=args.thinking_budget,
        label=args.label,
    )
