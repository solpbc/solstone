# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from pathlib import Path

from .prompts import load_prompt


def _load_prompt() -> str:
    """Return system instruction text for planning."""
    prompt_content = load_prompt("planner", base_dir=Path(__file__).parent)
    return prompt_content.text


def generate_plan(request: str) -> str:
    """Return a detailed agent plan for ``request`` using configured provider."""
    from solstone.think.models import generate

    return generate(
        contents=request,
        context="planner.generate",
        temperature=0.3,
        max_output_tokens=4096,
        thinking_budget=4096,
        system_instruction=_load_prompt(),
    )
