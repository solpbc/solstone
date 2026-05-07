# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
from pathlib import Path

from solstone.think.prompts import load_prompt


def _seed_config(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "journal.json").write_text(
        json.dumps(
            {
                "identity": {"name": "Test User", "preferred": "Test User"},
                "agent": {"name": "sol", "name_status": "default", "named_date": None},
            }
        ),
        encoding="utf-8",
    )


def test_skill_editor_prompt_substitutes_template_vars(monkeypatch, tmp_path):
    _seed_config(tmp_path)
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    repo_root = Path(__file__).resolve().parent.parent

    prompt = load_prompt(
        "skill_editor",
        base_dir=repo_root / "solstone" / "apps" / "skills" / "talent",
        context={
            "day": "Saturday, April 19, 2026",
            "skill_mode_instruction": "Create the profile for this skill.",
            "skill_context": "## Metadata\nName: Alpha Skill",
            "existing_profile": "",
            "owner_instructions": "Refine wording",
            "slug": "alpha-skill",
        },
    )

    assert "Create the profile for this skill." in prompt.text
    assert "## Metadata" in prompt.text
    assert "alpha-skill" in prompt.text
    assert "$skill_context" not in prompt.text
    assert "$owner_instructions" not in prompt.text
