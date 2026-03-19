# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Generate AGENTS.md from muse/unified.md using journal config values."""

from __future__ import annotations

import json
import os
from pathlib import Path
from string import Template
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "think" / "journal_default.json"
SOURCE_PATH = PROJECT_ROOT / "muse" / "unified.md"
OUTPUT_PATH = PROJECT_ROOT / "AGENTS.md"
GENERATED_HEADER = "<!-- generated from muse/unified.md — do not edit directly -->\n\n"


def _load_config() -> dict[str, Any]:
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    journal_root = os.getenv("_SOLSTONE_JOURNAL_OVERRIDE")
    if journal_root:
        journal_path = Path(journal_root)
        if not journal_path.is_absolute():
            journal_path = PROJECT_ROOT / journal_path
    else:
        journal_path = PROJECT_ROOT / "journal"

    config_path = journal_path / "config" / "journal.json"
    if not config_path.exists():
        return config

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_identity_to_template_vars(identity: dict[str, Any]) -> dict[str, str]:
    template_vars: dict[str, str] = {}

    for key, value in identity.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                var_name = f"{key}_{subkey}"
                template_vars[var_name] = str(subvalue)
                template_vars[var_name.capitalize()] = str(subvalue).capitalize()
        elif isinstance(value, (str, int, float, bool)):
            template_vars[key] = str(value)
            template_vars[key.capitalize()] = str(value).capitalize()

    return template_vars


def _apply_human_defaults(
    template_vars: dict[str, str], config: dict[str, Any]
) -> None:
    agent_name = str(config.get("agent", {}).get("name", "sol"))
    template_vars["agent_name"] = agent_name
    template_vars["Agent_name"] = agent_name.capitalize()

    resolved_name = template_vars.get("name", "") or "your journal owner"
    resolved_preferred = template_vars.get("preferred", "") or resolved_name

    template_vars["name"] = resolved_name
    template_vars["Name"] = resolved_name.capitalize()
    template_vars["preferred"] = resolved_preferred
    template_vars["Preferred"] = resolved_preferred.capitalize()

    pronoun_defaults = {
        "subject": "they",
        "object": "them",
        "possessive": "their",
        "reflexive": "themselves",
    }
    for key, value in pronoun_defaults.items():
        var_name = f"pronouns_{key}"
        resolved = template_vars.get(var_name, "") or value
        template_vars[var_name] = resolved
        template_vars[var_name.capitalize()] = resolved.capitalize()


def _load_template_body() -> str:
    with open(SOURCE_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip() == "}":
            return "".join(lines[i + 1 :])
    return "".join(lines)


def main() -> None:
    config = _load_config()
    identity = config.get("identity", {})
    template_vars = _flatten_identity_to_template_vars(identity)
    _apply_human_defaults(template_vars, config)

    content = _load_template_body()
    rendered = Template(content).safe_substitute(template_vars)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(GENERATED_HEADER)
        f.write(rendered)

    print("Generated AGENTS.md from muse/unified.md")


if __name__ == "__main__":
    main()
