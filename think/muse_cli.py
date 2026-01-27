# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI for inspecting muse prompt configurations.

Lists all system and app prompts with their frontmatter metadata,
supports filtering by schedule and source, and provides detail views.

Usage:
    sol muse                     List all prompts grouped by schedule
    sol muse <name>              Show details for a specific prompt
    sol muse <name> --json       Output a single prompt as JSONL
    sol muse --json              Output all configs as JSONL
    sol muse --schedule daily    Filter by schedule type
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import frontmatter

from think.utils import (
    MUSE_DIR,
    _load_prompt_metadata,
    get_muse_configs,
    setup_cli,
)

# Project root for computing relative paths
_PROJECT_ROOT = Path(__file__).parent.parent

# Keys injected by get_agent() or internal bookkeeping — not frontmatter
_SKIP_KEYS = frozenset(
    {
        "path",
        "mtime",
        "hook_path",
        "system_instruction",
        "user_instruction",
        "extra_context",
        "system_prompt_name",
        "sources",
    }
)


def _relative_path(abs_path: str) -> str:
    """Convert absolute path to project-relative path."""
    try:
        return str(Path(abs_path).relative_to(_PROJECT_ROOT))
    except ValueError:
        return abs_path


def _resolve_md_path(name: str) -> Path:
    """Resolve a prompt name to its .md file path."""
    if ":" in name:
        app, agent_name = name.split(":", 1)
        return _PROJECT_ROOT / "apps" / app / "muse" / f"{agent_name}.md"
    return MUSE_DIR / f"{name}.md"


def _resolve_file_path(key: str, info: dict[str, Any]) -> str:
    """Resolve the relative file path for a config entry."""
    if info.get("path"):
        return _relative_path(str(info["path"]))
    # Configs loaded via get_agent() lose their path — reconstruct from key
    return _relative_path(str(_resolve_md_path(key)))


def _scan_variables(body: str) -> list[str]:
    """Scan prompt body text for $template variables."""
    # Match $word or ${word} but not $$ (escaped dollar signs)
    matches = re.findall(r"(?<!\$)\$\{?([a-zA-Z_]\w*)\}?", body)
    # Deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def _property_tags(info: dict[str, Any]) -> str:
    """Build compact property tags string for list view."""
    tags: list[str] = []

    if info.get("output"):
        tags.append(f"output:{info['output']}")

    if info.get("tools"):
        tools = info["tools"]
        if isinstance(tools, list):
            tools = ",".join(tools)
        tags.append(f"tools:{tools}")

    if info.get("hook"):
        tags.append(f"hook:{info['hook']}")

    if info.get("disabled"):
        tags.append("disabled")

    return "  ".join(tags)


def _collect_configs(
    *,
    schedule: str | None = None,
    source: str | None = None,
    include_disabled: bool = False,
) -> dict[str, dict[str, Any]]:
    """Collect all muse configs with optional filters applied."""
    configs = get_muse_configs(schedule=schedule, include_disabled=True)

    filtered: dict[str, dict[str, Any]] = {}
    for key, info in configs.items():
        if not include_disabled and info.get("disabled", False):
            continue
        if source and info.get("source") != source:
            continue
        filtered[key] = info

    return filtered


def _to_jsonl_record(key: str, info: dict[str, Any]) -> dict[str, Any]:
    """Build a clean JSONL record from a config entry."""
    record: dict[str, Any] = {"file": _resolve_file_path(key, info)}
    for k, v in info.items():
        if k not in _SKIP_KEYS:
            record[k] = v
    return record


def list_prompts(
    *,
    schedule: str | None = None,
    source: str | None = None,
    include_disabled: bool = False,
) -> None:
    """Print prompts grouped by schedule."""
    all_configs = get_muse_configs(include_disabled=True)
    configs = _collect_configs(
        schedule=schedule, source=source, include_disabled=include_disabled
    )

    if not configs:
        print("No prompts found matching filters.")
        return

    # Group by schedule
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "segment": [],
        "daily": [],
        "unscheduled": [],
    }

    for key, info in sorted(configs.items()):
        sched = info.get("schedule")
        if sched in ("segment", "daily"):
            groups[sched].append((key, info))
        else:
            groups["unscheduled"].append((key, info))

    # Compute column width from longest name
    all_names = list(configs.keys())
    name_width = max(len(n) for n in all_names) if all_names else 20
    name_width = max(name_width, 10)

    # Print each non-empty group
    for group_name in ("segment", "daily", "unscheduled"):
        items = groups[group_name]
        if not items:
            continue

        # Skip group header if filtering to a single schedule
        if not schedule:
            print(f"{group_name}:")

        for key, info in items:
            title = info.get("title", "")
            tags = _property_tags(info)
            src = ""
            if info.get("source") == "app":
                src = f"  [{info.get('app', 'app')}]"

            line = f"  {key:<{name_width}}  {title:<32}  {tags}{src}"
            print(line.rstrip())

        if not schedule:
            print()

    # Show disabled count hint
    if not include_disabled:
        disabled_count = sum(
            1 for info in all_configs.values() if info.get("disabled", False)
        )
        if disabled_count:
            total = len(configs)
            print(f"{total} prompts ({disabled_count} disabled hidden, use --disabled)")


def show_prompt(name: str, *, as_json: bool = False) -> None:
    """Print detailed info for a single prompt."""
    md_path = _resolve_md_path(name)

    if not md_path.exists():
        print(f"Prompt not found: {name}", file=sys.stderr)
        print(f"  looked at: {_relative_path(str(md_path))}", file=sys.stderr)
        sys.exit(1)

    info = _load_prompt_metadata(md_path)
    rel_path = _relative_path(str(md_path))

    # Load body once for variables and line count
    try:
        post = frontmatter.load(md_path)
        body = post.content.strip()
    except Exception:
        body = None

    if as_json:
        record = _to_jsonl_record(name, info)
        print(json.dumps(record, default=str))
        return

    print(f"\n{rel_path}\n")

    # Display frontmatter fields
    # Order: title, description, then alphabetical for the rest
    priority_keys = [
        "title",
        "description",
        "schedule",
        "output",
        "tools",
        "hook",
        "color",
    ]
    skip_keys = {"path", "mtime", "hook_path"}

    label_width = 14

    def print_field(key: str, value: Any) -> None:
        if key in skip_keys:
            return
        val_str = str(value)
        # Truncate long descriptions for readability
        if key == "description" and len(val_str) > 72:
            val_str = val_str[:72] + "..."
        # Show hook path inline
        if key == "hook" and info.get("hook_path"):
            val_str += f" \u2192 {_relative_path(str(info['hook_path']))}"
        print(f"  {key + ':':<{label_width}} {val_str}")

    printed: set[str] = set()
    for key in priority_keys:
        if key in info and key not in skip_keys:
            print_field(key, info[key])
            printed.add(key)

    # Remaining fields alphabetically
    for key in sorted(info.keys()):
        if key not in printed and key not in skip_keys:
            print_field(key, info[key])

    # Template variables and body line count from single parse
    if body is not None:
        variables = _scan_variables(body)
        if variables:
            vars_str = ", ".join(f"${v}" for v in variables)
            print(f"  {'variables:':<{label_width}} {vars_str}")

        line_count = len(body.splitlines())
        print(f"  {'body:':<{label_width}} {line_count} lines")

    print()


def json_output(
    *,
    schedule: str | None = None,
    source: str | None = None,
    include_disabled: bool = False,
) -> None:
    """Print JSONL output with one config per line, including filename."""
    configs = _collect_configs(
        schedule=schedule, source=source, include_disabled=include_disabled
    )

    for key, info in sorted(configs.items()):
        print(json.dumps(_to_jsonl_record(key, info), default=str))


def main() -> None:
    """Entry point for sol muse."""
    parser = argparse.ArgumentParser(description="Inspect muse prompt configurations")
    parser.add_argument("name", nargs="?", help="Show details for a specific prompt")
    parser.add_argument(
        "--schedule",
        choices=["daily", "segment"],
        help="Filter by schedule type",
    )
    parser.add_argument(
        "--source",
        choices=["system", "app"],
        help="Filter by origin",
    )
    parser.add_argument(
        "--disabled",
        action="store_true",
        help="Include disabled prompts",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSONL (one JSON object per line)",
    )

    args = setup_cli(parser)

    if args.name:
        show_prompt(args.name, as_json=args.json)
    elif args.json:
        json_output(
            schedule=args.schedule,
            source=args.source,
            include_disabled=args.disabled,
        )
    else:
        list_prompts(
            schedule=args.schedule,
            source=args.source,
            include_disabled=args.disabled,
        )


if __name__ == "__main__":
    main()
