# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI for inspecting muse prompt configurations.

Lists all system and app prompts with their frontmatter metadata,
supports filtering by schedule and source, and provides detail views.

Usage:
    sol muse                     List all prompts grouped by schedule
    sol muse <name>              Show details for a specific prompt
    sol muse <name> --json       Output a single prompt as JSONL
    sol muse <name> --prompt     Show full prompt context (dry-run)
    sol muse --json              Output all configs as JSONL
    sol muse --schedule daily    Filter by schedule type
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import frontmatter

from think.utils import (
    MUSE_DIR,
    _load_prompt_metadata,
    get_muse_configs,
    get_output_topic,
    setup_cli,
)

# Project root for computing relative paths
_PROJECT_ROOT = Path(__file__).parent.parent

# Internal bookkeeping keys to exclude from JSONL output
_INTERNAL_KEYS = frozenset({"path", "mtime"})


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


def _format_output_path(key: str, info: dict[str, Any]) -> str:
    """Compose output path pattern for a prompt.

    Returns path pattern like '<day>/agents/flow.md' or '<segment>/activity.md',
    or '-' if the prompt has no output field.
    """
    output_format = info.get("output")
    if not output_format:
        return "-"

    # Determine extension
    ext = "json" if output_format == "json" else "md"

    # Get topic name (handles app namespacing like entities:entities -> _entities_entities)
    topic = get_output_topic(key)

    # Determine path based on schedule
    schedule = info.get("schedule")
    if schedule == "segment":
        return f"<segment>/{topic}.{ext}"
    else:
        # daily and unscheduled both go to agents/
        return f"<day>/agents/{topic}.{ext}"


def _format_tools(info: dict[str, Any]) -> str:
    """Extract tools field or return '-' if none."""
    tools = info.get("tools")
    if not tools:
        return "-"
    if isinstance(tools, list):
        return ", ".join(tools)
    return tools


def _format_tags(info: dict[str, Any]) -> str:
    """Build compact tags string for hook and disabled status."""
    tags: list[str] = []

    if info.get("hook"):
        hook = info["hook"]
        if isinstance(hook, dict):
            # Format as "hook:pre=name,post=name"
            parts = []
            if hook.get("pre"):
                parts.append(f"pre={hook['pre']}")
            if hook.get("post"):
                parts.append(f"post={hook['post']}")
            tags.append(f"hook:{','.join(parts)}")
        else:
            tags.append(f"hook:{hook}")

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
    record: dict[str, Any] = {"file": _relative_path(str(info["path"]))}
    for k, v in info.items():
        if k not in _INTERNAL_KEYS:
            record[k] = v
    return record


def list_prompts(
    *,
    schedule: str | None = None,
    source: str | None = None,
    include_disabled: bool = False,
) -> None:
    """Print prompts grouped by schedule."""
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

    # Compute column widths
    all_names = list(configs.keys())
    name_width = max(len(n) for n in all_names) if all_names else 20
    name_width = max(name_width, 10)

    # Fixed widths for other columns
    title_width = 28
    output_width = 34
    tools_width = 24

    # Print column header
    header = (
        f"  {'NAME':<{name_width}}  {'TITLE':<{title_width}}  "
        f"{'OUTPUT':<{output_width}}  {'TOOLS':<{tools_width}}  TAGS"
    )
    print(header)
    print()

    # Print each non-empty group
    for group_name in ("segment", "daily", "unscheduled"):
        items = groups[group_name]
        if not items:
            continue

        # Skip group header if filtering to a single schedule
        if not schedule:
            print(f"{group_name}:")

        for key, info in items:
            title = info.get("title", "")[:title_width]
            output_path = _format_output_path(key, info)
            tools = _format_tools(info)[:tools_width]
            tags = _format_tags(info)
            src = ""
            if info.get("source") == "app":
                src = f" [{info.get('app', 'app')}]"

            # Build line with columns: name, title, output, tools, tags
            tag_part = f"  {tags}" if tags else ""
            line = (
                f"  {key:<{name_width}}  {title:<{title_width}}  "
                f"{output_path:<{output_width}}  {tools:<{tools_width}}{tag_part}{src}"
            )
            print(line.rstrip())

        if not schedule:
            print()

    # Show disabled count hint
    if not include_disabled:
        all_configs = _collect_configs(
            schedule=schedule, source=source, include_disabled=True
        )
        disabled_count = len(all_configs) - len(configs)
        if disabled_count:
            print(
                f"{len(configs)} prompts ({disabled_count} disabled hidden, use --disabled)"
            )


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
    skip_keys = {"path", "mtime"}

    label_width = 14

    def print_field(key: str, value: Any) -> None:
        if key in skip_keys:
            return
        val_str = str(value)
        # Truncate long descriptions for readability
        if key == "description" and len(val_str) > 72:
            val_str = val_str[:72] + "..."
        # Format hook config nicely
        if key == "hook" and isinstance(value, dict):
            post_hook = value.get("post", "")
            if post_hook:
                val_str = f"post: {post_hook}"
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


def _truncate_content(text: str, max_lines: int = 100) -> tuple[str, int]:
    """Truncate text to max_lines, returning (text, omitted_count)."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text, 0
    # Show first half and last half
    half = max_lines // 2
    truncated = lines[:half] + ["", f"... ({len(lines) - max_lines} lines omitted)"] + lines[-half:]
    return "\n".join(truncated), len(lines) - max_lines


def _format_section(title: str, content: str, full: bool = False) -> None:
    """Print a section with header and content."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")
    if not content or not content.strip():
        print("(empty)")
    elif full:
        print(content)
    else:
        truncated, omitted = _truncate_content(content)
        print(truncated)
        if omitted:
            print(f"\n(use --full to see all {omitted + 100} lines)")


def _yesterday() -> str:
    """Return yesterday's date in YYYYMMDD format."""
    return (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")


def show_prompt_context(
    name: str,
    *,
    day: str | None = None,
    segment: str | None = None,
    facet: str | None = None,
    query: str | None = None,
    full: bool = False,
) -> None:
    """Show full prompt context via dry-run.

    Builds config and pipes to `sol agents --dry-run` to show exactly
    what would be sent to the LLM provider.
    """
    # Load prompt metadata
    configs = get_muse_configs(include_disabled=True)
    if name not in configs:
        print(f"Prompt not found: {name}", file=sys.stderr)
        sys.exit(1)

    info = configs[name]
    has_output = bool(info.get("output"))
    has_tools = bool(info.get("tools"))
    schedule = info.get("schedule")
    is_multi_facet = info.get("multi_facet", False)

    # Determine prompt type
    if has_output and not has_tools:
        prompt_type = "generator"
    elif has_tools:
        prompt_type = "agent"
    else:
        print(f"Prompt '{name}' has no output or tools field", file=sys.stderr)
        sys.exit(1)

    # Validate day format if provided
    if day and (len(day) != 8 or not day.isdigit()):
        print(f"Invalid --day format: {day}. Expected YYYYMMDD.", file=sys.stderr)
        sys.exit(1)

    # Validate arguments based on type and schedule
    if prompt_type == "generator":
        # Generators need day, and segment-scheduled need segment
        if schedule == "segment" and not segment:
            print(
                f"Prompt '{name}' is segment-scheduled. Use --segment HHMMSS_LEN",
                file=sys.stderr,
            )
            sys.exit(1)
        if not day:
            day = _yesterday()
            print(f"Using day: {day} (yesterday)")

    if is_multi_facet and not facet:
        # List available facets
        try:
            from think.facets import get_facets

            facets = get_facets()
            facet_names = [k for k, v in facets.items() if not v.get("muted", False)]
            print(
                f"Prompt '{name}' is multi-facet. Use --facet NAME",
                file=sys.stderr,
            )
            print(f"Available facets: {', '.join(facet_names)}", file=sys.stderr)
        except Exception:
            print(
                f"Prompt '{name}' is multi-facet. Use --facet NAME",
                file=sys.stderr,
            )
        sys.exit(1)

    # Build config for dry-run
    config: dict[str, Any] = {"name": name}

    if prompt_type == "generator":
        config["day"] = day
        config["output"] = info.get("output", "md")
        if segment:
            config["segment"] = segment
        if facet:
            config["facet"] = facet
    else:
        # Tool agent - use get_agent() to build full config with instructions
        from think.utils import get_agent

        try:
            agent_config = get_agent(name, facet=facet)
            config.update(agent_config)
        except Exception as e:
            print(f"Failed to load agent config: {e}", file=sys.stderr)
            sys.exit(1)

        # Override prompt with user query
        if query:
            config["prompt"] = query
        else:
            config["prompt"] = "(no --query provided)"

    # Run sol agents --dry-run
    config_json = json.dumps(config)
    try:
        result = subprocess.run(
            ["sol", "agents", "--dry-run"],
            input=config_json + "\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        print("Dry-run timed out", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Could not find 'sol' command", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        print(f"Dry-run failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Parse JSONL output to find dry_run event
    dry_run_event = None
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        try:
            event = json.loads(line)
            if event.get("event") == "dry_run":
                dry_run_event = event
                break
            elif event.get("event") == "error":
                print(f"Error: {event.get('error')}", file=sys.stderr)
                sys.exit(1)
        except json.JSONDecodeError:
            continue

    if not dry_run_event:
        print("No dry_run event received", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(1)

    # Format and display output
    print(f"\n  Dry-run for: {name} ({dry_run_event.get('type', 'unknown')})")
    print(f"  Provider: {dry_run_event.get('provider')} / {dry_run_event.get('model')}")
    if dry_run_event.get("day"):
        print(f"  Day: {dry_run_event.get('day')}")
    if dry_run_event.get("segment"):
        print(f"  Segment: {dry_run_event.get('segment')}")
    if dry_run_event.get("output_path"):
        print(f"  Output: {dry_run_event.get('output_path')}")

    # Pre-hook info
    if dry_run_event.get("pre_hook"):
        mods = dry_run_event.get("pre_hook_modifications", [])
        print(f"  Pre-hook: {dry_run_event.get('pre_hook')} (modified: {', '.join(mods) or 'none'})")

    # System instruction (show before first if pre-hook modified it)
    if dry_run_event.get("system_instruction_before"):
        _format_section(
            "SYSTEM INSTRUCTION (before pre-hook)",
            dry_run_event.get("system_instruction_before", ""),
            full=full,
        )
    _format_section(
        f"SYSTEM INSTRUCTION (source: {dry_run_event.get('system_instruction_source', 'unknown')})",
        dry_run_event.get("system_instruction", ""),
        full=full,
    )

    # User instruction (agents only, show before first if pre-hook modified it)
    if dry_run_event.get("user_instruction"):
        if dry_run_event.get("user_instruction_before"):
            _format_section(
                "USER INSTRUCTION (before pre-hook)",
                dry_run_event.get("user_instruction_before", ""),
                full=full,
            )
        _format_section("USER INSTRUCTION", dry_run_event.get("user_instruction", ""), full=full)

    # Extra context (agents only)
    if dry_run_event.get("extra_context"):
        _format_section("EXTRA CONTEXT", dry_run_event.get("extra_context", ""), full=full)

    # Prompt (show before first if pre-hook modified it)
    prompt_source = dry_run_event.get("prompt_source", "")
    if prompt_source:
        prompt_source = f" (source: {_relative_path(prompt_source)})"
    if dry_run_event.get("prompt_before"):
        _format_section("PROMPT (before pre-hook)", dry_run_event.get("prompt_before", ""), full=full)
    _format_section(f"PROMPT{prompt_source}", dry_run_event.get("prompt", ""), full=full)

    # Transcript (generators only, show before first if pre-hook modified it)
    if "transcript" in dry_run_event:
        chars = dry_run_event.get("transcript_chars", 0)
        files = dry_run_event.get("transcript_files", 0)
        if dry_run_event.get("transcript_before"):
            before_chars = dry_run_event.get("transcript_before_chars", 0)
            _format_section(
                f"TRANSCRIPT (before pre-hook, {before_chars:,} chars)",
                dry_run_event.get("transcript_before", ""),
                full=full,
            )
        _format_section(
            f"TRANSCRIPT ({chars:,} chars from {files} files)",
            dry_run_event.get("transcript", ""),
            full=full,
        )

    # Tools (agents only)
    if dry_run_event.get("tools"):
        tools = dry_run_event.get("tools", [])
        if isinstance(tools, list):
            tools_str = ", ".join(tools)
        else:
            tools_str = str(tools)
        print(f"\n{'=' * 60}")
        print("  TOOLS")
        print(f"{'=' * 60}\n")
        print(tools_str)

    print()


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
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Show full prompt context (dry-run mode)",
    )
    parser.add_argument(
        "--day",
        metavar="YYYYMMDD",
        help="Day for prompt context (default: yesterday)",
    )
    parser.add_argument(
        "--segment",
        metavar="HHMMSS_LEN",
        help="Segment for segment-scheduled prompts",
    )
    parser.add_argument(
        "--facet",
        metavar="NAME",
        help="Facet for multi-facet prompts",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help="Sample query for tool agents",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full content without truncation",
    )

    args = setup_cli(parser)

    if args.prompt:
        if not args.name:
            print("--prompt requires a prompt name", file=sys.stderr)
            sys.exit(1)
        show_prompt_context(
            args.name,
            day=args.day,
            segment=args.segment,
            facet=args.facet,
            query=args.query,
            full=args.full,
        )
    elif args.name:
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
