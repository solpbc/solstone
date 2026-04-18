# Sol CLI Developer Guide

How the `sol` CLI is organized, how to add new commands, and what files to maintain.

## Architecture

The CLI has two tiers with distinct purposes:

| Tier | Pattern | Framework | Purpose |
|------|---------|-----------|---------|
| **Top-level** | `sol <cmd>` | Custom dispatcher + argparse | System infrastructure — pipelines, daemons, orchestration |
| **Call** | `sol call <app> <cmd>` | Typer (auto-discovered) | Tool-callable functions — what agents and humans invoke for data operations |

### The boundary

**If an AI agent should tool-call it → `sol call`.** These commands appear in SKILL.md files and are invoked by talent agents during conversations.

**If it's system plumbing → `sol <cmd>`.** Processing pipelines, supervisor, services, capture — things that cron or systemd runs.

**Interactive entry points** (`sol chat`, `sol help`, `sol engage`) are top-level for discoverability even though they're user-facing. Agents don't invoke these.

## Top-Level Commands (`sol <cmd>`)

### How they work

`sol.py` contains a static `COMMANDS` dict mapping command names to module paths:

```python
COMMANDS: dict[str, str] = {
    "think": "think.thinking",
    "import": "think.importers.cli",
    ...
}
```

Each module must export a `main()` function. The dispatcher does `importlib.import_module(path)` then calls `module.main()`.

Commands are organized into `GROUPS` for help display, and `ALIASES` provide shortcuts (e.g., `sol up` → `sol service up`).

### Adding a top-level command

1. **Create the module** with a `main()` function:

```python
# think/my_cmd.py
import argparse

def main() -> None:
    parser = argparse.ArgumentParser(prog="sol my-cmd")
    parser.add_argument("--day", help="Day YYYYMMDD")
    args = parser.parse_args()
    # ... implementation
```

2. **Register in `sol.py`** — add to `COMMANDS`:

```python
COMMANDS: dict[str, str] = {
    ...
    "my-cmd": "think.my_cmd",
}
```

3. **Add to a group** in `GROUPS` for help display.

4. **No skill or AGENTS.md changes needed** — top-level commands aren't agent tools.

### Files to maintain

| File | What to do |
|------|-----------|
| `sol.py` COMMANDS dict | Register the command |
| `sol.py` GROUPS dict | Add to appropriate group |
| Module file (e.g., `think/my_cmd.py`) | Implement with `main()` |

## Call Commands (`sol call <app> <cmd>`)

### How they work

`think/call.py` is the gateway. It creates a root `typer.Typer()` and mounts sub-apps from two sources:

**Auto-discovered apps** — scans `apps/*/call.py` at import time:
```
apps/todos/call.py      → sol call todos ...
apps/calendar/call.py   → sol call calendar ...
apps/entities/call.py   → sol call entities ...
```

Each `call.py` must export `app = typer.Typer()`. The directory name becomes the sub-command name. Errors in one app don't prevent others from loading.

**Manually mounted built-ins** — for commands tightly coupled to `think/` internals:
```python
# think/call.py
from think.tools.call import app as journal_app
from think.tools.navigate import app as navigate_app
from think.tools.routines import app as routines_app
from think.tools.sol import app as sol_app

call_app.add_typer(journal_app, name="journal")
call_app.add_typer(navigate_app, name="navigate")
call_app.add_typer(routines_app, name="routines")
call_app.add_typer(sol_app, name="identity")
```

These live under `think/tools/` because they import think-internal APIs directly.

### Adding a new auto-discovered app

This is the happy path for most new commands.

1. **Create `apps/<name>/call.py`**:

```python
# apps/myapp/call.py
import typer
from think.facets import log_call_action

app = typer.Typer(help="Short description of what this app does.")

@app.command("list")
def list_items(
    day: str | None = typer.Option(None, "--day", "-d", help="Day YYYYMMDD."),
    facet: str | None = typer.Option(None, "--facet", "-f", help="Facet name."),
) -> None:
    """List items."""
    from think.utils import resolve_sol_day, resolve_sol_facet
    day = resolve_sol_day(day)
    facet = resolve_sol_facet(facet)
    # ... implementation
```

2. **That's it for the CLI.** Auto-discovery picks it up on next run.

3. **Create the agent skill** (if agents should use these commands):

```markdown
# apps/myapp/talent/myapp/SKILL.md
---
name: myapp
description: >
  What this skill does. When to trigger it.
  TRIGGER: keyword1, keyword2, keyword3.
---

# MyApp CLI Skill

Common pattern:
\`\`\`bash
sol call myapp <command> [args...]
\`\`\`

## list

\`\`\`bash
sol call myapp list [-d DAY] [-f FACET]
\`\`\`

List items for a day.

- `-d, --day`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `-f, --facet`: facet name (default: `SOL_FACET` env).
```

4. **Run `make skills`** to create the symlink in `journal/.agents/skills/`.

5. **Update AGENTS.md** — add the skill to the Skills table.

### Adding a manually-mounted built-in

Use this when the command depends heavily on `think/` internals and shouldn't live in `apps/`.

1. **Create `think/tools/<name>.py`** with `app = typer.Typer()`.
2. **Mount in `think/call.py`**:
```python
from think.tools.mytools import app as mytools_app
call_app.add_typer(mytools_app, name="mytools")
```
3. **Optionally create a skill** in `talent/<name>/SKILL.md`.

### Files to maintain for a new call command

| File | What to do | Required? |
|------|-----------|-----------|
| `apps/<name>/call.py` | Typer app with commands | Yes |
| `apps/<name>/talent/<name>/SKILL.md` | Skill doc for agents | If agents should use it |
| `journal/.agents/skills/<name>` | Symlink (via `make skills`) | Auto-generated |
| `AGENTS.md` Skills table | Add trigger description | If skill exists |
| `tests/test_<name>_call.py` | CLI tests | Yes |

## Conventions

### Environment defaults

Commands that take `--day` or `--facet` should respect `SOL_DAY` and `SOL_FACET` environment variables as defaults. Use the helpers:

```python
from think.utils import resolve_sol_day, resolve_sol_facet

day = resolve_sol_day(day_arg)    # Falls back to SOL_DAY env
facet = resolve_sol_facet(facet_arg)  # Falls back to SOL_FACET env
```

### Action logging

All mutating `sol call` commands should log their actions for audit:

```python
from think.facets import log_call_action

log_call_action(
    facet=facet,
    action="myapp_create",
    params={"key": "value"},
    day=day,
)
```

This writes to `facets/{facet}/logs/{day}.jsonl` (or `config/actions/{day}.jsonl` if facet is None).

### The `--consent` flag

Commands that agents invoke proactively (without the user explicitly asking) should accept a `--consent` flag:

```python
consent: bool = typer.Option(
    False,
    "--consent",
    help="Assert that explicit user approval was obtained before calling this command (agent audit trail).",
)
```

This is for audit trail — it records that the agent confirmed user consent before acting.

### Output patterns

- **JSON output**: Use `--json` flag for machine-readable output. Default to human-friendly text.
- **Errors**: Write to stderr via `typer.echo(..., err=True)` and `raise typer.Exit(1)`.
- **Confirmations**: Use `--yes` to skip interactive confirmation for destructive operations.
- **Pagination**: Use `--limit` / `--cursor` for list commands.

### Typer command naming

```python
@app.command("list")      # Verb as command name
@app.command("show")      # Singular operations
@app.command("create")    # CRUD verbs
```

Use lowercase, single-word names. Hyphenated names for multi-word (`list-nudges-due`, `set-name`).

## Directory Structure

```
solstone/
├── sol.py                          # Entry point + COMMANDS registry
├── think/
│   ├── call.py                     # sol call gateway (Typer root + mounts)
│   ├── tools/
│   │   ├── call.py                 # sol call journal (built-in)
│   │   ├── routines.py             # sol call routines (built-in)
│   │   └── sol.py                  # sol call identity (built-in)
│   └── *.py                        # Top-level command modules
├── apps/
│   ├── todos/
│   │   ├── call.py                 # sol call todos (auto-discovered)
│   │   ├── todo.py                 # Data models
│   │   └── talent/todos/SKILL.md     # Agent skill doc
│   ├── calendar/
│   │   ├── call.py                 # sol call calendar (auto-discovered)
│   │   └── talent/calendar/SKILL.md
│   ├── entities/call.py
│   ├── speakers/call.py
│   ├── support/call.py
│   ├── transcripts/call.py
│   ├── agent/call.py
│   ├── awareness/call.py
│   └── ... (web-only apps without call.py)
├── talent/
│   ├── journal/SKILL.md            # Skills not tied to an app
│   ├── coding/SKILL.md
│   └── *.md                        # Agent prompt files
├── journal/.agents/skills/          # Symlinks (generated by make skills)
└── AGENTS.md                        # Sol identity + skill table
```

### The `apps/` dual role

`apps/` contains both CLI apps (with `call.py`) and convey web routes (without). The presence of `call.py` is the marker for "this app exposes CLI commands." Web-only apps (home, search, settings, stats, etc.) only serve the convey UI.

## Current Command Inventory

### Top-level (`sol <cmd>`)

| Group | Commands |
|-------|----------|
| Think (processing) | `import`, `think`, `planner`, `indexer`, `supervisor`, `schedule`, `top`, `health`, `callosum`, `notify`, `heartbeat` |
| Service | `service` (+ aliases `up`, `down`, `start`) |
| Observe (capture) | `transcribe`, `describe`, `sense`, `transfer`, `observer` |
| Talent (AI agents) | `agents`, `cortex`, `talent`, `call`, `engage` |
| Convey (web UI) | `convey`, `restart-convey`, `screenshot`, `maint` |
| Specialized | `config`, `streams`, `journal-stats`, `formatter`, `detect-created` |
| Help | `help`, `chat` |

### Call (`sol call <app> <cmd>`)

| App | Source | Commands |
|-----|--------|----------|
| `todos` | `apps/todos/call.py` | list, add, done, cancel, move, upcoming, list-nudges-due, dispatch-nudges |
| `calendar` | `apps/calendar/call.py` | list, add, update, cancel, move, import-day |
| `entities` | `apps/entities/call.py` | list, show, search, observe, merge |
| `speakers` | `apps/speakers/call.py` | list, show, detect-owner, confirm-owner, clusters, suggest |
| `transcripts` | `apps/transcripts/call.py` | list, read, segments |
| `support` | `apps/support/call.py` | register, search, article, create, list, show, reply, attach, feedback, announcements, diagnose |
| `sol` | `apps/sol/call.py` | name, set-name, reset, thickness, set-owner, sol-init |
| `awareness` | `apps/awareness/call.py` | status, imports, log, log-read |
| `journal` | `think/tools/call.py` | search, events, facets, facet (show/create/update/rename/mute/unmute/delete/merge), news, agents, read, imports, import, retention purge, storage-summary |
| `routines` | `think/tools/routines.py` | list, templates, create, edit, delete, run, output, suggestions, suggest-respond, suggest-state |
| `identity` | `think/tools/sol.py` | self, partner, agency, pulse, briefing |
| `navigate` | `think/tools/navigate.py` | *(single command)* |

## Skill System

Skills are documented in `SKILL.md` files and symlinked into `journal/.agents/skills/` by `make skills`.

**Skill locations:**
- App skills: `apps/<name>/talent/<name>/SKILL.md`
- Core skills: `talent/<name>/SKILL.md`

**Skill ≠ call command.** Not every skill has a corresponding `call.py`, and not every `call.py` has a skill:
- `health`, `coding`, `vit` have skills but no `call.py`
- Some call apps provide the CLI while the skill provides agent behavioral context

Skills document the CLI commands but also add behavioral guidance beyond what `--help` shows (e.g., "check upcoming before adding a future todo to avoid duplicates").

### Keeping skills in sync

When you add or change a `sol call` command, update the corresponding SKILL.md. The skill doc is what agents actually read — they don't parse `--help` output. Include:
- Full command syntax with all flags
- Behavior notes (edge cases, defaults, validation)
- Examples showing common usage patterns
