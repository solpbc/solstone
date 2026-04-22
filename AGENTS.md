# solstone Developer Guide

This file is the **developer guide** for the solstone repository. Read it before writing code.

Audience:

- **Coders** (cwd = repo root, editing `observe/`, `think/`, `convey/`, `apps/`, `talent/`, `tests/`) — you're in the right place.
- **Cogitate talents** (cwd = `journal/`, running inside the live system) — your entry is `talent/journal/SKILL.md`, installed into `journal/.claude/skills/journal/` and `journal/.agents/skills/journal/`.
- **Operators** debugging a running system — see `docs/DOCTOR.md`.

For the journal-side runtime entry point, see `journal/AGENTS.md`.

`CLAUDE.md` and `GEMINI.md` at the repo root are symlinks to this file.

## 1. Start here

Read, in order, when you enter the repo for a coding task:

1. **This file through §8** — the invariants must be in working memory before your first edit.
2. **`sol.py`** — the CLI entry point. Skim the `COMMANDS`, `ALIASES`, and `GROUPS` dicts. ~340 lines, scannable in one pass. You now know the whole top-level command surface.
3. **`think/top.py` (first ~100 lines)** — the interactive TUI. Ties callosum + supervisor + service status together in one vantage point. Good "oh, this is how it connects" moment.
4. **The area you're about to touch:**
   - User-visible feature or `sol call <app> <verb>` → `apps/<name>/call.py` + `apps/<name>/routes.py` + `apps/<name>/templates/`.
   - Think pipeline → `think/<module>.py` + its tests.
   - AI talent prompt or behavior → `talent/<name>.md` (+ optional `.py` post-hook).
   - Capture / observe → `observe/<module>.py`.
5. **Run `sol`** (no args) — prints current journal status + grouped command list. Orients you to live state.
6. **`make dev`** or **`make sandbox`** when you need a running stack to iterate against.

> If you cannot state in one sentence **which module owns the data your change touches**, stop and re-read §7 L2 (the domain ownership table). Writing to a domain from the wrong module is how we got the 14 layer violations the April 2026 audit catalogued.

## 2. Repo map

| Dir | Purpose | Go here when | Depth doc |
|-----|---------|--------------|-----------|
| `sol.py` | CLI entry point — `COMMANDS` / `ALIASES` / `GROUPS` dicts | adding a top-level `sol <cmd>` | `docs/SOLCLI.md` |
| `observe/` | Multimodal capture — screen, audio, transcribe, describe, sense, transfer | capture-side bugs, new input modalities | `docs/OBSERVE.md` |
| `think/` | Post-processing core — cortex, talent, callosum, indexer, entities, facets, activities, scheduler, heartbeat, supervisor | anything downstream of capture; most coder work lives here | `docs/THINK.md`, `docs/CORTEX.md`, `docs/CALLOSUM.md` |
| `convey/` | Web app framework — app discovery, routing, bridge, screenshot tooling | layout / framework-level UI changes | `docs/CONVEY.md` |
| `apps/` | Convey apps — each self-contained (`call.py` Typer sub-app + `routes.py` + `templates/`) | adding a user-facing feature, a `sol call <app>` verb, a UI surface | `docs/APPS.md` (required reading before modifying `apps/`) |
| `talent/` | AI talent configs (markdown prompts + optional `.py` post-hooks) + `SKILL.md`s (journal, coder, partner, …) | defining or tuning a talent; adding a journal-side skill | `talent/journal/SKILL.md`, `docs/PROMPT_TEMPLATES.md` |
| `scripts/` | Repo maintenance scripts — `check_layer_hygiene.py`, `gate_agents_rename.py` | tooling that guards the codebase; wired into `make ci` | (none) |
| `tests/` | Pytest suites + `tests/fixtures/journal/` mock journal | writing tests; debugging flakiness; `make dev` / `make sandbox` use fixtures as the journal | `docs/testing.md` |
| `docs/` | All longform documentation | reference lookups; never your first stop | §10 below |
| `journal/` | The live journal (user data). Git-ignored content; checked-in template (`AGENTS.md`, skills symlinks) | **rarely as a coder** — modify `think/`, `apps/`, or `talent/`, not journal data | `talent/journal/SKILL.md` |

Top-level dirs intentionally not in the table: `.venv/`, `scratch/`, `logs/`, `tmp/`, `observers/`, `routines/`, `skills/` — not active coder surfaces.

## 3. Mental model

**The pipeline:** `observe` (capture) → JSON transcripts in `journal/chronicle/YYYYMMDD/` → `think` (analyze) → SQLite index + derived artifacts → `convey` (web UI) and `sol call` CLIs.

**Think is the center.** observe feeds it raw material; convey + apps render its outputs; talent prompts + cortex run AI against it; indexer makes it searchable. A change in `think/` usually ripples outward.

**Key concepts, priority-ordered:**

- **Journal** — the on-disk record rooted at `journal/` in the repo. Every day is a `journal/chronicle/YYYYMMDD/` directory. Segments (timestamped capture windows) are anchored to creation/modification time, not content "about" time. `get_journal()` from `think.utils` is the single source of truth for journal path resolution; trust it unconditionally. Never set `_SOLSTONE_JOURNAL_OVERRIDE` from application code (see §8).
- **Talents** — AI processors (markdown prompt + optional Python post-hook). Each has a config in `talent/<name>.md` with frontmatter that declares hooks, priority, model, and output. Cortex spawns them as subprocesses.
- **Callosum** — Unix-socket JSON message bus at `journal/health/callosum.sock`. Real-time event distribution across services (`tract` + `event` + payload). If components need to talk asynchronously, they talk through callosum.
- **Cortex** — process manager for talent runs. Listens on callosum (`tract="cortex"`, `event="request"`), spawns `python -m think.talents` subprocesses, writes `<talent>/<ts>_active.jsonl` then renames to `<talent>/<ts>.jsonl` on completion, broadcasts all events back through callosum. Read `docs/CORTEX.md` before modifying talent execution.
- **Facets** — project/context scopes (`work`, `personal`, …). Group related entities, activities, and relationships. Facet data lives under `journal/facets/<facet>/`.
- **Entities** — tracked people / projects / tools. Extracted from transcripts and accumulated across time. Canonical records in `journal/entities/<slug>/entity.json`.
- **Activities** — scheduled or observed "things that happen" (meetings, deadlines, anticipated events). Per-facet JSONL at `journal/facets/<facet>/activities/<day>.jsonl`. Sources: `anticipated` (from `talent/schedule.md`), `user` (manual), `cogitate` (talent-inferred).
- **Indexer** — reads journal state, builds SQLite + FTS5 index. **Never** mutates source data (§7 L6). Rerunning on unchanged data is a no-op.
- **Supervisor** — top-level process manager. Starts/restarts services, talks to callosum. `sol supervisor` / `sol start`.

## 4. The sol CLI

Two surfaces:

- **`sol <command>`** — top-level commands registered in `sol.py`'s `COMMANDS` dict (e.g., `sol import`, `sol think`, `sol indexer`, `sol supervisor`, `sol heartbeat`). `ALIASES` provides a couple of shorthand compound commands (`sol start` → `sol supervisor`, `sol up/down` → `sol service up/down`).
- **`sol call <app> <verb>`** — routes to `think/call.py`, which discovers each `apps/*/call.py` Typer sub-app and mounts it as a subcommand. Example: `sol call entities list`, `sol call activities create`, `sol call journal search`.

**Adding a top-level command:** add an entry to `COMMANDS` in `sol.py`; ensure the module has a `main()` function.

**Adding a `sol call` sub-verb:** add it to the app's `apps/<app>/call.py` Typer sub-app. No central registration needed — `think/call.py` discovers apps automatically.

Run `sol` (no args) for live status plus the full grouped command list.

## 5. Make commands

Verified against `Makefile`. Grouped by use.

### Install

| Target | When to use |
|--------|-------------|
| `make install` | First setup and whenever `pyproject.toml` or `uv.lock` changes. Creates `.venv/`, syncs deps, installs Playwright chromium, runs `make skills`. |
| `make skills` | After adding or renaming a `SKILL.md` under `talent/` or `apps/*/talent/`. Rewrites the `.claude/` + `.agents/` skill symlinks into `journal/`. (`make install` depends on this; rarely run alone.) |
| `make update` | Upgrade all deps to latest, regenerate `uv.lock`. Expect test churn. |
| `make update-prices` | Refresh genai-prices model-cost data when adding a new provider model or when pricing tests fail. |
| `make clean` | Remove build artifacts, caches, and the skill symlinks. Does not touch `.venv/`. |
| `make clean-install` | Nuke `.venv/` and `.installed`, then reinstall. Recovery path when the venv is wedged. |

### Run the stack

| Target | When to use |
|--------|-------------|
| `make dev` | Start the full stack (supervisor + callosum + sense + cortex + convey) against `tests/fixtures/journal/`, no observers, no daily processing. Primary inner-loop for UI work. Ctrl-C to stop. |
| `make sandbox` | Ephemeral background sandbox: copies fixtures to a temp journal, starts supervisor in the background, waits for readiness, writes `.sandbox.pid` / `.sandbox.journal`. Pair with verify targets below. Always follow with `make sandbox-stop`. |
| `make sandbox-stop` | Terminate the backgrounded sandbox and clean up state files. |
| `make sail` | Restart the **installed** solstone service via `sol service restart --if-installed`. No-op when no service is installed (typical in a dev worktree). |

### Format, lint, test

| Target | When to use |
|--------|-------------|
| `make format` | Auto-fix formatting and imports with ruff. Safe to run anytime; modifies files. |
| `make format-check` | Format dry-run. Part of `make ci`; rarely run alone. |
| `make test` | Unit tests (`tests/`). Format-check runs first; failures block tests. Fast inner loop. |
| `make test-apps` | Run all `apps/*/tests/` suites. |
| `make test-app APP=<name>` | Run a single app's tests. |
| `make test-only TEST=<path-or-pattern>` | Run a specific test file or pytest node id (`TEST="-k test_name"` also works). |
| `make test-integration` | Full integration suite. Requires `.env` API keys. Slow; run before shipping AI-behavior changes. |
| `make test-integration-only TEST=<path>` | Single integration test by path or pattern. |
| `make test-all` | Everything — core + apps + integration. Pre-ship gate. |
| `make coverage` | HTML coverage report under `htmlcov/`. Occasional. |
| `make watch` | pytest-watch — reruns tests on file change. Useful during a test-heavy sprint. |
| `make ci` | Format-check + ruff + agents-rename gate + layer-hygiene + tests. **Run before every commit.** |
| `make verify` | Same steps as `make ci`. Either name is fine. |
| `make install-checks` | The pre-test half of `make ci` (format-check + ruff + gates). Called by `ci` / `verify`. |
| `make check-layer-hygiene` | Run `scripts/check_layer_hygiene.py` alone. Useful when iterating on an L1–L2 violation flagged by CI. |
| `make gate-agents-rename` | Guard against reintroducing the old `agents/` naming. Part of `install-checks`. |

### Verification against a running sandbox

| Target | When to use |
|--------|-------------|
| `make verify-api` | Start a sandbox, run `tests/verify_api.py` against its convey port, stop the sandbox. API-regression check. |
| `make update-api-baselines` | Same, but update the baseline fixtures instead of failing on diff. Run after intentional API changes. |
| `make verify-browser` | Start a sandbox, run `tests/verify_browser.py` (pinchtab-driven browser scenarios), stop the sandbox. UI-regression check. |
| `make update-browser-baselines` | Browser-baselines equivalent of `update-api-baselines`. |
| `make review` | Full product verification: sandbox + API verify + browser verify, in one command. Pre-ship gate for anything user-visible. Requires pinchtab. |
| `make install-pinchtab` | One-time install of the pinchtab browser driver used by `make review` / `make verify-browser`. |

### Service management (systemd / launchd)

| Target | When to use |
|--------|-------------|
| `make install-service` | Install `sol` as a systemd user service (Linux) or launchd agent (macOS), convey on port 5015 (override with `PORT=8000`). Makes the machine a live solstone host — rarely wanted in a worktree. |
| `make uninstall-service` | Remove the installed service. |
| `make service-logs` | Tail the installed service's logs. |

### Other

| Target | When to use |
|--------|-------------|
| `make pre-commit` | Install pre-commit hooks (optional; most coders rely on `make ci` directly). |
| `make versions` | Print versions of Python, uv, and key deps. Diagnostic. |

### Don't use

| Target | Why not |
|--------|---------|
| `make uninstall` | Disabled by design. Use `make uninstall-service` (for installed artifacts) or `make clean-install` (to rebuild the dev env). |

## 6. Testing quickstart

- **Framework:** pytest. Files `test_*.py`, functions `test_*`. Shared fixtures in `tests/conftest.py`.
- **Fixture journal:** `tests/fixtures/journal/` — a complete mock journal with facets, entities, segments, index state. Tests set `os.environ["_SOLSTONE_JOURNAL_OVERRIDE"] = "tests/fixtures/journal"` (or use `monkeypatch.setenv`); this is the **only** place that env var is valid (see §8).
- **Run one test:** `make test-only TEST=tests/test_utils.py::test_foo` or `TEST="-k test_foo"`.
- **Run app tests:** `make test-apps` or `make test-app APP=<name>`.
- **Integration tests** (`tests/integration/`): hit real provider APIs, require `.env` keys, run via `make test-integration`.
- **After editing `convey/` or `apps/`:** `sol restart-convey` to reload code in a running stack.
- **Screenshots for UI review:** `sol screenshot <route>` (captures into `scratch/`).
- **`make dev` + `make sandbox`** both write runtime artifacts into the fixtures journal; `tests/fixtures/journal/.gitignore` covers those — never commit them.

Full depth: `docs/testing.md`.

## 7. Layer hygiene — required reading (L1–L9)

**Why this lives here.** A codebase-wide audit in April 2026 found 14 layer-hygiene violations in `think/` and `apps/`. Infrastructure modules (indexer, importers, schedulers) were silently writing domain state; CLI read-verbs were mutating; get-prefixed functions were creating records on miss. These invariants encode the rules the audit distilled, so the same landmines don't get re-planted. They're inlined here because a one-click-away invariant is a routinely-skipped invariant.

The low-bar grep enforcement is `scripts/check_layer_hygiene.py`, wired into `make ci`. Known audit-flagged files are allowlisted with audit-reference TODOs; the allowlist shrinks as remediation bundles ship.

### L1 — Layer boundaries are load-bearing

Each module family has a declared responsibility. Infrastructure modules (indexer, importer, scheduler, search, graph, stats) may write **only their own output artifacts**. They may not create, modify, or delete domain state (entities, facets, observations, activities, events, chronicle day content). If an infrastructure module needs to trigger a domain mutation, it emits a callosum event or invokes a `sol call <domain> <verb>` subprocess — never writes domain state directly.

### L2 — Domain write ownership

Each domain has exactly **one** write-owning module (or one tightly-scoped family of modules). No other module may call `atomic_write`, `json.dump`, `open("w")`, `Path.write_text`, `unlink`, `rmtree`, etc. on that domain's on-disk state.

| Domain | Write-owning module(s) |
|--------|------------------------|
| Entities (`entities/*/entity.json`, `entities/*/*.npz`) | `think/entities/journal.py` + `think/entities/consolidation.py` + `think/entities/saving.py` + `think/entities/merge.py` + `apps/entities/call.py` |
| Facets (`facets/*/facet.json`, `facets/*/relationships/`) | `think/facets.py` + `apps/facets/*` (if/when created) |
| Observations (`observations.jsonl`) | `think/entities/observations.py` |
| Activities (`facets/*/activities/*.jsonl`) | `think/activities.py` |
| Chronicle day content (`chronicle/YYYYMMDD/**`) | The capturing module (observer, importer) per its declared outputs |
| Index (SQLite, `indexer/*`) | `think/indexer/*` |

If you're about to write to a domain from a module not in this table, stop and route through the owner.

### L3 — Naming is a contract

Function and CLI-subcommand verbs signal read vs. write intent.

**Read verbs** (functions and CLI subcommands): `load_*`, `get_*`, `read_*`, `scan_*`, `list_*`, `show_*`, `find_*`, `match_*`, `resolve_*`, `query_*`, `lookup_*`, `status_*`, `check_*`, `validate_*`, `discover_*`, `format_*`, `render_*`, `extract_*`, `parse_*`, `view_*`, `inspect_*`, `info_*`, `describe_*`, `search_*`.

A read-verb function must not mutate on-disk state. No exceptions for caches. No exceptions for "create on miss."

If a function needs create-on-miss semantics, split it:

```python
entity = load_entity(eid) or create_entity(eid, ...)
```

This makes the write visible at every call site.

**Write verbs** are the ones allowed to write — choose the right one: `save_`, `create_`, `add_`, `insert_`, `append_`, `attach_`, `delete_`, `remove_`, `update_`, `rename_`, `move_`, `promote_`, `merge_`, `seed_`, `consolidate_`, `bootstrap_`, `backfill_`, `dispatch_`, `record_`, `ingest_`, `import_`, `rebuild_`.

### L4 — CLI read-verbs are read-only

CLI subcommands with read verbs (list, show, status, get, search, find, check, validate, discover, inspect, info, describe, read, view) must not write to journal domain state under any flag combination. If a command needs a write path, split it into two commands — a read-verb reader and a write-verb writer.

### L5 — Write-verb defaults

CLI subcommands with write verbs default to safe.

- Preferred: no default mutation; an explicit `--commit` (or `--apply`) flag is required to perform the write.
- Acceptable alternative: `--dry-run` defaulting to `False` *only if* the subcommand name is unambiguously a write verb AND the command's primary user journey is the write (e.g., `sol call entities create`).

"Bootstrap", "backfill", and "resolve-names" are not unambiguous — default them to dry-run.

### L6 — Indexers never mutate source data

An indexer's job is to build indexes from source-of-truth data. Indexers may not mutate the source data they read. Re-running `sol indexer --rescan` on an unchanged journal must be a no-op for domain state.

### L7 — Importers only write to imports/

Importers write source material to `imports/` and the raw-content areas of `chronicle/`. They may not create or modify entities, facets, observations, or other cross-cutting state. If an importer needs to create an entity for deduplication, it calls a domain-owned `seed_entity()` function in `think/entities/` that surfaces the write explicitly.

### L8 — Hooks have declared outputs

Post-processing hooks (`think/hooks.py`, `talent/*.py` hook functions) declare every path they will write in their frontmatter. The hook runner validates that all actual writes match the declaration. Writes outside the declared set fail loudly — raise at runtime; assert in tests.

### L9 — Event handlers are idempotent

Any function that handles a callosum event, a scheduled tick, or a supervisor-started automation is idempotent w.r.t. on-disk state. Append-only history records dedupe by a natural key (usually `(day, segment)` or `(day, segment, ts)`). Before adding a write to an event handler, ask: "what happens if this event fires twice?"

## 8. Coding invariants

The rules above govern *where* code lives. The rules below govern *how* code behaves. They exist because we got burned.

- **No backwards-compatibility shims.** All code that depends on this project lives in this repository — never add fallback aliases, re-exports for moved symbols, deprecated-parameter handling, or legacy support code. When renaming or removing something, update every usage directly. For journal data-format changes, write a migration script (see `docs/APPS.md` for `maint` commands); do not add a compatibility layer. Cogitate agents default to adding shims; resist this.
- **Trust `get_journal()` unconditionally.** `get_journal()` from `think.utils` is the single source of truth for journal path resolution. **Never** set `_SOLSTONE_JOURNAL_OVERRIDE` from application code, agent prompts, subprocess environments, or service files. The env var exists exclusively for two external contexts: test harnesses (`monkeypatch.setenv`) and Makefile sandboxes. If you think you need to override the path from app code, you don't — fix the actual problem. See `docs/environment.md`.
- **SPDX header on every source file.** All Python (and other source) files begin with:

  ```python
  # SPDX-License-Identifier: AGPL-3.0-only
  # Copyright (c) 2026 sol pbc
  ```

  (`//` for JavaScript.) Markdown, text, and prompt files don't need it.
- **Fail loudly, not silently.** Raise specific exceptions with clear messages; use the `logging` module, not `print`. Validate inputs at module boundaries. A silent swallow in production costs days of forensics — an error at the boundary is free.
- **Trust internal code.** Don't add defensive validation for things internal callers can't violate. Validate at system boundaries (user input, external APIs, imported files) — not between modules you control.

Generic software principles (DRY, KISS, YAGNI, single responsibility, small focused commits) apply; see `docs/coding-standards.md` for the full list.

## 9. File headers, naming, dependencies

- **SPDX header** as above — mandatory on source code files.
- **Naming:** modules / functions / variables `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`; private members `_leading_underscore`. Full table in `docs/coding-standards.md`.
- **Imports:** prefer absolute (`from think.utils import get_journal`), grouped stdlib → third-party → local, one per line.
- **Type hints** on function signatures; `mypy` via `make check`.
- **Dependencies:** managed by [uv](https://docs.astral.sh/uv/). `pyproject.toml` is authoritative; `uv.lock` is committed; `make install` syncs; `make update` refreshes.
- **Python 3.10+.**

## 10. Commit hygiene

- Small, focused commits with descriptive messages.
- Run `make ci` before every commit.
- Run `git` commands directly — not `git -C` — you're already in the repo.
- Don't commit runtime artifacts written under `tests/fixtures/journal/` by `make dev` / `make sandbox` (`.gitignore` covers them; verify with `git status` anyway).

## 11. Where to go deeper

Bare links don't motivate clicking. Each entry below says when you actually need the doc.

| Doc | When to read |
|-----|--------------|
| `docs/APPS.md` | **Required before modifying `apps/`** — pattern catalog for Convey apps, hook-idempotency guidance, Typer sub-app conventions, `maint` commands for data migrations |
| `docs/THINK.md` | Understanding the think-layer pipeline (importers, indexer, segment/stream processing) |
| `docs/CORTEX.md` | Modifying talent execution, cortex lifecycle, talent process management |
| `docs/CALLOSUM.md` | Adding a new tract/event, debugging message flow |
| `docs/CONVEY.md` | Framework-level web changes (as opposed to an individual app) |
| `docs/OBSERVE.md` | Capture-side work: new modalities, transcription, sensing |
| `docs/SOLCLI.md` | Adding a new `sol <cmd>` or `sol call <app> <verb>` |
| `docs/PROMPT_TEMPLATES.md` | Modifying talent prompt format or frontmatter |
| `docs/PROVIDERS.md` | Adding a new AI provider; debugging model selection |
| `docs/testing.md` | Writing integration tests; setting up fixtures; debugging test isolation |
| `docs/environment.md` | Journal path resolution, service install details, `_SOLSTONE_JOURNAL_OVERRIDE` rules |
| `docs/coding-standards.md` | Full naming conventions, ruff / mypy config, dep-management details — reference for everything not promoted into this file |
| `docs/project-structure.md` | Canonical directory layout; resolving "where does this file go" debates |
| `docs/DOCTOR.md` | Diagnostics and debugging a running system |
| `docs/SCREEN_CATEGORIES.md` | Screen-understanding classifier taxonomy (observe side) |
| `docs/INTEGRATION_TESTS.md` | Deep integration-test setup |
| `docs/VENDOR.md` | Vendor-level integrations |
| `docs/design/` | Per-subsystem design docs |
| `docs/JOURNAL.md` | **Breadcrumb only** — redirects to `talent/journal/SKILL.md`, the progressive-disclosure journal-layout reference |
| `talent/journal/SKILL.md` | Journal layout, vocabulary, and `sol call journal` CLI (loaded by cogitate talents on demand via skills) |
| `talent/journal/references/cli.md` | Full `sol call journal` reference, including **Talent CLI Boundaries** (which infrastructure commands cogitate talents must not call) |

The live journal also carries `journal/AGENTS.md` as its runtime-facing breadcrumb.

`docs/BACKLOG.md` and `docs/ROADMAP.md` are product-planning docs — CPO/CEO reading, not coder reading.

## 12. What this file is NOT

- **Not a runtime guide for cogitate talents.** Runtime CLI restrictions on talents live in `talent/journal/references/cli.md` § Talent CLI Boundaries. If you're tuning what a talent can or cannot call, look there, not here.
- **Not the journal-layout reference.** `talent/journal/SKILL.md` + its `references/` is the cogitate-audience entry point. This file describes *how those commands are implemented*, not *which ones talents can't call*.
- **Not an operations manual.** For debugging a live system see `docs/DOCTOR.md`; for service management, the `make install-service` family.
