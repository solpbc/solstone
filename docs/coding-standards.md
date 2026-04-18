# Coding Standards

## Language & Tools

- **Ruff** (`make format`) - Formatting, linting, and import sorting
- **mypy** (`make check`) - Type checking
- Configuration in `pyproject.toml`

## Naming Conventions

- **Modules/Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private Members**: `_leading_underscore`

## Code Organization

- **Imports**: Prefer absolute imports, grouped (stdlib, third-party, local), one per line
- **Docstrings**: Google or NumPy style with parameter/return descriptions
- **Type Hints**: Should be included on function signatures (legacy helpers may still need updates)
- **File Structure**: Constants → helpers → classes → main/CLI

## File Headers

All source code files (but not text or markdown files or prompts) must begin with a license and copyright header:

```
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc
```

Use `//` comments for JavaScript files.

## Development Principles

- **DRY, KISS, YAGNI**: Extract common logic, prefer simple solutions, don't over-engineer
- **Single Responsibility**: Functions/classes do one thing well
- **Conciseness & Maintainability**: Clear code over clever code
- **Robustness**: Minimize assumptions that must be kept in sync across the codebase, avoid fragility and increasing maintenance burden.
- **Self-Contained Codebase**: All code that depends on this project lives within this repository—never add backwards-compatibility shims, fallback aliases, re-exports for moved symbols, deprecated parameter handling, or legacy support code. When renaming or removing something, update all usages directly. For journal data format changes, write a migration script (see [docs/APPS.md](docs/APPS.md) for `maint` commands) instead of adding compatibility layers.
- **Trust system path resolution**: Never set `_SOLSTONE_JOURNAL_OVERRIDE` or bypass `get_journal()` from application code, agent prompts, subprocess environments, or service files. The env var exists only for tests and Makefile sandboxes. See `environment.md`.
- **Security**: Never expose secrets, validate/sanitize all inputs
- **Performance**: Profile before optimizing
- **Git**: Small focused commits, descriptive branch names. Run git commands directly (not `git -C`) since you're already in the repo.

## Dependencies

- **Minimize Dependencies**: Use standard library when possible
- **All Dependencies**: Add to `dependencies` in `pyproject.toml`
- **Package Manager**: [uv](https://docs.astral.sh/uv/) — lock file (`uv.lock`) is committed, `make install` syncs from it
- **Installation**: `make install` (creates isolated `.venv/` and syncs deps from the lock file for repo-local development)
- **Updating**: `make update` upgrades all deps to latest and regenerates the lock file

## Layer Hygiene

These invariants keep read paths pure, concentrate domain writes in one place per domain, and stop infrastructure modules (indexer, importer, scheduler, search, graph) from silently mutating cross-cutting state. They were derived from a codebase-wide audit of layer violations in April 2026 — see the motivating record at `vpe/workspace/solstone-layer-violations-audit.md` in the sol pbc internal extro repo (14 violations inventoried, remediation plan in flight).

The low-bar grep enforcement is `scripts/check_layer_hygiene.py`, wired into `make ci`. Known audit-flagged files are allowlisted with audit-reference TODOs; the allowlist shrinks as remediation bundles ship.

### L1 — Layer boundaries are load-bearing

Each module family has a declared responsibility. Infrastructure modules (indexer, importer, scheduler, search, graph, stats) may write **only their own output artifacts**. They may not create, modify, or delete domain state (entities, facets, observations, activities, events, chronicle day content). If an infrastructure module needs to trigger a domain mutation, it emits a callosum event or invokes a `sol call <domain> <verb>` subprocess — never writes domain state directly.

### L2 — Domain write ownership

Each domain has exactly **one** write-owning module. No other module may call `atomic_write`, `json.dump`, `open("w")`, `Path.write_text`, `unlink`, `rmtree`, etc. on that domain's on-disk state.

| Domain | Write-owning module(s) |
|--------|------------------------|
| Entities (`entities/*/entity.json`, `entities/*/*.npz`) | `think/entities/journal.py` + `think/entities/consolidation.py` + `think/entities/saving.py` + `think/entities/merge.py` + `apps/entities/call.py` |
| Facets (`facets/*/facet.json`, `facets/*/relationships/`) | `think/facets.py` + `apps/facets/*` (if/when created) |
| Observations (`observations.jsonl`) | `think/entities/observations.py` |
| Activities (`facets/*/activities/*.jsonl`) | `think/activities.py` |
| Facet events (`facets/*/events/*.jsonl`) | `think/hooks.py::write_events_jsonl`, called only via declared hook contract |
| Chronicle day content (`chronicle/YYYYMMDD/**`) | The capturing module (observer, importer) per its declared outputs |
| Index (SQLite, `indexer/*`) | `think/indexer/*` |

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
