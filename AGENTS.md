# solstone Developer Guide

This is the developer-facing documentation for the solstone codebase. If you're an AI agent working **inside a journal**, read the journal's own `AGENTS.md` instead — it's seeded from `docs/JOURNAL.md` and tells you about the journal layout.

## Key Concepts

- **Journal**: Central data structure organized as `journal/YYYYMMDD/` directories. All captured data, transcripts, and analysis artifacts are stored here.
- **Facets**: Project/context organization system that groups related content and provides scoped views of entities, tasks, and activities.
- **Entities**: Extracted information tracked over time across transcripts and interactions and associated with facets for semantic navigation.
- **Talents**: AI processors with configurable prompts that analyze content, extract insights, and respond to queries.
- **Callosum**: Message bus that enables asynchronous communication between components.
- **Indexer**: Builds and maintains a SQLite database from journal data, enabling fast search and retrieval.

## Architecture

**Core pipeline**: `observe` (capture) -> JSON transcripts -> `think` (analyze) -> SQLite index -> `convey` (web UI)

**Data organization**:
- Everything lives under `journal/YYYYMMDD/` daily directories.
- Import segments are anchored to creation/modification time, not content "about" time.
- Facets provide project-scoped organization and filtering.
- Entities are extracted from transcripts and tracked across time.
- The indexer builds a SQLite database for fast search and retrieval.

**Component communication**:
- Callosum enables async communication between services.
- Cortex orchestrates AI talent execution via `sol cortex`, spawning talent subprocesses with talent configurations.
- The unified CLI is `sol`. Run `sol` to see status and available commands.

## Quick Commands

```bash
make install   # Install package (includes all deps)
make skills    # Discover and symlink Anthropic Skills from talent/ dirs
make format    # Auto-fix formatting, then report remaining issues
make test      # Run unit tests
make ci        # Full CI check (format check + lint + test)
make dev       # Start stack (Ctrl+C to stop)
```

## Talent CLI Boundaries

Cogitate talents have access to all `sol` commands. The following infrastructure commands must never be called by talents because they manage services and data pipelines that should only be operated by the supervisor or a human operator:

- `sol supervisor` / `sol start`
- `sol dream` except heartbeat's targeted `sol dream --segment`
- `sol import`
- `sol config`
- `sol cortex`
- `sol providers check`
- `sol callosum`
- `sol observer` / `sol observe-*`
- `sol sense`
- `sol transcribe` / `sol describe`
- `sol indexer --reset`

Talents should use `sol call` commands for journal interaction and `sol health` / `sol talent logs` for diagnostics.

## Reference

For deeper material, see:
- `docs/project-structure.md`
- `docs/coding-standards.md`
- `docs/testing.md`
- `docs/environment.md`

## Known limitations

- Per-journal `AGENTS.md` files are seeded once at journal init by `apps/sol/maint/003_seed_agents_md.py`. They are not automatically refreshed if `docs/JOURNAL.md` changes upstream. Re-seed manually by deleting the journal's `AGENTS.md` and restarting the supervisor.
