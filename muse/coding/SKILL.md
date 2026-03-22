---
name: coding
description: >
  Solstone development guidelines, project structure, coding standards,
  testing, and environment. The single source of truth for anyone writing
  code on solstone — Claude Code sessions, hopper lodes, and coder
  sub-agents. Use when writing, reviewing, or testing solstone code.
  TRIGGER: code contribution, coding standards, project structure,
  testing, make commands, development setup, PR review.
---

# Development Guidelines

**solstone** is a Python-based AI-driven desktop journaling toolkit with three packages: `observe/` for multimodal capture and AI-powered analysis, `think/` for data post-processing, AI agent orchestration, and intelligent insights, and `convey/` for the web application, with `apps/` for extensions. The project uses a modular architecture where each package can operate independently while sharing common utilities and data formats through the journal system.

## Key Concepts

- **Journal**: Central data structure organized as `journal/YYYYMMDD/` directories. All captured data, transcripts, and analysis artifacts are stored here.
- **Facets**: Project/context organization system that groups related content and provides scoped views of entities, tasks, and activities.
- **Entities**: Extracted information tracked over time across transcripts and interactions and associated with facets for semantic navigation.
- **Agents**: AI processors with configurable prompts that analyze content, extract insights, and respond to queries.
- **Callosum**: Message bus that enables asynchronous communication between components.
- **Indexer**: Builds and maintains SQLite database from journal data, enabling fast search and retrieval.

## Architecture

**Core Pipeline**: `observe` (capture) → JSON transcripts → `think` (analyze) → SQLite index → `convey` (web UI)

**Data Organization**:
- Everything organized under `journal/YYYYMMDD/` daily directories.
- Import segments are anchored to creation/modification time, not content "about" time.
- Facets provide project-scoped organization and filtering.
- Entities are extracted from transcripts and tracked across time.
- Indexer builds SQLite database for fast search and retrieval.

**Component Communication**:
- Callosum message bus enables async communication between services.
- Cortex orchestrates AI agent execution via `sol cortex`, spawning agent subprocesses with agent configurations.
- The unified CLI is `sol`. Run `sol` to see status and available commands.

## Quick Commands

```bash
make install   # Install package (includes all deps)
make skills    # Discover and symlink Agent Skills from muse/ dirs
make format    # Auto-fix formatting, then report remaining issues
make test      # Run unit tests
make ci        # Full CI check (format check + lint + test)
make dev       # Start stack (Ctrl+C to stop)
```

## Agent CLI Boundaries

Cogitate agents have access to all `sol` commands. The following infrastructure
commands must **never** be called by agents — they manage services and data
pipelines that should only be operated by the supervisor or human operators:

- `sol supervisor` / `sol start` — service lifecycle management
- `sol dream` — full processing pipeline (only heartbeat uses `sol dream --segment` for targeted reprocessing)
- `sol import` — data injection into journal
- `sol config` — system configuration changes
- `sol cortex` — agent process manager (meta-spawning)
- `sol agents` — direct agent execution
- `sol callosum` — message bus server
- `sol observer` / `sol observe-*` — capture services
- `sol sense` — capture event dispatcher
- `sol transcribe` / `sol describe` — processing pipelines
- `sol indexer --reset` — destructive index rebuild (read-only queries via `sol indexer` are fine)

Agents should use `sol call` commands for journal interaction and `sol health` /
`sol muse logs` for diagnostics.

## Reference

- `reference/project-structure.md` — Directory layout, package organization, CLI routing, file locations.
- `reference/coding-standards.md` — Style rules, naming conventions, file headers, development principles, dependencies.
- `reference/testing.md` — Test structure, fixtures, make commands, worktree development.
- `reference/environment.md` — Journal paths, API keys, error handling, documentation pointers, git practices.
