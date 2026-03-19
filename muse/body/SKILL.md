---
name: body
description: Development guidelines, project structure, coding standards, testing, and environment for solstone. TRIGGER when contributing code, reviewing PRs, setting up development environment, or asking about project conventions.
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

## Reference

- `reference/project-structure.md` — Directory layout, package organization, CLI routing, file locations.
- `reference/coding-standards.md` — Style rules, naming conventions, file headers, development principles, dependencies.
- `reference/testing.md` — Test structure, fixtures, make commands, worktree development.
- `reference/environment.md` — Journal paths, API keys, error handling, documentation pointers, git practices.
