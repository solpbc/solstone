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
