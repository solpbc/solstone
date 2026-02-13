# Development Guidelines & Contribution Standards

This document provides comprehensive guidelines for contributing to solstone, whether you're an AI assistant, human developer, or automated system.

## Project Overview

**solstone** is a Python-based AI-driven desktop journaling toolkit that provides:

* **observe/** - Multimodal capture (audio + visual) and AI-powered analysis
* **think/** - Data post-processing, AI agent orchestration, and intelligent insights
* **convey/** - Web application for navigating and interacting with captured content (extensible via apps/)

The project uses a modular architecture where each package can operate independently while sharing common utilities and data formats through the journal system.

---

## Key Concepts

Understanding these core concepts is essential for working with solstone:

* **Journal**: Central data structure organized as `JOURNAL_PATH/YYYYMMDD/` directories. All captured data, transcripts, and analysis artifacts are stored here. See [docs/JOURNAL.md](docs/JOURNAL.md).

* **Facets**: Project/context organization system (e.g., "work", "personal", "acme"). Facets group related content and provide scoped views of entities, tasks, and activities.

* **Entities**: Extracted information (people, projects, concepts) tracked over time across transcripts and interactions. Entities are associated with facets and enable semantic navigation.

* **Agents**: AI processors with configurable prompts that analyze content, extract insights, and respond to queries. See [docs/THINK.md](docs/THINK.md) for the agent system and [docs/CORTEX.md](docs/CORTEX.md) for eventing.

* **Callosum**: Message bus that enables asynchronous communication between components. See [docs/CALLOSUM.md](docs/CALLOSUM.md).

* **Indexer**: Builds and maintains SQLite database from journal data, enabling fast search and retrieval.

---

## Project Structure

```
solstone/
├── sol.py          # Unified CLI entry point (run: sol <command>)
├── observe/        # Multimodal capture & AI analysis
├── think/          # Data post-processing, AI agents & orchestration
├── convey/         # Web app frontend & backend
├── apps/           # Convey app extensions (see docs/APPS.md)
├── muse/           # Agent/generator configs + Agent Skills (muse/*/SKILL.md)
├── tests/          # Pytest test suites + test fixtures under tests/fixtures/
├── docs/           # All documentation (*.md files)
├── AGENTS.md       # Development guidelines (this file)
├── CLAUDE.md       # Symlink to AGENTS.md for Claude Code
└── README.md       # Project overview
```

Each package has a README.md symlink pointing to its documentation in `docs/`.

### Package Organization

* **Python**: Requires Python 3.10+
* **Modules**: Each top-level folder is a Python package with `__init__.py` unless it is data-only (e.g., `tests/fixtures/`)
* **Imports**: Prefer absolute imports (e.g., `from think.utils import setup_cli`) whenever feasible
* **Entry Points**: Commands are registered in `sol.py`'s `COMMANDS` dict (pyproject.toml just defines the `sol` entry point)
* **Journal**: Data stored under `JOURNAL_PATH` (see Environment Management below)
* **Calling**: When calling other modules as a separate process always use `sol <command>` and never call using `python -m ...` (e.g., use `sol indexer`, NOT `python -m think.indexer`)

---

## Architecture & Data Flow

**Core Pipeline**: `observe` (capture) → JSON transcripts → `think` (analyze) → SQLite index → `convey` (web UI)

**Data Organization**:
* Everything organized under `JOURNAL_PATH/YYYYMMDD/` daily directories
* Facets provide project-scoped organization and filtering
* Entities are extracted from transcripts and tracked across time
* Indexer builds SQLite database for fast search and retrieval

**Component Communication**:
* Callosum message bus enables async communication between services
* Cortex orchestrates AI agent execution via `sol cortex`, spawning agent subprocesses with agent configurations
* See [docs/THINK.md](docs/THINK.md) for agent system details and [docs/CORTEX.md](docs/CORTEX.md) for the eventing protocol

**Command Reference**:
The unified CLI is `sol`. Run `sol` to see status and available commands. Use `sol <command>` for subcommands or `sol <module.path>` for direct module access.

---

## Testing with Fixtures

```python
# Use comprehensive mock journal data for testing
os.environ["JOURNAL_PATH"] = "tests/fixtures/journal"
# Now all journal operations work with test data
```

The `tests/fixtures/journal/` directory contains a complete mock journal structure with sample facets, agents, transcripts, and indexed data for testing.

---

## Coding Standards & Style

### Language & Tools
* **Black** (`make format`) - Code formatting
* **isort** (`make format`) - Import sorting
* **flake8** (`make lint`) - Linting
* **mypy** (`make check`) - Type checking
* Configuration in `pyproject.toml`

### Naming Conventions
* **Modules/Functions/Variables**: `snake_case`
* **Classes**: `PascalCase`
* **Constants**: `UPPER_SNAKE_CASE`
* **Private Members**: `_leading_underscore`

### Code Organization
* **Imports**: Prefer absolute imports, grouped (stdlib, third-party, local), one per line
* **Docstrings**: Google or NumPy style with parameter/return descriptions
* **Type Hints**: Should be included on function signatures (legacy helpers may still need updates)
* **File Structure**: Constants → helpers → classes → main/CLI

### File Headers
All source code files (but not text or markdown files or prompts) must begin with a license and copyright header:
```
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc
```
Use `//` comments for JavaScript files.

---

## Testing & Quality Assurance

### Test Structure
* **Framework**: pytest with coverage reporting
* **Unit Tests**: `tests/` root directory
  - Fast, no external API calls
  - Use `tests/fixtures/journal/` mock data
  - Test individual functions and modules
* **Integration Tests**: `tests/integration/` subdirectory
  - Test real backends (Anthropic, OpenAI, Google)
  - Require API keys in `.env`
  - Test end-to-end workflows
* **Naming**: Files `test_*.py`, functions `test_*`
* **Fixtures**: Shared fixtures in `tests/conftest.py`

### Running Tests

See **Quick Reference** below for all `make` commands. Key patterns:
- `make test` for unit tests, `make test-integration` for integration tests
- `make test-only TEST=path` to run specific tests
- `make ci` before committing (formats, lints, tests)
- Always run `sol restart-convey` after editing `convey/` or `apps/` to reload code
- Use `sol screenshot <route>` to capture UI screenshots for visual testing

---

## Important Development Notes

### Environment Management
* **JOURNAL_PATH**: The live journal path is stored in `.env`. To access it:
  - **Shell/CLI**: Run `grep JOURNAL_PATH .env` to get the path, then use it directly
  - **Python**: Use `get_journal()` from `think.utils` - it handles `.env` loading and auto-creates a platform-specific default if unset
* **API Keys**: Store in `.env` file, never commit to repository

### Error Handling & Logging
* Raise specific exceptions with clear messages
* Use logging module, not print statements
* Validate all external inputs (paths, user data)
* Fail fast with clear errors - avoid silent failures

### Documentation & References
* Update README files for new functionality
* Code comments explain "why" not "what"
* Function signatures should include type hints; highlight gaps when touching older modules
* **All docs in `docs/`**: Browse for JOURNAL.md, APPS.md, CORTEX.md, CALLOSUM.md, THINK.md, and more
* **App/UI work**: [docs/APPS.md](docs/APPS.md) is required reading before modifying `apps/`

---

## Dependencies Management

* **Minimize Dependencies**: Use standard library when possible
* **All Dependencies**: Add to `dependencies` in `pyproject.toml`
* **Package Manager**: [uv](https://docs.astral.sh/uv/) — lock file (`uv.lock`) is committed, `make install` syncs from it
* **Installation**: `make install` (creates isolated `.venv/`, syncs deps from lock file, symlinks `sol` to `~/.local/bin`)
* **Updating**: `make update` upgrades all deps to latest and regenerates the lock file

---

## Development Principles

* **DRY, KISS, YAGNI**: Extract common logic, prefer simple solutions, don't over-engineer
* **Single Responsibility**: Functions/classes do one thing well
* **Conciseness & Maintainability**: Clear code over clever code
* **Robustness**: Minimize assumptions that must be kept in sync across the codebase, avoid fragility and increasing maintenance burden.
* **Self-Contained Codebase**: All code that depends on this project lives within this repository—never add backwards-compatibility shims, fallback aliases, re-exports for moved symbols, deprecated parameter handling, or legacy support code. When renaming or removing something, update all usages directly. For journal data format changes, write a migration script (see [docs/APPS.md](docs/APPS.md) for `maint` commands) instead of adding compatibility layers.
* **Security**: Never expose secrets, validate/sanitize all inputs
* **Performance**: Profile before optimizing
* **Git**: Small focused commits, descriptive branch names. Run git commands directly (not `git -C`) since you're already in the repo.

---

## Quick Reference

### Common Commands
```bash
# Development setup
make install       # Install package (includes all deps)
make skills        # Discover and symlink Agent Skills from muse/ dirs
make format        # Auto-fix formatting, then report remaining issues
make test          # Run unit tests

# Testing
make test-apps              # Run app tests
make test-integration       # Run integration tests
make test-all               # Run all tests (core + apps + integration)
make coverage               # Generate coverage report

# Before pushing
make ci            # Full CI check (format check + lint + test)

# Debugging
sol restart-convey            # Restart Convey service (after code changes)
sol screenshot <route>        # Capture Convey view screenshot (use -h for options)

# Cleanup
make clean         # Remove artifacts
make clean-install # Clean and reinstall
```

### Worktree Development

Run the full stack (supervisor + callosum + sense + cortex + convey) against test fixture data:

```bash
make dev                    # Start stack (Ctrl+C to stop)
```

In a second terminal, take screenshots or hit endpoints:
```bash
export JOURNAL_PATH=tests/fixtures/journal
export PATH=$(pwd)/.venv/bin:$PATH
sol screenshot / -o scratch/home.png
curl -s http://localhost:$(cat tests/fixtures/journal/health/convey.port)/
```

Notes:
* Agents won't execute without API keys — this is expected in worktrees
* Output artifacts go in `scratch/` (git-ignored)
* Service logs: `tests/fixtures/journal/health/<service>.log`

### File Locations
* **Entry Points**: `sol.py` `COMMANDS` dict
* **Test Fixtures**: `tests/fixtures/journal/` - complete mock journal
* **Live Logs**: `$JOURNAL_PATH/health/<service>.log`
* **Agent Personas**: `muse/*.md` (apps can add their own in `muse/`, see [docs/APPS.md](docs/APPS.md))
* **Generator Templates**: `muse/*.md` (apps can add their own in `muse/`, see [docs/APPS.md](docs/APPS.md))
* **Agent Skills**: `muse/*/SKILL.md` - symlinked to `.agents/skills/` and `.claude/skills/` via `make skills`, read https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices to create the best skills
* **Scratch Space**: `scratch/` - git-ignored local workspace

### Getting Help
* Run `sol` for status and CLI command list
* Check [docs/DOCTOR.md](docs/DOCTOR.md) for debugging and diagnostics
* Browse `docs/` for all subsystem documentation
* Review test files in `tests/` for usage examples
