# Project Structure

## Directory Layout

```text
solstone/
├── observe/        # Multimodal capture & AI analysis
├── think/          # Data post-processing, AI agents & orchestration
│   └── sol_cli.py  # Unified CLI entry point (run: sol <command>)
├── convey/         # Web app frontend & backend
├── apps/           # Convey app extensions (see docs/APPS.md)
├── talent/           # Agent/generator configs + Agent Skills (talent/*/SKILL.md)
├── tests/          # Pytest test suites + test fixtures under tests/fixtures/
├── docs/           # All documentation (*.md files)
├── AGENTS.md       # Development guidelines (this file)
├── CLAUDE.md       # Symlink to AGENTS.md for Claude Code
└── README.md       # Project overview
```

Each package has a README.md symlink pointing to its documentation in `docs/`.

## Package Organization

- **Python**: Requires Python 3.10+
- **Modules**: Each top-level folder is a Python package with `__init__.py` unless it is data-only (e.g., `tests/fixtures/`)
- **Imports**: Prefer absolute imports (e.g., `from think.utils import setup_cli`) whenever feasible
- **Entry Points**: Commands are registered in `think/sol_cli.py`'s `COMMANDS` dict (pyproject.toml just defines the `sol` entry point)
- **Journal**: Data stored under `journal/` at the project root; day content lives under `journal/chronicle/`
- **Calling**: When calling other modules as a separate process always use `sol <command>` and never call using `python -m ...` (e.g., use `sol indexer`, NOT `python -m think.indexer`)

## CLI Routing

`think/sol_cli.py`'s `COMMANDS` dict maps command names to module paths. The unified CLI is `sol`. Run `sol` to see status and available commands. `sol call` routes to `think/call.py`, which discovers `apps/*/call.py` Typer sub-apps and mounts them as subcommands.

## Agent & Skill Organization

`talent/*.md` stores agent personas and generator templates. Apps can add their own in `apps/*/talent/*.md`. Skills live at `talent/*/SKILL.md` and are symlinked into `journal/.agents/skills/` and `journal/.claude/skills/` via `make skills`.

## File Locations

- **Entry Points**: `think/sol_cli.py` `COMMANDS` dict
- **Test Fixtures**: `tests/fixtures/journal/` - complete mock journal
- **Live Logs**: `journal/health/<service>.log`
- **Agent Personas**: `talent/*.md` (apps can add their own in `talent/`, see [docs/APPS.md](docs/APPS.md))
- **Generator Templates**: `talent/*.md` (apps can add their own in `talent/`, see [docs/APPS.md](docs/APPS.md))
- **Agent Skills**: `talent/*/SKILL.md` - symlinked into `journal/.agents/skills/` and `journal/.claude/skills/` via `make skills`, read https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices to create the best skills
- **Scratch Space**: `scratch/` - git-ignored local workspace
