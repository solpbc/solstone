# Environment

## Journal Path

The journal lives at `journal/` in the project root. `get_journal()` from `think.utils` returns the path — trust it unconditionally. Never set `_SOLSTONE_JOURNAL_OVERRIDE` from application code, service files, agent prompts, or subprocess environments. The env var exists exclusively for external use: test harnesses (`monkeypatch.setenv`) and Makefile sandboxes. If you think you need to override the journal path, you don't — fix the actual problem instead.

## Service Installation

`make install-service` installs solstone as a systemd user service (Linux) or launchd agent (macOS) with convey on port 5015. Override with `make install-service PORT=8000`. Managed via `sol service <install|start|stop|restart|status|logs>`.

## API Keys

Store API keys in `.env` file, never commit to repository.

## Error Handling & Logging

- Raise specific exceptions with clear messages
- Use logging module, not print statements
- Validate all external inputs (paths, owner data)
- Fail fast with clear errors - avoid silent failures

## Documentation

- Update README files for new functionality
- Code comments explain "why" not "what"
- Function signatures should include type hints; highlight gaps when touching older modules
- **All docs in `docs/` plus journal references in `talent/journal/`**: Browse `talent/journal/SKILL.md`, APPS.md, CORTEX.md, CALLOSUM.md, THINK.md, and more
- Each package has a README.md symlink pointing to its documentation in `docs/`.
- **App/UI work**: [docs/APPS.md](docs/APPS.md) is required reading before modifying `apps/`

## Git Practices

- **Git**: Small focused commits, descriptive branch names. Run git commands directly (not `git -C`) since you're already in the repo.

## Getting Help

- Run `sol` for status and CLI command list
- Check [docs/DOCTOR.md](docs/DOCTOR.md) for debugging and diagnostics
- Browse `docs/` for all subsystem documentation
- Review test files in `tests/` for usage examples
