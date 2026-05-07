# Environment

## Journal Path

`get_journal()` / `get_journal_info()` in `solstone.think.utils` are the canonical journal resolvers. Trust them unconditionally.

Resolver order (with the source label `get_journal_info()` returns):

1. `SOLSTONE_JOURNAL` env var, when set and non-empty → `"env"`
2. `~/.config/solstone/config.toml`, when it has a non-empty `journal = "..."` key → `"config"`
3. source-tree fallback: `<project_root>/journal` when both `<project_root>/pyproject.toml` and `<project_root>/.git` exist → `"source"`
4. built-in default: `~/Documents/journal` → `"default"`

`get_journal_info()` no longer raises — there is always a resolved path. `get_journal()` raises `SolstoneNotConfigured` only when `os.makedirs` on the resolved path fails.

Who sets `SOLSTONE_JOURNAL`:

- Installed runs: the managed wrapper at `~/.local/bin/sol`
- Unit tests: the `set_test_journal_path` autouse fixture in `tests/conftest.py`
- Makefile sandboxes: explicit per-command env injection in `make sandbox` / verify targets

Who must **not** set it:

- application code
- service files
- agent prompts
- ad hoc subprocess environments spawned by app code

If you think you need to set `SOLSTONE_JOURNAL` from application code, fix the actual resolution problem instead.

## Service Installation

From a fresh source checkout, `.venv/bin/sol setup` installs the managed wrapper at `~/.local/bin/sol`, then installs solstone as a systemd user service (Linux) or launchd agent (macOS) with convey on port 5015. After the first run, the wrapper lets you use `sol setup` from anywhere. Override with `.venv/bin/sol setup --port 8000` on the first run or `sol setup --port 8000` after the wrapper exists. Service installation runs only on source-checkout installs in v1; packaged installs skip the service step.

Installed services invoke `~/.local/bin/sol`. They do **not** write `SOLSTONE_JOURNAL` into the service env block; the wrapper exports it before execing the venv `sol`.

Use:

- `sol config show` to display the resolved journal path, user-facing source label, and wrapper status
- `sol config journal <path>` to atomically rewrite the wrapper's embedded journal path
- `sol service <install|start|stop|restart|status|logs>` for service lifecycle management

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
- **All docs in `docs/` plus journal references in `solstone/talent/journal/`**: Browse `solstone/talent/journal/SKILL.md`, APPS.md, CORTEX.md, CALLOSUM.md, THINK.md, and more
- Each package has a README.md symlink pointing to its documentation in `docs/`.
- **App/UI work**: [docs/APPS.md](docs/APPS.md) is required reading before modifying `solstone/apps/`

## Git Practices

- **Git**: Small focused commits, descriptive branch names. Run git commands directly (not `git -C`) since you're already in the repo.

## Getting Help

- Run `sol` for status and CLI command list
- Check [docs/DOCTOR.md](docs/DOCTOR.md) for debugging and diagnostics
- Browse `docs/` for all subsystem documentation
- Review test files in `tests/` for usage examples
