# Development Guidelines & Contribution Standards

This document provides comprehensive guidelines for contributing to Sunstone, whether you're an AI assistant, human developer, or automated system.

## 📋 Project Overview

**Sunstone** is a Python-based AI-driven desktop journaling toolkit that provides:

* **observe/** - Multimodal capture (audio + visual) and AI-powered analysis
* **think/** - Data post-processing, summarization, and intelligent insights
* **convey/** - Web application for navigating and interacting with captured content
* **muse/** - AI agent system and MCP tooling

The project uses a modular architecture where each package can operate independently while sharing common utilities and data formats through the journal system.

---

## 🔑 Key Concepts

Understanding these core concepts is essential for working with Sunstone:

* **Journal**: Central data structure organized as `JOURNAL_PATH/YYYYMMDD/` directories. All captured data, transcripts, and analysis artifacts are stored here. See **JOURNAL.md** for detailed structure.

* **Facets**: Project/context organization system (e.g., "work", "personal", "acme"). Facets group related content and provide scoped views of entities, tasks, and activities.

* **Entities**: Extracted information (people, projects, concepts) tracked over time across transcripts and interactions. Entities are associated with facets and enable semantic navigation.

* **Agents**: AI processors with configurable personas that analyze content, extract insights, and respond to queries. Managed by Cortex. See **CORTEX.md** for agent system architecture.

* **Callosum**: Message bus that enables asynchronous communication between components (observe, think, convey, cortex). Uses file-based persistence. See **CALLOSUM.md** for protocol details.

* **Indexer**: Builds and maintains SQLite database from journal data, enabling fast search and retrieval. Indexes transcripts, summaries, entities, and events.

* **Crumbs**: JSON transcript format with timestamps, speaker attribution, and metadata. See **CRUMBS.md** for specification.

---

## 🏗️ Project Structure

```
sunstone/
├── observe/        # Multimodal capture & AI analysis
├── think/          # Data post-processing & AI analysis
│   ├── indexer/    # Database indexing subsystem
│   └── topics/     # Topic extraction templates
├── convey/         # Web app frontend & backend
│   ├── static/     # JavaScript and CSS assets
│   ├── templates/  # Jinja2 HTML templates
│   └── views/      # Flask view modules
├── muse/           # AI agent system and MCP tooling
│   └── agents/     # Agent system prompts and configs
├── tests/          # Comprehensive pytest test suites
│   └── integration/# Integration test suite
├── fixtures/       # Test data and examples
│   └── journal/    # Mock journal structure for testing
├── Makefile        # Build and development automation
├── pyproject.toml  # Package configuration and dependencies
├── JOURNAL.md      # Journal directory structure documentation
├── README.md       # Project overview and quick start
├── CRUMBS.md       # Crumb file format specification
├── CORTEX.md       # Agent system documentation
├── CALLOSUM.md     # Callosum connection system documentation
└── AGENTS.md       # Development guidelines (this file)
```

### Package Organization

* **Modules**: Each top-level folder is a Python package with `__init__.py`
* **Imports**: Use absolute imports (e.g., `from think.utils import setup_cli`)
* **Entry Points**: Defined in `pyproject.toml` under `[project.scripts]` - see this file for the full list of available commands
* **Journal**: Data stored under `JOURNAL_PATH` environment variable location always loaded from .env
* **Calling**: When calling other modules as a separate process always use their command name and never call using `python -m ...` (e.g., use `think-indexer`, NOT `python -m think.indexer`)

---

## 🏛️ Architecture & Data Flow

**Core Pipeline**: `observe` (capture) → JSON transcripts → `think` (analyze) → SQLite index → `convey` (web UI)

**Data Organization**:
* Everything organized under `JOURNAL_PATH/YYYYMMDD/` daily directories
* Facets provide project-scoped organization and filtering
* Entities are extracted from transcripts and tracked across time
* Indexer builds SQLite database for fast search and retrieval

**Component Communication**:
* Callosum message bus enables async communication between services
* Cortex orchestrates agent execution, managing requests and event distribution
* Agents process via `muse-agents` command with persona configurations

**Main Entry Points**:
* `observe-gnome` - Multimodal capture (requires Linux/GNOME)
* `think-indexer` - Build searchable database
* `convey` - Launch web UI
* `muse-cortex` - Agent system server
* See `pyproject.toml` `[project.scripts]` for complete command list

---

## 🧪 Testing with Fixtures

```python
# Use comprehensive mock journal data for testing
os.environ["JOURNAL_PATH"] = "fixtures/journal"
# Now all journal operations work with test data
```

The `fixtures/journal/` directory contains a complete mock journal structure with sample facets, agents, transcripts, and indexed data for testing.

---

## 💻 Coding Standards & Style

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
* **Imports**: Absolute only, grouped (stdlib, third-party, local), one per line
* **Docstrings**: Google or NumPy style with parameter/return descriptions
* **Type Hints**: Required on function signatures
* **File Structure**: Constants → helpers → classes → main/CLI

---

## 🧪 Testing & Quality Assurance

### Test Structure
* **Framework**: pytest with coverage reporting
* **Unit Tests**: `tests/` root directory
  - Fast, no external API calls
  - Use `fixtures/journal/` mock data
  - Test individual functions and modules
* **Integration Tests**: `tests/integration/` subdirectory
  - Test real backends (Anthropic, OpenAI, Google)
  - Require API keys in `.env`
  - Test end-to-end workflows
* **Naming**: Files `test_*.py`, functions `test_*`
* **Fixtures**: Shared fixtures in `tests/conftest.py`

### Running Tests
```bash
# Unit tests (fast)
make test                    # Quick run
make test-verbose           # With coverage details
make test-only TEST=path    # Specific test

# Integration tests (require API keys)
make test-integration
make test-integration-only TEST=name

# Coverage report
make coverage               # Generates htmlcov/index.html

# UI testing
make screenshot VIEW=<route>  # Capture Convey view screenshot
```

### Development Workflow
```bash
make format      # Auto-format code
make lint        # Check code quality
make test        # Run unit tests
make check-all   # Format, lint, and test (run before commit)
```

---

## 📝 Important Development Notes

### Environment Management
* **JOURNAL_PATH**: MUST call `setup_cli()` first in any CLI tool, or manually use `load_dotenv()` - then available via environment. To find current path: `grep JOURNAL_PATH .env`
* **API Keys**: Store in `.env` file, never commit to repository
* **Entry Points**: Use command names (e.g., `think-indexer`) NOT `python -m ...`

### Error Handling & Logging
* Raise specific exceptions with clear messages
* Use logging module, not print statements
* Validate all external inputs (paths, user data)
* Fail fast with clear errors - avoid silent failures

### Documentation & References
* Update README files for new functionality
* Code comments explain "why" not "what"
* Type hints required on function signatures
* **See subsystem docs**: JOURNAL.md, CORTEX.md, CALLOSUM.md, CRUMBS.md

---

## 📦 Dependencies Management

* **Minimize Dependencies**: Use standard library when possible
* **Production**: Add to `dependencies` in `pyproject.toml`
* **Development**: Add to `[project.optional-dependencies]` dev section
* **Installation**: `make install` (basic), `make dev` (with dev tools), `make full` (all optional)

---

## 🎯 Development Principles

* **DRY, KISS, YAGNI**: Extract common logic, prefer simple solutions, don't over-engineer
* **Single Responsibility**: Functions/classes do one thing well
* **Conciseness & Maintainability**: Clear code over clever code
* **Security**: Never expose secrets, validate/sanitize all inputs
* **Performance**: Profile before optimizing
* **Git**: Small focused commits, descriptive branch names

---

## 🚀 Quick Reference

### Common Commands
```bash
# Development setup
make dev            # Install with dev dependencies
make test          # Run unit tests
make format        # Format code
make lint          # Check code quality

# Testing & Debugging
make test-integration         # Run integration tests
make screenshot VIEW=<route>  # Capture Convey view screenshot
make coverage                 # Generate coverage report

# Before pushing
make check-all     # Format, lint, and test

# Cleanup
make clean         # Remove artifacts
make clean-install # Clean and reinstall
```

### File Locations
* **Entry Points**: `pyproject.toml` `[project.scripts]`
* **Test Fixtures**: `fixtures/journal/` - complete mock journal
* **Live Logs**: Active services have logs available at `$JOURNAL_PATH/health/<service>.log`
* **Journal Data**: Path from `JOURNAL_PATH` env var (set in `.env`)
* **Config**: `.env` file in project root
* **Agent Personas**: `muse/agents/*.txt` and `*.json`
* **Topic Templates**: `think/topics/*.txt` and `*.json`

### Getting Help
* Run `make help` for available Make targets
* Run `sunstone` for CLI command list
* Check **JOURNAL.md**, **CORTEX.md**, **CALLOSUM.md**, **CRUMBS.md** for subsystem details
* Review test files in `tests/` for usage examples
