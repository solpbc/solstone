# Development Guidelines & Contribution Standards

This document provides comprehensive guidelines for contributing to Sunstone, whether you're an AI assistant, human developer, or automated system.

## 📋 Project Overview

**Sunstone** is a Python-based AI-driven desktop journaling toolkit that provides:

* **hear/** - System audio capture and AI-powered transcription
* **see/** - Screenshot capture and visual analysis with AI vision models
* **think/** - Data post-processing, summarization, and intelligent insights
* **dream/** - Web application for navigating and interacting with captured content

The project uses a modular architecture where each package can operate independently while sharing common utilities and data formats through the journal system.

---

## 🏗️ Project Structure

```
sunstone/
├── hear/           # Audio capture & transcription
├── see/            # Screenshot capture & image analysis
├── think/          # Data post-processing & AI analysis
│   ├── agents/     # Agent system prompts and configs
│   ├── indexer/    # Database indexing subsystem
│   └── topics/     # Topic extraction templates
├── dream/          # Web app frontend & backend
│   ├── static/     # JavaScript and CSS assets
│   ├── templates/  # Jinja2 HTML templates
│   └── views/      # Flask view modules
├── tests/          # Comprehensive pytest test suites
│   └── integration/# Integration test suite
├── fixtures/       # Test data and examples
│   └── journal/    # Mock journal structure for testing
├── Makefile        # Build and development automation
├── pyproject.toml  # Package configuration and dependencies
├── JOURNAL.md      # Journal directory structure documentation
├── README.md       # Project overview and quick start
├── CHANGELOG.md    # Version history & release notes
├── CRUMBS.md       # Crumb file format specification
├── CORTEX.md       # Agent system documentation
└── CLAUDE.md       # Development guidelines (this file)
```

### Package Organization

* **Modules**: Each top-level folder is a Python package with `__init__.py`
* **Imports**: Use absolute imports (e.g., `from hear.capture import record_audio`)
* **Entry Points**: Defined in `pyproject.toml` under `[project.scripts]`
* **Journal**: Data stored under `JOURNAL_PATH` environment variable location
* **Calling**: When calling other modules as a separate process always use their command name and never call using `python -m ...` (e.g., use `think-indexer`, NOT `python -m think.indexer`)

---

## 🏛️ Architecture & Data Flow

**Pipeline**: `hear/see` (capture) → JSON transcripts → `think` (analyze) → SQLite index → `dream` (web UI)

---

## 🔌 Critical Integration Points

* **JOURNAL_PATH**: Central to all operations - set via environment
* **Cortex**: WebSocket server spawns agents, captures stdout events → `.jsonl` files  
* **MCP Tools**: HTTP server provides journal search/retrieval (check `agents/mcp.uri`)
* **Flask/Async**: Use `dream.cortex_utils` for sync wrappers in Flask views

---

## 🤖 Agent System Patterns

```python
# Async (in think/ modules)
from think.cortex_client import run_agent
result = await run_agent(prompt, persona="default", backend="openai")

# Sync (in dream/ views)  
from dream.cortex_utils import run_agent_via_cortex
result = run_agent_via_cortex(prompt, on_event=lambda e: push_server.push(e))
```

**Event flow**: Agent stdout → Cortex → `.jsonl` + WebSocket broadcast

**Essential utilities**: Always check `think.utils` for common operations (setup_cli, day_path, parse_time_range, etc.) and `think.crumbs` for dependency tracking.

---

## 🧪 Testing with Fixtures

```python
# Use comprehensive mock journal data for testing
os.environ["JOURNAL_PATH"] = "fixtures/journal"
# Now all journal operations work with test data
```

---

## 💻 Coding Standards & Style

### Language & Tools
* **Python Version**: 3.9+ required
* **Code Formatting**: 
  - Black (`make format` or `black .`)
  - isort (`make format` or `isort .`)
  - Settings configured in `pyproject.toml`
* **Linting & Type Checking**:
  - flake8 (`make lint-flake8` or `flake8 .`)
  - mypy (`make check` or `mypy .`)
  - All new code must pass these checks

### Naming Conventions
* **Modules & Packages**: `snake_case` (e.g., `audio_utils.py`)
* **Classes**: `PascalCase` (e.g., `AudioProcessor`)
* **Functions & Variables**: `snake_case` (e.g., `process_audio()`)
* **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_BUFFER_SIZE`)
* **Private Members**: Leading underscore (e.g., `_internal_method()`)

### Code Organization
* **Docstrings**: Google or NumPy style with parameter/return descriptions
* **Imports**: 
  - Absolute imports only
  - Grouped in order: standard library, third-party, local
  - One import per line for clarity
* **File Structure**:
  - Constants at top
  - Helper functions before main functions
  - Classes after functions
  - Main/CLI code at bottom

---

## 🧪 Testing & Quality Assurance

### Test Framework
* **Framework**: pytest with coverage reporting
* **Structure**: Tests in `tests/` mirroring source structure
  - Unit tests in `tests/` root directory
  - Integration tests in `tests/integration/` subdirectory
* **Naming**: Test files prefixed with `test_`, functions with `test_`
* **Fixtures**: Shared fixtures in `tests/conftest.py`
* **Test Data**: The `fixtures/journal/` directory contains a comprehensive mock journal structure that mimics real journal data for testing, including sample domains, matters, agents, inbox messages, and indexed data

### Running Tests
```bash
# Quick test run
make test

# Verbose with coverage
make test-verbose

# Specific test file or pattern
make test-only TEST=tests/test_utils.py
make test-only TEST="-k test_function_name"

# Generate HTML coverage report
make coverage
```

### Development Workflow
```bash
# Before committing
make check-all  # Formats, lints, and tests

# Or individually
make format     # Auto-format code
make lint       # Check code quality
make test       # Run tests

# Clean up
make clean      # Remove build artifacts
```

### Continuous Integration
* Tests must pass for all changes
* Maintain or improve code coverage
* Fix linting issues before merging
* Update tests when adding features

---

## 📝 Important Development Notes

### Environment Management
* **JOURNAL_PATH**: MUST call `setup_cli()` first in any CLI tool, or manually use `load_dotenv()` - then available via environment
* **API Keys**: Store in `.env` file, never commit to repository
* **Configuration**: Use `python-dotenv` for environment variables

### Error Handling
* **Exceptions**: Raise specific exceptions with clear messages
* **Validation**: Validate all external inputs (paths, user data)
* **Logging**: Use Python's logging module, not print statements
* **Silent Failures**: Avoid them - fail fast with clear errors

### Documentation
* **CHANGELOG.md**: Update for all features/fixes following format
* **README Files**: Update package READMEs for new functionality
* **Code Comments**: Explain "why" not "what" - code should be self-documenting
* **Type Hints**: Use type hints for function signatures

---

## 📦 Dependencies Management

### Adding Dependencies
* **Minimize Dependencies**: Use standard library when possible
* **Production Dependencies**: Add to `dependencies` in `pyproject.toml`
* **Development Dependencies**: Add to `[project.optional-dependencies]` dev section
* **Optional Features**: Add to appropriate optional dependency group (e.g., `full`)

### Installing Dependencies
```bash
# Basic package installation
make install

# With all optional dependencies
make full

# Development environment
make dev
```

---

## 🎯 Development Principles

### Code Quality
* **DRY (Don't Repeat Yourself)**: Extract common logic into utilities
* **KISS (Keep It Simple)**: Prefer simple, readable solutions
* **YAGNI (You Aren't Gonna Need It)**: Don't over-engineer
* **Single Responsibility**: Functions/classes should do one thing well

### Best Practices
* **Conciseness**: Write clear, concise code without sacrificing readability
* **Maintainability**: Structure code for easy maintenance and extension
* **Performance**: Profile before optimizing, focus on bottlenecks
* **Security**: Never expose secrets, validate inputs, sanitize outputs

### Git Workflow
* **Commits**: Small, focused commits with clear messages
* **Branches**: Feature branches from main, descriptive names
* **Pull Requests**: Include tests, update docs, pass CI checks
* **Reviews**: Address feedback promptly and thoroughly

---

## 🚀 Quick Reference

### Common Commands
```bash
# Development setup
make dev            # Install with dev dependencies
make test          # Run tests
make format        # Format code
make lint          # Check code quality

# Before pushing
make check-all     # Format, lint, and test

# Cleanup
make clean         # Remove artifacts
make clean-install # Clean and reinstall
```

### File Locations
* **Entry Points**: `pyproject.toml` `[project.scripts]`
* **Test Fixtures**: `fixtures/` directory
* **Logs**: `logs/` directory (gitignored)
* **Journal Data**: Path from `JOURNAL_PATH` env var
* **Config**: `.env` file in project root

### Getting Help
* Run `make help` for available Make targets
* Run `sunstone` for CLI command list
* Check package READMEs for detailed usage
* Review test files for usage examples