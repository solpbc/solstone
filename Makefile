# solstone Makefile
# Python-based AI-driven desktop journaling toolkit

.PHONY: install uninstall test test-apps test-app test-only test-integration test-integration-only test-all format ci clean clean-install coverage watch versions update-deps update-prices pre-commit all

# Default target - install package in editable mode
all: install

# Virtual environment directory
VENV := .venv
VENV_BIN := $(VENV)/bin
PYTHON := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip

# User bin directory for symlink (standard location, usually already in PATH)
USER_BIN := $(HOME)/.local/bin

# Create virtual environment
$(VENV)/pyvenv.cfg:
	@echo "Creating virtual environment in $(VENV)..."
	python3 -m venv $(VENV)

# Marker file to track installation
.installed: pyproject.toml $(VENV)/pyvenv.cfg
	@echo "Installing package in isolated virtual environment..."
	$(PIP) install --upgrade pip
	@# Python 3.14+ needs onnxruntime from nightly (not yet on PyPI)
	@PY_MINOR=$$($(PYTHON) -c "import sys; print(sys.version_info.minor)"); \
	if [ "$$PY_MINOR" -ge 14 ]; then \
		echo "Python 3.14+ detected - installing onnxruntime from nightly feed..."; \
		$(PIP) install --pre --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime; \
	fi
	$(PIP) install -e .
	@echo "Updating genai-prices to latest (for current model pricing)..."
	$(PIP) install --upgrade genai-prices
	@echo "Installing Playwright browser for sol screenshot..."
	$(VENV_BIN)/playwright install chromium
	@mkdir -p $(USER_BIN)
	@ln -sf $(CURDIR)/$(VENV_BIN)/sol $(USER_BIN)/sol
	@echo ""
	@echo "Done! 'sol' command installed to $(USER_BIN)/sol"
	@if ! echo "$$PATH" | grep -q "$(USER_BIN)"; then \
		echo ""; \
		echo "NOTE: $(USER_BIN) is not in your PATH."; \
		echo "Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.):"; \
		echo "  export PATH=\"\$$HOME/.local/bin:\$$PATH\""; \
		echo ""; \
		echo "Or run sol directly: $(CURDIR)/$(VENV_BIN)/sol"; \
	fi
	@touch .installed

# Install package in editable mode with isolated venv
install: .installed

# Test environment - use fixtures journal for all tests
TEST_ENV = JOURNAL_PATH=fixtures/journal

# Venv tool shortcuts
PYTEST := $(VENV_BIN)/pytest
BLACK := $(VENV_BIN)/black
ISORT := $(VENV_BIN)/isort
FLAKE8 := $(VENV_BIN)/flake8
MYPY := $(VENV_BIN)/mypy

# Run core tests (excluding integration and app tests)
test: .installed
	@echo "Running core tests..."
	$(TEST_ENV) $(PYTEST) tests/ -q --cov=. --ignore=tests/integration

# Run app tests
test-apps: .installed
	@echo "Running app tests..."
	$(TEST_ENV) $(PYTEST) apps/ -q

# Run specific app tests
test-app: .installed
	@if [ -z "$(APP)" ]; then \
		echo "Usage: make test-app APP=<app_name>"; \
		echo "Example: make test-app APP=todos"; \
		exit 1; \
	fi
	$(TEST_ENV) $(PYTEST) apps/$(APP)/tests/ -v

# Run specific test file or pattern
test-only: .installed
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make test-only TEST=<test_file_or_pattern>"; \
		echo "Example: make test-only TEST=tests/test_utils.py"; \
		echo "Example: make test-only TEST=\"-k test_function_name\""; \
		exit 1; \
	fi
	$(TEST_ENV) $(PYTEST) $(TEST)

# Run integration tests
test-integration: .installed
	@echo "Running integration tests..."
	$(TEST_ENV) $(PYTEST) tests/integration/ -v --tb=short --timeout=20

# Run specific integration test
test-integration-only: .installed
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make test-integration-only TEST=<test_file_or_pattern>"; \
		echo "Example: make test-integration-only TEST=test_api.py"; \
		exit 1; \
	fi
	$(TEST_ENV) $(PYTEST) tests/integration/$(TEST) --timeout=20

# Run all tests (core + apps + integration)
test-all: .installed
	@echo "Running all tests (core + apps + integration)..."
	$(TEST_ENV) $(PYTEST) tests/ -v --cov=. && $(TEST_ENV) $(PYTEST) apps/ -v --cov=. --cov-append

# Auto-format code, then report any remaining issues
# Linting config: .flake8  |  Formatting config: pyproject.toml [tool.black] / [tool.isort]
format: .installed
	@echo "Formatting code with black and isort..."
	@$(BLACK) .
	@$(ISORT) .
	@echo ""
	@echo "Checking for remaining issues..."
	@FLAKE8_OK=true; MYPY_OK=true; \
	$(FLAKE8) . || FLAKE8_OK=false; \
	$(MYPY) . || MYPY_OK=false; \
	if $$FLAKE8_OK && $$MYPY_OK; then \
		echo ""; \
		echo "All clean!"; \
	else \
		echo ""; \
		echo "Issues above need manual fixes."; \
	fi

# Clean build artifacts and cache files
clean:
	@echo "Cleaning build artifacts and cache files..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage .mypy_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	rm -f .installed

# Uninstall - remove venv and sol symlink
uninstall: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@if [ -L $(USER_BIN)/sol ]; then \
		echo "Removing sol symlink from $(USER_BIN)..."; \
		rm -f $(USER_BIN)/sol; \
	fi

# Clean everything and reinstall
clean-install: uninstall install

# Run continuous integration checks (what CI would run)
ci: .installed
	@echo "Running CI checks..."
	@echo "=== Checking formatting ==="
	@$(BLACK) --check . || { echo "Run 'make format' to fix formatting"; exit 1; }
	@$(ISORT) --check-only . || { echo "Run 'make format' to fix imports"; exit 1; }
	@echo ""
	@echo "=== Running flake8 ==="
	@$(FLAKE8) . || exit 1
	@echo ""
	@echo "=== Running mypy ==="
	@$(MYPY) . || true
	@echo ""
	@echo "=== Running tests ==="
	@$(MAKE) test
	@echo ""
	@echo "All CI checks passed!"

# Watch for changes and run tests (requires pytest-watch)
watch: .installed
	@$(PIP) show pytest-watch >/dev/null 2>&1 || { echo "Installing pytest-watch..."; $(PIP) install pytest-watch; }
	$(VENV_BIN)/ptw -- -q

# Generate coverage report (core + apps, excluding core integration tests)
coverage: .installed
	$(TEST_ENV) $(PYTEST) tests/ --cov=. --cov-report=html --cov-report=term --ignore=tests/integration
	$(TEST_ENV) $(PYTEST) apps/ --cov=. --cov-report=html --cov-report=term --cov-append
	@echo "Coverage report generated in htmlcov/index.html"

# Update dependencies to latest versions
update-deps: .installed
	@echo "Updating dependencies to latest versions..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e .

# Update genai-prices to get latest model pricing data
# Run this when adding new models or if pricing tests fail
update-prices: .installed
	@echo "Updating genai-prices to latest version..."
	$(PIP) install --upgrade genai-prices
	@echo "Done. Re-run tests to verify model pricing support."

# Show installed package versions
versions: .installed
	@echo "=== Python version ==="
	$(PYTHON) --version
	@echo ""
	@echo "=== Key package versions ==="
	@$(PIP) list | grep -E "^(pytest|black|flake8|mypy|isort|Flask|numpy|Pillow|openai|anthropic|google-genai)" || true

# Install pre-commit hooks (if using pre-commit)
pre-commit: .installed
	@$(PIP) show pre-commit >/dev/null 2>&1 || { echo "Installing pre-commit..."; $(PIP) install pre-commit; }
	$(VENV_BIN)/pre-commit install
	@echo "Pre-commit hooks installed!"