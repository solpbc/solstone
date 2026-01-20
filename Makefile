# solstone Makefile
# Python-based AI-driven desktop journaling toolkit

.PHONY: install test test-apps test-app lint format check clean all update-prices

# Default target - install package in editable mode
all: install

# Marker file to track installation
.installed: pyproject.toml
	@echo "Installing package in editable mode..."
	pip install -e .
	@echo "Updating genai-prices to latest (for current model pricing)..."
	pip install --upgrade genai-prices
	@touch .installed

# Install package in editable mode
install: .installed

# Test environment - use fixtures journal for all tests
TEST_ENV = JOURNAL_PATH=fixtures/journal

# Run core tests (excluding integration and app tests)
test: .installed
	@echo "Running core tests..."
	$(TEST_ENV) pytest tests/ -q --cov=. --ignore=tests/integration

# Run core tests with verbose output
test-verbose: .installed
	@echo "Running core tests with verbose output..."
	$(TEST_ENV) pytest tests/ -v --cov=. --cov-report=term-missing --ignore=tests/integration

# Run app tests
test-apps: .installed
	@echo "Running app tests..."
	$(TEST_ENV) pytest apps/ -q

# Run app tests with verbose output
test-apps-verbose: .installed
	@echo "Running app tests with verbose output..."
	$(TEST_ENV) pytest apps/ -v

# Run specific app tests
test-app: .installed
	@if [ -z "$(APP)" ]; then \
		echo "Usage: make test-app APP=<app_name>"; \
		echo "Example: make test-app APP=todos"; \
		exit 1; \
	fi
	$(TEST_ENV) pytest apps/$(APP)/tests/ -v

# Run specific test file or pattern
test-only: .installed
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make test-only TEST=<test_file_or_pattern>"; \
		echo "Example: make test-only TEST=tests/test_utils.py"; \
		echo "Example: make test-only TEST=\"-k test_function_name\""; \
		exit 1; \
	fi
	$(TEST_ENV) pytest $(TEST)

# Run integration tests
test-integration: .installed
	@echo "Running integration tests..."
	$(TEST_ENV) pytest tests/integration/ -v --tb=short

# Run integration tests with coverage
test-integration-cov: .installed
	@echo "Running integration tests with coverage..."
	$(TEST_ENV) pytest tests/integration/ -v --cov=. --cov-report=term-missing

# Run specific integration test
test-integration-only: .installed
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make test-integration-only TEST=<test_file_or_pattern>"; \
		echo "Example: make test-integration-only TEST=test_api.py"; \
		exit 1; \
	fi
	$(TEST_ENV) pytest tests/integration/$(TEST)

# Run all tests (core + apps + integration)
test-all: .installed
	@echo "Running all tests (core + apps + integration)..."
	$(TEST_ENV) pytest tests/ -v --cov=. && $(TEST_ENV) pytest apps/ -v --cov=. --cov-append

# Auto-format code
format: .installed
	@echo "Formatting code with black and isort..."
	black .
	isort .

# Run all linting and formatting checks
lint: .installed
	@echo "Running linting checks..."
	@echo "=== Running black (check mode) ==="
	black --check . || true
	@echo ""
	@echo "=== Running isort (check mode) ==="
	isort --check-only . || true
	@echo ""
	@echo "=== Running flake8 ==="
	flake8 . || true
	@echo ""
	@echo "=== Running mypy ==="
	mypy . || true

# Run only flake8 linting
lint-flake8: .installed
	flake8 .

# Run only black formatting check
lint-black: .installed
	black --check .

# Run only isort import check
lint-isort: .installed
	isort --check-only .

# Run type checking with mypy
check: .installed
	@echo "Running type checking with mypy..."
	mypy .

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

# Clean everything and reinstall
clean-install: clean install

# Run continuous integration checks (what CI would run)
ci: lint test
	@echo "All CI checks passed!"

# Development workflow - format, lint, and test
check-all: format lint test
	@echo "All checks completed!"

# Watch for changes and run tests (requires pytest-watch)
watch:
	@command -v ptw >/dev/null 2>&1 || { echo "Installing pytest-watch..."; pip install pytest-watch; }
	ptw -- -q

# Generate coverage report (core + apps, excluding core integration tests)
coverage: .installed
	$(TEST_ENV) pytest tests/ --cov=. --cov-report=html --cov-report=term --ignore=tests/integration
	$(TEST_ENV) pytest apps/ --cov=. --cov-report=html --cov-report=term --cov-append
	@echo "Coverage report generated in htmlcov/index.html"

# Update dependencies to latest versions
update-deps:
	@echo "Updating dependencies to latest versions..."
	pip install --upgrade pip setuptools wheel
	pip install --upgrade -e .

# Update genai-prices to get latest model pricing data
# Run this when adding new models or if pricing tests fail
update-prices:
	@echo "Updating genai-prices to latest version..."
	pip install --upgrade genai-prices
	@echo "Done. Re-run tests to verify model pricing support."

# Show installed package versions
versions:
	@echo "=== Python version ==="
	python --version
	@echo ""
	@echo "=== Key package versions ==="
	@pip list | grep -E "^(pytest|black|flake8|mypy|isort|Flask|numpy|Pillow|openai|anthropic|google-genai)" || true

# Install pre-commit hooks (if using pre-commit)
pre-commit:
	@command -v pre-commit >/dev/null 2>&1 || { echo "Installing pre-commit..."; pip install pre-commit; }
	pre-commit install
	@echo "Pre-commit hooks installed!"

# Quick check before committing
pre-push: format lint-flake8 test
	@echo "Ready to push!"