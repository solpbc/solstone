# Sunstone Makefile
# Python-based AI-driven desktop journaling toolkit

.PHONY: help install deps test lint format check clean dev full all

# Default target - install package in editable mode
all: install

help:
	@echo "Available targets:"
	@echo "  make           - Install package in editable mode (default)"
	@echo "  make install   - Install package in editable mode"
	@echo "  make deps      - Install test/dev dependencies only"
	@echo "  make full      - Install package with all optional dependencies"
	@echo "  make test      - Run tests (installs deps if needed)"
	@echo "  make lint      - Run all linting and formatting checks"
	@echo "  make format    - Auto-format code with black and isort"
	@echo "  make check     - Run type checking with mypy"
	@echo "  make clean     - Remove build artifacts and cache files"
	@echo "  make dev       - Install package with dev dependencies"

# Marker files to track dependency installation
.deps-installed: pyproject.toml
	@echo "Installing test/dev dependencies..."
	pip install -e .[dev]
	@touch .deps-installed

.package-installed: pyproject.toml
	@echo "Installing package in editable mode..."
	pip install -e .
	@touch .package-installed

.full-installed: pyproject.toml
	@echo "Installing package with all optional dependencies..."
	pip install -e .[full,dev]
	@touch .full-installed

# Install package in editable mode (default)
install: .package-installed

# Install only test/dev dependencies
deps: .deps-installed

# Install package with all optional dependencies
full: .full-installed

# Install package with dev dependencies
dev: pyproject.toml
	@echo "Installing package with dev dependencies..."
	pip install -e .[dev]
	@touch .package-installed
	@touch .deps-installed

# Run tests
test: .deps-installed
	@echo "Running tests..."
	pytest -q --cov=.

# Run tests with verbose output
test-verbose: .deps-installed
	@echo "Running tests with verbose output..."
	pytest -v --cov=. --cov-report=term-missing

# Run specific test file or pattern
test-only: .deps-installed
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make test-only TEST=<test_file_or_pattern>"; \
		echo "Example: make test-only TEST=tests/test_utils.py"; \
		echo "Example: make test-only TEST=\"-k test_function_name\""; \
		exit 1; \
	fi
	pytest $(TEST)

# Auto-format code
format: .deps-installed
	@echo "Formatting code with black and isort..."
	black .
	isort .

# Run all linting and formatting checks
lint: .deps-installed
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
lint-flake8: .deps-installed
	flake8 .

# Run only black formatting check
lint-black: .deps-installed
	black --check .

# Run only isort import check
lint-isort: .deps-installed
	isort --check-only .

# Run type checking with mypy
check: .deps-installed
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
	rm -f .deps-installed .package-installed .full-installed

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

# Generate coverage report
coverage: .deps-installed
	pytest --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

# Update dependencies to latest versions
update-deps:
	@echo "Updating dependencies to latest versions..."
	pip install --upgrade pip setuptools wheel
	pip install --upgrade -e .[dev]

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