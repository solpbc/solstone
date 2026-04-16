# solstone Makefile
# Python-based AI-driven desktop journaling toolkit

.PHONY: install uninstall test test-apps test-app test-only test-integration test-integration-only test-all format format-check ci clean clean-install coverage watch versions update update-prices pre-commit skills dev all sail sandbox sandbox-stop install-pinchtab verify-browser update-browser-baselines review verify-api update-api-baselines install-service uninstall-service

# Default target - install package in editable mode
all: install

# Virtual environment directory
VENV := .venv
VENV_BIN := $(VENV)/bin
PYTHON := $(VENV_BIN)/python

# Require uv
UV := $(shell command -v uv 2>/dev/null)
ifndef UV
$(error uv is not installed. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh)
endif

# Node — add nvm bin dir to PATH if npx isn't already available
NVM_BIN := $(lastword $(wildcard $(HOME)/.nvm/versions/node/*/bin))
ifneq ($(NVM_BIN),)
export PATH := $(NVM_BIN):$(PATH)
endif

# User bin directory for symlink (standard location, usually already in PATH)
USER_BIN := $(HOME)/.local/bin

# Marker file to track installation
.installed: pyproject.toml uv.lock
	@echo "Installing package with uv..."
	$(UV) sync
	@# Python 3.14+ needs onnxruntime from nightly (not yet on PyPI)
	@PY_MINOR=$$($(PYTHON) -c "import sys; print(sys.version_info.minor)"); \
	if [ "$$PY_MINOR" -ge 14 ]; then \
		echo "Python 3.14+ detected - installing onnxruntime from nightly feed..."; \
		$(UV) pip install --pre --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime; \
	fi
	@echo "Installing Playwright browser for sol screenshot..."
	$(VENV_BIN)/playwright install chromium
	@if [ -d .git ]; then \
		mkdir -p $(USER_BIN); \
		ln -sf $(CURDIR)/$(VENV_BIN)/sol $(USER_BIN)/sol; \
		echo ""; \
		echo "Done! 'sol' command installed to $(USER_BIN)/sol"; \
		if ! echo "$$PATH" | grep -q "$(USER_BIN)"; then \
			echo ""; \
			echo "NOTE: $(USER_BIN) is not in your PATH."; \
			echo "Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.):"; \
			echo "  export PATH=\"\$$HOME/.local/bin:\$$PATH\""; \
			echo ""; \
			echo "Or run sol directly: $(CURDIR)/$(VENV_BIN)/sol"; \
		fi; \
	else \
		echo ""; \
		echo "Done! (worktree detected, skipping ~/.local/bin/sol symlink)"; \
	fi
	@$(MAKE) --no-print-directory skills
	@if [ -d .git ] && [ -f skills/solstone/SKILL.md ]; then \
		echo "Installing solstone skill user-wide..."; \
		npx skills add ./skills/solstone -g -a claude-code -y; \
	fi
	@touch .installed

# Generate lock file if missing
uv.lock: pyproject.toml
	$(UV) lock

# Install package in editable mode with isolated venv
install: .installed

# Directories where AI coding agents look for skills
SKILL_DIRS := .agents/skills .claude/skills

# Discover SKILL.md files in talent/ and apps/*/talent/, symlink into agent skill dirs
skills:
	@# Collect all skill directories (containing SKILL.md)
	@SKILLS=""; \
	for skill_md in talent/*/SKILL.md apps/*/talent/*/SKILL.md; do \
		[ -f "$$skill_md" ] || continue; \
		skill_dir=$$(dirname "$$skill_md"); \
		skill_name=$$(basename "$$skill_dir"); \
		if echo "$$SKILLS" | grep -qw "$$skill_name"; then \
			echo "Error: duplicate skill name '$$skill_name' found in $$skill_dir" >&2; \
			echo "Each skill directory name must be unique across talent/ and apps/*/talent/." >&2; \
			exit 1; \
		fi; \
		SKILLS="$$SKILLS $$skill_name"; \
	done; \
	for dir in $(SKILL_DIRS); do \
		mkdir -p "$$dir"; \
		for link in "$$dir"/*; do \
			[ -L "$$link" ] && rm -f "$$link"; \
		done; \
	done; \
	count=0; \
	for skill_md in talent/*/SKILL.md apps/*/talent/*/SKILL.md; do \
		[ -f "$$skill_md" ] || continue; \
		skill_dir=$$(dirname "$$skill_md"); \
		skill_name=$$(basename "$$skill_dir"); \
		for dir in $(SKILL_DIRS); do \
			ln -sf "../../$$skill_dir" "$$dir/$$skill_name"; \
		done; \
		count=$$((count + 1)); \
	done; \
	if [ "$$count" -gt 0 ]; then \
		echo "Linked $$count skill(s) into $(SKILL_DIRS)"; \
	fi
	@$(PYTHON) scripts/generate_agents_md.py

# Start local dev stack against fixture journal (no observers, no daily processing)
dev: .installed
	$(TEST_ENV) PATH=$(CURDIR)/$(VENV_BIN):$$PATH $(VENV_BIN)/sol supervisor 0 --no-observers --no-daily

# Restart solstone service (noop in dev mode)
sail: .installed
	$(VENV_BIN)/sol service restart --if-installed

# Start sandbox stack: fixture copy + background supervisor + readiness wait
sandbox: .installed
	@# Fail if sandbox already running
	@if [ -f .sandbox.pid ] && kill -0 $$(cat .sandbox.pid) 2>/dev/null; then \
		echo "Sandbox already running (PID $$(cat .sandbox.pid))"; \
		echo "Run 'make sandbox-stop' first."; \
		exit 1; \
	fi
	@# Clean up stale state from a previous crashed sandbox
	@if [ -f .sandbox.journal ]; then \
		rm -rf "$$(cat .sandbox.journal)" 2>/dev/null; \
		rm -f .sandbox.pid .sandbox.journal; \
	fi
	@# Copy fixtures to temp dir
	@SANDBOX_JOURNAL=$$(mktemp -d /tmp/solstone-sandbox-XXXXXX); \
	cp -r tests/fixtures/journal/* "$$SANDBOX_JOURNAL/"; \
	echo "$$SANDBOX_JOURNAL" > .sandbox.journal; \
	echo "Sandbox journal: $$SANDBOX_JOURNAL"; \
	# Boot supervisor in background \
	_SOLSTONE_JOURNAL_OVERRIDE="$$SANDBOX_JOURNAL" PATH=$(CURDIR)/$(VENV_BIN):$$PATH \
		$(VENV_BIN)/sol supervisor 0 --no-observers --no-daily \
		> "$$SANDBOX_JOURNAL/health/supervisor.log" 2>&1 & \
	echo $$! > .sandbox.pid; \
	echo "Supervisor PID: $$(cat .sandbox.pid)"; \
	# Poll for readiness \
	echo "Waiting for services..."; \
	READY=false; \
	for i in $$(seq 1 20); do \
		if _SOLSTONE_JOURNAL_OVERRIDE="$$SANDBOX_JOURNAL" $(VENV_BIN)/sol health > /dev/null 2>&1; then \
			READY=true; \
			break; \
		fi; \
		sleep 1; \
	done; \
	if [ "$$READY" = "false" ]; then \
		echo "Readiness timeout - killing supervisor"; \
		kill $$(cat .sandbox.pid) 2>/dev/null || true; \
		rm -rf "$$SANDBOX_JOURNAL" .sandbox.pid .sandbox.journal; \
		exit 1; \
	fi; \
	CONVEY_PORT=$$(cat "$$SANDBOX_JOURNAL/health/convey.port" 2>/dev/null); \
	echo ""; \
	echo "Sandbox is ready!"; \
	echo "  Convey: http://localhost:$$CONVEY_PORT/"; \
	echo "  Journal: $$SANDBOX_JOURNAL"; \
	echo "  Stop:   make sandbox-stop"

# Stop sandbox: terminate supervisor, clean up temp dir and state files
sandbox-stop:
	@if [ ! -f .sandbox.pid ]; then \
		echo "No sandbox running."; \
		exit 0; \
	fi; \
	PID=$$(cat .sandbox.pid); \
	echo "Stopping supervisor (PID $$PID)..."; \
	kill "$$PID" 2>/dev/null || true; \
	# Wait up to 5s for clean shutdown \
	for i in $$(seq 1 10); do \
		kill -0 "$$PID" 2>/dev/null || break; \
		sleep 0.5; \
	done; \
	kill -9 "$$PID" 2>/dev/null || true; \
	if [ -f .sandbox.journal ]; then \
		SANDBOX_JOURNAL=$$(cat .sandbox.journal); \
		rm -rf "$$SANDBOX_JOURNAL"; \
		echo "Removed $$SANDBOX_JOURNAL"; \
	fi; \
	rm -f .sandbox.pid .sandbox.journal; \
	echo "Sandbox stopped."

# Verify API baselines against running sandbox
verify-api: .installed
	@echo "Verifying API baselines (sandbox)..."
	@$(MAKE) sandbox
	@SANDBOX_JOURNAL=$$(cat .sandbox.journal); \
	CONVEY_PORT=$$(cat "$$SANDBOX_JOURNAL/health/convey.port"); \
	RESULT=0; \
	_SOLSTONE_JOURNAL_OVERRIDE="$$SANDBOX_JOURNAL" $(VENV_BIN)/python tests/verify_api.py verify --base-url "http://localhost:$$CONVEY_PORT" || RESULT=$$?; \
	$(MAKE) sandbox-stop; \
	exit $$RESULT

# Regenerate all API baseline files from the deterministic Flask test-client path
update-api-baselines: .installed
	@echo "Updating API baselines (test client)..."
	@$(VENV_BIN)/python tests/verify_api.py update


# Install pinchtab browser automation tool
install-pinchtab:
	@if command -v pinchtab >/dev/null 2>&1; then \
		echo "pinchtab already installed: $$(pinchtab --version 2>/dev/null || echo 'unknown')"; \
	else \
		echo "Installing pinchtab..."; \
		curl -fsSL https://pinchtab.com/install.sh | bash; \
	fi

# Run browser scenarios against sandbox
verify-browser: .installed
	@echo "Running browser scenarios (sandbox)..."
	@$(MAKE) sandbox
	@SANDBOX_JOURNAL=$$(cat .sandbox.journal); \
	CONVEY_PORT=$$(cat "$$SANDBOX_JOURNAL/health/convey.port"); \
	RESULT=0; \
	$(VENV_BIN)/python tests/verify_browser.py verify --base-url "http://localhost:$$CONVEY_PORT" || RESULT=$$?; \
	$(MAKE) sandbox-stop; \
	exit $$RESULT

# Re-capture all browser baseline screenshots
update-browser-baselines: .installed
	@echo "Updating browser baselines (sandbox)..."
	@$(MAKE) sandbox
	@SANDBOX_JOURNAL=$$(cat .sandbox.journal); \
	CONVEY_PORT=$$(cat "$$SANDBOX_JOURNAL/health/convey.port"); \
	RESULT=0; \
	$(VENV_BIN)/python tests/verify_browser.py update --base-url "http://localhost:$$CONVEY_PORT" || RESULT=$$?; \
	$(MAKE) sandbox-stop; \
	exit $$RESULT

# Full product verification: API baselines + browser scenarios
review: .installed
	@command -v pinchtab >/dev/null 2>&1 || { \
		echo "pinchtab is required for browser verification."; \
		echo "Run 'make install-pinchtab' to install it."; \
		exit 1; \
	}
	@echo "=== Starting review ==="
	@$(MAKE) sandbox
	@SANDBOX_JOURNAL=$$(cat .sandbox.journal); \
	CONVEY_PORT=$$(cat "$$SANDBOX_JOURNAL/health/convey.port"); \
	BASE_URL="http://localhost:$$CONVEY_PORT"; \
	RESULT_API=0; \
	RESULT_BROWSER=0; \
	echo ""; \
	echo "=== API baseline verification ==="; \
	_SOLSTONE_JOURNAL_OVERRIDE="$$SANDBOX_JOURNAL" $(VENV_BIN)/python tests/verify_api.py verify --base-url "$$BASE_URL" || RESULT_API=$$?; \
	echo ""; \
	echo "=== Browser scenario verification ==="; \
	$(VENV_BIN)/python tests/verify_browser.py verify --base-url "$$BASE_URL" || RESULT_BROWSER=$$?; \
	echo ""; \
	echo "=== Stopping sandbox ==="; \
	$(MAKE) sandbox-stop; \
	echo ""; \
	echo "=== Review Summary ==="; \
	if [ $$RESULT_API -eq 0 ]; then \
		echo "  API:     PASS"; \
	else \
		echo "  API:     FAIL"; \
	fi; \
	if [ $$RESULT_BROWSER -eq 0 ]; then \
		echo "  Browser: PASS"; \
	else \
		echo "  Browser: FAIL"; \
	fi; \
	echo ""; \
	if [ $$RESULT_API -eq 0 ] && [ $$RESULT_BROWSER -eq 0 ]; then \
		echo "Review: ALL PASS"; \
	else \
		echo "Review: FAIL"; \
		exit 1; \
	fi

# Test environment - use fixtures journal for all tests
TEST_ENV = _SOLSTONE_JOURNAL_OVERRIDE=tests/fixtures/journal

# Venv tool shortcuts
PYTEST := $(VENV_BIN)/pytest
RUFF := $(VENV_BIN)/ruff
MYPY := $(VENV_BIN)/mypy

# Check formatting without modifying files — gates `make test`
format-check: .installed
	@$(RUFF) format --check . || { echo "Run 'make format' to fix formatting"; exit 1; }

# Run core tests (excluding integration and app tests)
test: .installed format-check
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

# Auto-format and fix code, then report any remaining issues
format: .installed
	@echo "Formatting and fixing code with ruff..."
	@$(RUFF) format .
	@$(RUFF) check --fix .
	@echo ""
	@echo "Checking for remaining issues..."
	@RUFF_OK=true; MYPY_OK=true; \
	$(RUFF) check . || RUFF_OK=false; \
	$(MYPY) . || MYPY_OK=false; \
	if $$RUFF_OK && $$MYPY_OK; then \
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
	rm -rf .agents/ .claude/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	rm -f .installed

# Service management (override port: make install-service PORT=8000)
install-service: .installed
	$(VENV_BIN)/sol service install --port $(or $(PORT),5015)
	$(VENV_BIN)/sol service start
	$(VENV_BIN)/sol service status

uninstall-service:
	-$(VENV_BIN)/sol service uninstall

# Uninstall - remove venv and sol symlink
uninstall: uninstall-service clean
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
	@$(RUFF) format --check . || { echo "Run 'make format' to fix formatting"; exit 1; }
	@echo ""
	@echo "=== Running ruff ==="
	@$(RUFF) check . || { echo "Run 'make format' to auto-fix"; exit 1; }
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
	@$(UV) pip show pytest-watch >/dev/null 2>&1 || { echo "Installing pytest-watch..."; $(UV) pip install pytest-watch; }
	$(VENV_BIN)/ptw -- -q

# Generate coverage report (core + apps, excluding core integration tests)
coverage: .installed
	$(TEST_ENV) $(PYTEST) tests/ --cov=. --cov-report=html --cov-report=term --ignore=tests/integration
	$(TEST_ENV) $(PYTEST) apps/ --cov=. --cov-report=html --cov-report=term --cov-append
	@echo "Coverage report generated in htmlcov/index.html"

# Update all dependencies to latest versions and refresh genai-prices
update: .installed
	@echo "Updating all dependencies to latest versions..."
	$(UV) lock --upgrade
	$(UV) sync
	@echo "Done. All packages updated to latest."

# Update genai-prices to get latest model pricing data
# Run this when adding new models or if pricing tests fail
update-prices: .installed
	@echo "Updating genai-prices to latest version..."
	$(UV) lock --upgrade-package genai-prices
	$(UV) sync
	@echo "Done. Re-run tests to verify model pricing support."

# Show installed package versions
versions: .installed
	@echo "=== Python version ==="
	$(PYTHON) --version
	@echo ""
	@echo "=== Key package versions ==="
	@$(UV) pip list | grep -E "^(pytest|ruff|mypy|Flask|numpy|Pillow|openai|anthropic|google-genai)" || true

# Install pre-commit hooks (if using pre-commit)
pre-commit: .installed
	@$(UV) pip show pre-commit >/dev/null 2>&1 || { echo "Installing pre-commit..."; $(UV) pip install pre-commit; }
	$(VENV_BIN)/pre-commit install
	@echo "Pre-commit hooks installed!"
