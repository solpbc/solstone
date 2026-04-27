# solstone Makefile
# Python-based AI-driven desktop journaling toolkit

# Route pytest tmp dirs to /var/tmp (disk) instead of default /tmp (tmpfs/RAM).
# Each top-level pytest invocation also gets a unique --basetemp so concurrent
# runs in different worktrees do not share /var/tmp/pytest-of-$USER/pytest-N/.
# Do not re-add --basetemp to pyproject — it would pin all runs to one path and
# pytest wipes it on startup, destroying concurrent state.
export TMPDIR := /var/tmp
PYTEST_BASETEMP := $(shell mktemp -d /var/tmp/solstone-pytest-XXXXXX)
PYTEST_BASETEMP_FLAG := --basetemp $(PYTEST_BASETEMP)

.PHONY: install uninstall test test-apps test-app test-only test-integration test-integration-only test-all format format-check install-checks ci clean clean-install coverage watch versions update update-prices pre-commit skills dev all sandbox sandbox-stop install-pinchtab install-models parakeet-helper parakeet-helper-clean verify-browser update-browser-baselines review verify verify-api update-api-baselines install-service uninstall-service service-logs gate-agents-rename check-layer-hygiene doctor FORCE

# Default target - install package in editable mode
all: install

# Virtual environment directory
VENV := .venv
VENV_BIN := $(VENV)/bin
VENV_PY := $(VENV_BIN)/python
PYTHON := $(VENV_PY)
PARAKEET_ONNX_VARIANT ?= $(shell if nvidia-smi -L >/dev/null 2>&1; then echo cuda; else echo cpu; fi)

# Require uv
UV := $(shell command -v uv 2>/dev/null)
ifeq (,$(filter-out doctor,$(or $(MAKECMDGOALS),all)))
# doctor-only invocation — skip uv requirement so a uv-less machine can run diagnostics
else
ifndef UV
$(error uv is not installed. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh)
endif
endif

# Node — add nvm bin dir to PATH if npx isn't already available
NVM_BIN := $(lastword $(wildcard $(HOME)/.nvm/versions/node/*/bin))
ifneq ($(NVM_BIN),)
export PATH := $(NVM_BIN):$(PATH)
endif

# User bin directory for symlink (standard location, usually already in PATH)
USER_BIN := $(HOME)/.local/bin

.python-version-hash: FORCE
	@tmp_file=$$(mktemp); \
	python3 -c "import sys; print(sys.version_info[:2])" > "$$tmp_file"; \
	if [ ! -f $@ ] || ! cmp -s "$$tmp_file" $@; then mv "$$tmp_file" $@; else rm -f "$$tmp_file"; fi

# Marker file to track installation
.installed: pyproject.toml uv.lock .python-version-hash
	@echo "Installing package with uv..."
	$(UV) sync
	@# Python 3.14+ needs onnxruntime from nightly (not yet on PyPI)
	@OS_NAME=$$(uname -s); \
	PY_MINOR=$$($(PYTHON) -c "import sys; print(sys.version_info.minor)"); \
	if [ "$$OS_NAME" = "Darwin" ] && [ "$$PY_MINOR" -ge 14 ]; then \
		echo "Python 3.14+ detected - installing onnxruntime from nightly feed..."; \
		$(UV) pip install --pre --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime; \
	fi
	@$(VENV_BIN)/python -c "from observe.transcribe.main import PYANNOTE_OVERLAP_MODEL_PATH, PYANNOTE_OVERLAP_MODEL_SHA256, WESPEAKER_MODEL_PATH, WESPEAKER_MODEL_SHA256; from observe.utils import compute_file_sha256; actual = compute_file_sha256(WESPEAKER_MODEL_PATH); assert actual == WESPEAKER_MODEL_SHA256, f'WeSpeaker asset hash mismatch: got {actual}, expected {WESPEAKER_MODEL_SHA256}'; print(f'wespeaker asset ok ({actual[:12]}...)'); actual = compute_file_sha256(PYANNOTE_OVERLAP_MODEL_PATH); assert actual == PYANNOTE_OVERLAP_MODEL_SHA256, f'pyannote asset hash mismatch: got {actual}, expected {PYANNOTE_OVERLAP_MODEL_SHA256}'; print(f'pyannote asset ok ({actual[:12]}...)')"
	@echo "Installing Playwright browser for sol screenshot..."
	$(VENV_BIN)/playwright install chromium
	@$(MAKE) --no-print-directory skills
	@touch .installed

# Generate lock file if missing
uv.lock: pyproject.toml
	$(UV) lock

# Install package in editable mode with isolated venv
install: doctor skills .installed
	@(cd /tmp && $(CURDIR)/$(VENV_BIN)/python -c "from think.sol_cli import main") 2>/dev/null || { \
		echo ">>> re-registering editable install"; \
		$(UV) pip install -e . --no-deps; \
		if (cd /tmp && $(CURDIR)/$(VENV_BIN)/python -c "from think.sol_cli import main"); then \
			echo ">>> re-registered successfully"; \
		else \
			echo ">>> editable install still broken; run make clean-install"; \
			exit 1; \
		fi; \
	}
	@OS_NAME=$$(uname -s); \
	ARCH=$$(uname -m); \
	if [ "$$OS_NAME" = "Darwin" ] && [ "$$ARCH" = "arm64" ]; then \
		$(MAKE) parakeet-helper || { echo 'parakeet install: helper build failed' >&2; exit 1; }; \
	elif [ "$$OS_NAME" = "Linux" ]; then \
		if [ "$$ARCH" = "x86_64" ]; then \
			echo "parakeet install: PARAKEET_ONNX_VARIANT=$(PARAKEET_ONNX_VARIANT)"; \
			$(UV) sync --extra parakeet-onnx-$(PARAKEET_ONNX_VARIANT) || { echo "parakeet install: uv sync --extra parakeet-onnx-$(PARAKEET_ONNX_VARIANT) failed" >&2; exit 1; }; \
			if [ "$(PARAKEET_ONNX_VARIANT)" = "cuda" ]; then \
				$(UV) pip install --reinstall onnxruntime-gpu || { echo "parakeet install: failed to force-reinstall onnxruntime-gpu" >&2; exit 1; }; \
				$(VENV_PY) -c "import onnxruntime as ort; ort.preload_dlls(cuda=True, cudnn=True); assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'CUDAExecutionProvider missing after install'; print('parakeet install: CUDA runtime ready')" || { echo "parakeet install: CUDA runtime validation failed" >&2; exit 1; }; \
			fi; \
		else \
			echo "parakeet install: skipping unsupported Linux arch $$ARCH"; \
		fi; \
	else \
		echo "parakeet install: unsupported host '$$OS_NAME/$$ARCH'; supported: darwin/arm64, linux/x86_64" >&2; \
		exit 1; \
	fi
	@touch .installed
	@OS_NAME=$$(uname -s); \
	ARCH=$$(uname -m); \
	if [ "$$OS_NAME" = "Darwin" ] && [ "$$ARCH" = "arm64" ] || [ "$$OS_NAME" = "Linux" ] && [ "$$ARCH" = "x86_64" ]; then \
		PARAKEET_ONNX_VARIANT=$(PARAKEET_ONNX_VARIANT) $(VENV_PY) scripts/install_parakeet_model.py || { echo "parakeet install: install_parakeet_model.py failed" >&2; exit 1; }; \
	fi

# Directories where AI coding agents look for skills
SKILL_DIRS := journal/.agents/skills journal/.claude/skills

# Discover SKILL.md files in talent/ and apps/*/talent/, symlink into agent skill dirs
skills:
	@rm -rf .agents/skills .claude/skills
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
			([ -e "$$link" ] || [ -L "$$link" ]) || continue; \
			skill_name=$$(basename "$$link"); \
			if ! echo "$$SKILLS" | grep -qw "$$skill_name"; then \
				rm -rf "$$link"; \
			fi; \
		done; \
	done; \
	count=0; \
	for skill_md in talent/*/SKILL.md apps/*/talent/*/SKILL.md; do \
		[ -f "$$skill_md" ] || continue; \
		skill_dir=$$(dirname "$$skill_md"); \
		skill_name=$$(basename "$$skill_dir"); \
		for dir in $(SKILL_DIRS); do \
			target="../../../$$skill_dir"; \
			link="$$dir/$$skill_name"; \
			if [ -L "$$link" ] && [ "$$(readlink "$$link")" = "$$target" ]; then \
				:; \
			else \
				rm -rf "$$link"; \
				ln -s "$$target" "$$link"; \
			fi; \
		done; \
		count=$$((count + 1)); \
	done; \
	if [ "$$count" -gt 0 ]; then \
		echo "Linked $$count skill(s) into $(SKILL_DIRS)"; \
	fi

# Start local dev stack against fixture journal (no observers, no daily processing)
dev: .installed
	$(TEST_ENV) PATH=$(CURDIR)/$(VENV_BIN):$$PATH $(VENV_BIN)/sol supervisor 0 --no-daily

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
	SOLSTONE_JOURNAL="$$SANDBOX_JOURNAL" PATH=$(CURDIR)/$(VENV_BIN):$$PATH \
		$(VENV_BIN)/sol supervisor 0 --no-daily \
		> "$$SANDBOX_JOURNAL/health/supervisor.log" 2>&1 & \
	echo $$! > .sandbox.pid; \
	echo "Supervisor PID: $$(cat .sandbox.pid)"; \
	# Poll for readiness \
	echo "Waiting for services..."; \
	READY=false; \
	for i in $$(seq 1 20); do \
		if SOLSTONE_JOURNAL="$$SANDBOX_JOURNAL" $(VENV_BIN)/sol health > /dev/null 2>&1; then \
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

.PHONY: sandbox-seed-observers
sandbox-seed-observers: ## Seed 4 sample observers into the running sandbox journal
	@test -s .sandbox.journal || (echo "No sandbox running. Run 'make sandbox' first." && exit 1)
	@SOLSTONE_JOURNAL=$$(cat .sandbox.journal) $(VENV_BIN)/python tests/fixtures/seed_observers.py

# Verify API baselines against running sandbox
verify-api: .installed
	@echo "Verifying API baselines (sandbox)..."
	@$(MAKE) sandbox
	@SANDBOX_JOURNAL=$$(cat .sandbox.journal); \
	CONVEY_PORT=$$(cat "$$SANDBOX_JOURNAL/health/convey.port"); \
	RESULT=0; \
	SOLSTONE_JOURNAL="$$SANDBOX_JOURNAL" $(VENV_BIN)/sol indexer --rescan-full > /dev/null; \
	SOLSTONE_JOURNAL="$$SANDBOX_JOURNAL" $(VENV_BIN)/python tests/verify_api.py verify --base-url "http://localhost:$$CONVEY_PORT" || RESULT=$$?; \
	$(MAKE) sandbox-stop; \
	exit $$RESULT

# Regenerate API baseline files. By default uses the deterministic Flask
# test-client path (frozen time). For sandbox-only endpoints (graph, search,
# badge-count, updated-days), pass SANDBOX=1 to regenerate from the live
# sandbox — these rely on the indexer and real clock.
update-api-baselines: .installed
	@if [ "$(SANDBOX)" = "1" ]; then \
		echo "Updating API baselines (sandbox, includes sandbox-only endpoints)..."; \
		$(MAKE) sandbox; \
		SANDBOX_JOURNAL=$$(cat .sandbox.journal); \
		CONVEY_PORT=$$(cat "$$SANDBOX_JOURNAL/health/convey.port"); \
		RESULT=0; \
		SOLSTONE_JOURNAL="$$SANDBOX_JOURNAL" $(VENV_BIN)/sol indexer --rescan-full > /dev/null; \
		SOLSTONE_JOURNAL="$$SANDBOX_JOURNAL" $(VENV_BIN)/python tests/verify_api.py update --base-url "http://localhost:$$CONVEY_PORT" || RESULT=$$?; \
		$(MAKE) sandbox-stop; \
		exit $$RESULT; \
	else \
		echo "Updating API baselines (test client)..."; \
		$(VENV_BIN)/python tests/verify_api.py update; \
	fi


# Install pinchtab browser automation tool
install-pinchtab:
	@if command -v pinchtab >/dev/null 2>&1; then \
		echo "pinchtab already installed: $$(pinchtab --version 2>/dev/null || echo 'unknown')"; \
	else \
		echo "Installing pinchtab..."; \
		curl -fsSL https://pinchtab.com/install.sh | bash; \
	fi

# Build the parakeet helper binary (macOS/arm64 only, requires Xcode CLT)
install-models:
	@test -x "$(VENV_PY)" || { echo "parakeet install: missing $(VENV_PY); run make install first" >&2; exit 1; }
	PARAKEET_ONNX_VARIANT=$(PARAKEET_ONNX_VARIANT) $(VENV_PY) scripts/install_parakeet_model.py

parakeet-helper:
	cd observe/transcribe/parakeet_helper && swift build -c release
	@echo "built: $$(pwd)/observe/transcribe/parakeet_helper/.build/release/parakeet-helper"

# Remove parakeet helper build artifacts
parakeet-helper-clean:
	rm -rf observe/transcribe/parakeet_helper/.build observe/transcribe/parakeet_helper/.swiftpm observe/transcribe/parakeet_helper/Package.resolved

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
	SOLSTONE_JOURNAL="$$SANDBOX_JOURNAL" $(VENV_BIN)/sol indexer --rescan-full > /dev/null; \
	echo ""; \
	echo "=== API baseline verification ==="; \
	SOLSTONE_JOURNAL="$$SANDBOX_JOURNAL" $(VENV_BIN)/python tests/verify_api.py verify --base-url "$$BASE_URL" || RESULT_API=$$?; \
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
TEST_ENV = SOLSTONE_JOURNAL=tests/fixtures/journal
LINK_LIVE_TESTS = --ignore=tests/link/test_integration.py --ignore=tests/link/test_privacy_scan.py

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
	$(TEST_ENV) $(PYTEST) $(PYTEST_BASETEMP_FLAG) tests/ -q --cov=. --ignore=tests/integration $(LINK_LIVE_TESTS)

# Run app tests
test-apps: .installed
	@echo "Running app tests..."
	$(TEST_ENV) $(PYTEST) $(PYTEST_BASETEMP_FLAG) apps/ -q

# Run specific app tests
test-app: .installed
	@if [ -z "$(APP)" ]; then \
		echo "Usage: make test-app APP=<app_name>"; \
		echo "Example: make test-app APP=todos"; \
		exit 1; \
	fi
	$(TEST_ENV) $(PYTEST) $(PYTEST_BASETEMP_FLAG) apps/$(APP)/tests/ -v

# Run specific test file or pattern
test-only: .installed
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make test-only TEST=<test_file_or_pattern>"; \
		echo "Example: make test-only TEST=tests/test_utils.py"; \
		echo "Example: make test-only TEST=\"-k test_function_name\""; \
		exit 1; \
	fi
	$(TEST_ENV) $(PYTEST) $(PYTEST_BASETEMP_FLAG) $(TEST)

# Run integration tests
test-integration: .installed
	@echo "Running integration tests..."
	@STATUS=0; \
	$(TEST_ENV) $(PYTEST) $(PYTEST_BASETEMP_FLAG) tests/integration/ tests/link/test_integration.py tests/link/test_privacy_scan.py -v --tb=short --timeout=20 || STATUS=$$?; \
	if [ "$$STATUS" -ne 0 ] && [ "$$STATUS" -ne 5 ]; then exit $$STATUS; fi

# Run specific integration test
test-integration-only: .installed
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make test-integration-only TEST=<test_file_or_pattern>"; \
		echo "Example: make test-integration-only TEST=test_api.py"; \
		exit 1; \
	fi
	@TARGET="$(TEST)"; \
	case "$$TARGET" in \
		tests/*|-*) ;; \
		*) TARGET="tests/integration/$$TARGET" ;; \
	esac; \
	STATUS=0; \
	$(TEST_ENV) $(PYTEST) $(PYTEST_BASETEMP_FLAG) "$$TARGET" --timeout=20 || STATUS=$$?; \
	if [ "$$STATUS" -ne 0 ] && [ "$$STATUS" -ne 5 ]; then exit $$STATUS; fi

# Run all tests (core + apps + integration)
test-all: .installed
	@echo "Running all tests (core + apps + integration)..."
	$(TEST_ENV) $(PYTEST) $(PYTEST_BASETEMP_FLAG) tests/ -v --cov=. --ignore=tests/integration $(LINK_LIVE_TESTS) && $(TEST_ENV) $(PYTEST) $(PYTEST_BASETEMP_FLAG) apps/ -v --cov=. --cov-append

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
	rm -rf journal/.agents/ journal/.claude/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	rm -f .installed

# Pre-install diagnostic — stdlib-only; runs on system python without uv/venv
doctor:
	@python3 scripts/doctor.py $(if $(VERBOSE),--verbose) $(if $(JSON),--json) $(if $(PORT),--port $(PORT))

# Service management (override port: make install-service PORT=8000)
install-service: doctor skills .installed
	@MODE=$$($(PYTHON) -m think.install_guard check); \
	RC=$$?; \
	case "$$MODE" in \
		worktree) \
			echo "mode: aborted — worktree"; \
			exit $$RC; \
			;; \
		cross_repo) \
			echo "mode: aborted — cross_repo"; \
			exit $$RC; \
			;; \
		dangling) \
			echo "mode: aborted — dangling"; \
			exit $$RC; \
			;; \
		not_symlink) \
			echo "mode: aborted — not_symlink"; \
			exit $$RC; \
			;; \
		up""grade) \
			echo "mode: up""grade"; \
			$(MAKE) install-checks || exit $$?; \
			;; \
		current) \
			echo "mode: current"; \
			$(MAKE) install-checks || exit $$?; \
			;; \
		fresh) \
			echo "mode: fresh install"; \
			;; \
		*) \
			echo "mode: aborted — unknown"; \
			exit 2; \
			;; \
	esac; \
	$(PYTHON) -m think.install_guard install; \
	CI=true npx --yes skills add ./skills/solstone -g -a claude-code -y; \
	$(VENV_BIN)/sol service install --port $(or $(PORT),5015); \
	$(VENV_BIN)/sol service restart; \
	echo "Waiting for service readiness..."; \
	READY=false; \
	for i in $$(seq 1 20); do \
		if $(VENV_BIN)/sol health > /dev/null 2>&1; then \
			READY=true; \
			break; \
		fi; \
		sleep 1; \
	done; \
	if [ "$$READY" = "false" ]; then \
		echo "Service readiness timeout after 20s" >&2; \
		exit 1; \
	fi; \
	$(VENV_BIN)/sol service status

# Follow installed service logs
service-logs:
	$(VENV_BIN)/sol service logs -f

uninstall-service:
	@MODE=$$($(PYTHON) -m think.install_guard check); \
	RC=$$?; \
	HAS_SERVICE=false; \
	HAS_SKILL=false; \
	if [ -f "$$HOME/.config/systemd/user/solstone.service" ] || [ -f "$$HOME/Library/LaunchAgents/org.solpbc.solstone.plist" ]; then \
		HAS_SERVICE=true; \
	fi; \
	if [ -e "$$HOME/.claude/skills/solstone" ]; then \
		HAS_SKILL=true; \
	fi; \
	case "$$MODE" in \
		worktree|cross_repo|dangling|not_symlink) \
			echo "mode: aborted — $$MODE"; \
			exit $$RC; \
			;; \
	esac; \
	if [ "$$MODE" = "fresh" ] && [ "$$HAS_SERVICE" = "false" ] && [ "$$HAS_SKILL" = "false" ]; then \
		echo "no artifacts to remove"; \
		exit 0; \
	fi; \
	$(VENV_BIN)/sol service stop > /dev/null 2>&1 || true; \
	$(VENV_BIN)/sol service uninstall; \
	CI=true npx --yes skills remove -g -a claude-code -y solstone; \
	$(PYTHON) -m think.install_guard uninstall

uninstall:
	@echo "Error: 'make uninstall' is disabled. Use the 'uninstall-service' target to remove installed user/system artifacts, or 'make clean-install' to rebuild the local dev environment." >&2
	@exit 1

FORCE:

# Clean everything and reinstall
clean-install: clean
	rm -rf $(VENV) .installed
	$(MAKE) install

# Run continuous integration checks (what CI would run)
install-checks: .installed
	@echo "=== Checking formatting ==="
	@$(RUFF) format --check . || { echo "Run 'make format' to fix formatting"; exit 1; }
	@echo ""
	@echo "=== Running ruff ==="
	@$(RUFF) check . || { echo "Run 'make format' to auto-fix"; exit 1; }
	@echo ""
	@echo "=== Running rename gate ==="
	@$(MAKE) gate-agents-rename
	@echo ""
	@echo "=== Running layer-hygiene check ==="
	@$(MAKE) check-layer-hygiene
	@echo ""
	@echo "=== Running mypy ==="
	@$(MYPY) . || true
	@echo ""

ci: install-checks
	@echo "=== Running tests ==="
	@$(MAKE) test
	@echo ""
	@echo "All CI checks passed!"

verify: install-checks
	@echo "=== Running tests ==="
	@$(MAKE) test
	@echo ""
	@echo "Verification complete!"

# Watch for changes and run tests (requires pytest-watch)
watch: .installed
	@$(UV) pip show pytest-watch >/dev/null 2>&1 || { echo "Installing pytest-watch..."; $(UV) pip install pytest-watch; }
	$(VENV_BIN)/ptw -- -q

# Generate coverage report (core + apps, excluding core integration tests)
coverage: .installed
	$(TEST_ENV) $(PYTEST) $(PYTEST_BASETEMP_FLAG) tests/ --cov=. --cov-report=html --cov-report=term --ignore=tests/integration $(LINK_LIVE_TESTS)
	$(TEST_ENV) $(PYTEST) $(PYTEST_BASETEMP_FLAG) apps/ --cov=. --cov-report=html --cov-report=term --cov-append
	@echo "Coverage report generated in htmlcov/index.html"

# Update all dependencies to latest versions and refresh genai-prices
update: .installed
	@echo "Updating all dependencies to latest versions..."
	$(UV) lock -U
	$(UV) sync
	@echo "Done. All packages updated to latest."

# Update genai-prices to get latest model pricing data
# Run this when adding new models or if pricing tests fail
update-prices: .installed
	@echo "Updating genai-prices to latest version..."
	$(UV) lock -P genai-prices
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
# Rename guard for the agents -> talents transition
gate-agents-rename: .installed
	$(VENV_BIN)/python scripts/gate_agents_rename.py

# Low-bar layer-hygiene check (see docs/coding-standards.md § Layer Hygiene)
check-layer-hygiene: .installed
	$(VENV_BIN)/python scripts/check_layer_hygiene.py
