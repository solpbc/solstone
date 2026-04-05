# Fork Changelog

This document tracks significant changes made on this fork of Solstone.


## Ollama (Local) Provider

**Branch:** `ollama-generate-provider`

Added a new `ollama` provider to `think/providers/` that routes text generation
requests to a local Ollama instance via its OpenAI-compatible API. This removes
the hard dependency on cloud API keys for the most common workload (`generate`).

**Key design decisions:**

- **Model prefix convention:** Ollama models use the `ollama-local/` prefix
  (e.g., `ollama-local/qwen3.5:9b`). The prefix is stripped before sending
  requests to the Ollama API. This supports a future `ollama-cloud/` variant.
- **Native API:** Uses Ollama's native ``/api/chat`` endpoint via ``httpx``
  (already a transitive dependency of ``openai``). The OpenAI-compatible
  endpoint was rejected because it silently ignores the ``think`` parameter
  on models like Qwen3.5.
- **No API key:** Ollama requires no authentication. ``validate_key()``
  checks Ollama reachability instead of key validity.
- **Thinking support:** Maps `thinking_budget` to Ollama's `think` parameter
  via `extra_body`. Budget > 0 enables thinking; None/0 disables it.
- **Cogitate deferred:** `run_cogitate()` raises `NotImplementedError` in this
  first pass. Users should configure a cloud provider (or backup) for agents.

**Default tier models:**

| Tier | Model |
|------|-------|
| PRO (1) | `qwen3.5:35b-a3b-bf16` |
| FLASH (2) | `qwen3.5:9b` |
| LITE (3) | `qwen3.5:2b` |

Override via `providers.models.ollama` in `journal.json`.

**Files added:** `think/providers/ollama.py`, `tests/test_ollama.py`,
`tests/integration/test_ollama_backend.py`

**Files modified:** `think/providers/__init__.py`, `think/models.py`,
`docs/PROVIDERS.md`, `docs/THINK.md`, `apps/settings/workspace.html`,
`tests/fixtures/journal/config/journal.json`


## Minimal Changes

- **Makefile NVM/npx lookup** (`Makefile` ~line 20): Added `NVM_BIN` detection
  so `npx` can be found outside interactive shells (e.g., nvm-managed Node
  installs). Also catches internal node calls.
