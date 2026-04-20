# Fork Changelog

This document tracks significant changes made on this fork of Solstone.


## Fixed Convey Port (3000)

**File:** `think/supervisor.py` (~line 1380)

Upstream defaults the Convey port to `0` (auto-select an available port). This
fork changes the default to `3000` so the service always binds to a known
address. A stable port is required for the home server's Pangolin reverse-proxy
relay, which routes Convey through a private network for remote access. With
auto-select, the port changes on every restart and the relay rule breaks.


---

*The following changes originated on this fork and have since been merged upstream.*


## Ollama (Local) Provider

> **Merged upstream.** The `think/providers/ollama.py` provider, tests, and
> associated config/docs changes are now part of upstream Solstone.

Added a new `ollama` provider that routes text generation requests to a local
Ollama instance via its native `/api/chat` endpoint, removing the hard
dependency on cloud API keys for the `generate` workload. Ollama models use
the `ollama-local/` prefix (e.g., `ollama-local/qwen3.5:9b`) to leave room for
a future `ollama-cloud/` variant. `run_cogitate()` shells out to the OpenCode
CLI, which connects to Ollama via its OpenAI-compatible endpoint and handles
tool execution internally. OpenCode provider config lives at the user level
(`~/.config/opencode/opencode.json`), per upstream's guidance in
`docs/PROVIDERS.md`.


## Makefile NVM/npx Lookup

> **Merged upstream.** The `NVM_BIN` detection block is now in the upstream
> `Makefile`.

Added `NVM_BIN` detection so `npx` can be found outside interactive shells
(e.g., nvm-managed Node installs).


## WebSocket HTTPS Support

> **Merged upstream** in commit [`27b0745`](https://github.com/solpbc/solstone/commit/27b0745fded2c507b5ccb94df906434c5bc7818d)

**File:** `convey/static/websocket.js`

The WebSocket connection URL was hardcoded to `ws://`, which causes a mixed
content error when the dashboard is served over HTTPS. The browser blocks
insecure WebSocket connections from HTTPS pages. Changed to auto-detect the
protocol (`wss:` for HTTPS, `ws:` for HTTP) based on `location.protocol`.
