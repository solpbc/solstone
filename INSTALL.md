# installing solstone

these instructions are for a coding agent and human working together. solstone is your co-brain: observers experience your day along with you, sol curates your memories, and your journal holds everything. open source, made by sol pbc.

**supported platforms:** linux (primary), macOS. windows is not yet supported.

the latest version of these instructions is at https://solstone.app/install

## before you begin

`sol setup` adds `~/.local/bin` to your shell `PATH` when it installs the source-checkout wrapper via the `userpath` library, updating `~/.bashrc`, `~/.zshrc`, or `~/.config/fish/config.fish` as needed. if `~/.local/bin` was not already on `PATH`, restart your shell after setup or run `exec $SHELL -l` before continuing.

check if solstone is already installed and running:

```bash
sol --version 2>&1 && sol service status 2>&1
```

if `sol` isn't found, try `~/.local/bin/sol`. if solstone is running and healthy, skip to [install an observer](#install-an-observer).

## prerequisites

the only hard requirement is `uv` — the Makefile checks for it and errors with install instructions if missing.

tool-using agents (entity detection, entity assist, etc.) additionally shell out to provider CLI binaries — `gemini`, `claude`, `codex`, or `opencode` — one per configured provider. install them after setting your API keys; see [docs/INSTALL.md § Cogitate CLI Binaries](docs/INSTALL.md#cogitate-cli-binaries-required-for-tool-using-agents) for the table.

### linux

```bash
git --version 2>&1; uv --version 2>&1
```

if git is missing, install it via your distro's package manager. install uv if not present:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### macOS

your human may need to handle interactive steps here — app store installs and password prompts.

**Xcode:** required for the macOS observer. check with `xcodebuild -version`. if missing, the human installs from the Mac App Store, then: `xcode-select --install`

**homebrew:** check with `brew --version`. if missing, the human installs from https://brew.sh.

```bash
brew install git uv
```

#### Parakeet backend (default)

- Parakeet is the default transcription backend on supported hosts. The journal config looks like:

```json
{
  "transcribe": {
    "backend": "parakeet",
    "parakeet": {
      "model_version": "v3",
      "device": "auto",
      "quantization": "auto",
      "timeout_sec": 120.0
    }
  }
}
```

- `device: "auto"` is primarily for Linux; macOS always routes to the CoreML helper on Apple Silicon.
- `quantization: "auto"` resolves to fp32 on Linux. The old `precision` key is still accepted as a one-release deprecated alias for `quantization`.
- `make install` builds the platform prerequisites and then runs `sol install-models`.
- `sol install-models` is idempotent and prints `model ready: <path>` when the local model cache is ready.
- **macOS (Apple Silicon):** Xcode command line tools are required because the helper is a Swift package; if the `xcodebuild -version` check above fails, fix that first. `sol install-models` downloads roughly 461 MB of model data into `~/Library/Application Support/solstone/parakeet/models`.
- **Linux (x86_64):** `make install` auto-detects `PARAKEET_ONNX_VARIANT` (`cuda` when `nvidia-smi -L` succeeds, otherwise `cpu`), runs `uv sync --extra parakeet-onnx-<variant>`, then runs `sol install-models`. Override direct model installation with `sol install-models --variant cuda` or `sol install-models --variant cpu`; `PARAKEET_ONNX_VARIANT` is honored for one release cycle as a fallback when `--variant auto` is used. The CPU extra installs `parakeet-onnx-cpu`; the CUDA extra installs `parakeet-onnx-cuda`. The ONNX model footprint is roughly 2.55 GB, and the CUDA wheels add roughly 700 MB more when using the CUDA variant.
- To use Whisper instead, set `transcribe.backend = "whisper"` in `journal/config/journal.json` or switch the backend in the settings UI. The Whisper code and dependencies remain available for rollback.
- helper contract details live in `solstone/observe/transcribe/parakeet_helper/README.md`.

## install

```bash
make install
```

sets up the repo-local python environment and installs all dependencies for development. it does not add `sol` to your PATH or install any user/system services.

once installed, run `sol doctor` to diagnose the install. before `make install` has run, or on a machine without `.venv`, run `python3 scripts/doctor.py` from the repo root for the same diagnostic.

for repo-local use after this step, run `.venv/bin/sol`.

## start solstone

```bash
.venv/bin/sol setup
```

runs doctor, confirms the journal path, installs local models, installs the `solstone` skill for Claude Code when Claude is configured, creates or refreshes the `~/.local/bin/sol` wrapper for source-checkout installs, and starts a background service (systemd on linux, launchd on macOS) with the web interface on port 5015. use `.venv/bin/sol setup --port 8000` to choose another port on the first run. after the first run, the wrapper at `~/.local/bin/sol` lets you use just `sol` from anywhere. Service installation runs only on source-checkout installs in v1; packaged installs skip the service step. re-running it is safe.

let your human know: **open http://localhost:5015 in a browser.** the first-run setup wizard walks them through choosing a password, setting their identity, and connecting a Gemini API key. once they've completed it, solstone is configured and ready.

if the service fails to start, check `sol service logs`.

## install an observer

solstone needs a platform observer alongside the server.

```bash
sol observer install
sol observer install laptop
sol observer install laptop --platform linux
sol observer install --platform tmux
sol observer install --dry-run
```

with no name, `sol observer install` uses the machine hostname. pass a name when you want a stable stream name, pass `--platform` to choose linux or tmux explicitly, and use `--dry-run` to see the plan before anything is changed.

on macOS, `sol observer install --platform macos` directs you to the signed app bundle at https://solstone.app/observers.

for manual build-from-source troubleshooting, use the per-observer repo docs: `solstone-linux/INSTALL.md`, `solstone-tmux/INSTALL.md`, or the macOS app instructions.

## updating after a code change

```bash
git pull && make install && .venv/bin/sol setup
```

`make install` refreshes the repo-local Python environment. `.venv/bin/sol setup` then reruns the runtime setup gates and refreshes user-level artifacts. for a high-confidence upgrade, run `make verify && make install && .venv/bin/sol setup` before touching the running service.

## done

once the observer is running, solstone observes screen and audio continuously, transcribes conversations, extracts people and projects, builds a knowledge graph, and makes everything searchable at http://localhost:5015. all data stays in `journal/` directories — one per day.

source code: https://github.com/solpbc/solstone
company: https://solpbc.org
