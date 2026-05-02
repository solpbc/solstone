# installing solstone

these instructions are for a coding agent and human working together. solstone is your co-brain: observers experience your day along with you, sol curates your memories, and your journal holds everything. open source, made by sol pbc.

**supported platforms:** linux (primary), macOS. windows is not yet supported.

the latest version of these instructions is at https://solstone.app/install

## before you begin

`make install-service` now auto-adds `~/.local/bin` to your shell `PATH` via the `userpath` library, updating `~/.bashrc`, `~/.zshrc`, or `~/.config/fish/config.fish` as needed. if `~/.local/bin` was not already on `PATH`, restart your shell after install or run `exec $SHELL -l` before continuing.

check if solstone is already installed and running:

```bash
sol --version 2>&1 && sol service status 2>&1
```

if `sol` isn't found, try `~/.local/bin/sol`. if solstone is running and healthy, skip to [install an observer](#install-an-observer).

## prerequisites

the only hard requirement is `uv` — the Makefile checks for it and errors with install instructions if missing.

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
- `make install` builds the platform prerequisites and then runs `make install-models`.
- `make install-models` is idempotent and prints `model ready: <path>` when the local model cache is ready.
- **macOS (Apple Silicon):** Xcode command line tools are required because the helper is a Swift package; if the `xcodebuild -version` check above fails, fix that first. `make install-models` downloads roughly 461 MB of model data into `~/Library/Application Support/solstone/parakeet/models`.
- **Linux (x86_64):** `make install` auto-detects `PARAKEET_ONNX_VARIANT` (`cuda` when `nvidia-smi -L` succeeds, otherwise `cpu`), runs `uv sync --extra parakeet-onnx-<variant>`, then runs `make install-models`. Override detection with `PARAKEET_ONNX_VARIANT=cuda make install` or `PARAKEET_ONNX_VARIANT=cpu make install`. The CPU extra installs `parakeet-onnx-cpu`; the CUDA extra installs `parakeet-onnx-cuda`. The ONNX model footprint is roughly 2.55 GB, and the CUDA wheels add roughly 700 MB more when using the CUDA variant.
- To use Whisper instead, set `transcribe.backend = "whisper"` in `journal/config/journal.json` or switch the backend in the settings UI. The Whisper code and dependencies remain available for rollback.
- helper contract details live in `observe/transcribe/parakeet_helper/README.md`.

## install

```bash
make install
```

sets up the repo-local python environment and installs all dependencies for development. it does not add `sol` to your PATH or install any user/system services.

once installed, run `sol doctor` to diagnose the install. before `make install` has run, or on a machine without `.venv`, run `python3 scripts/doctor.py` from the repo root for the same diagnostic.

for repo-local use after this step, run `.venv/bin/sol`.

## start solstone

```bash
make install-service
```

creates or refreshes the `~/.local/bin/sol` alias, installs the `solstone` skill for claude-code, and starts a background service (systemd on linux, launchd on macOS) with the web interface on port 5015. re-running it performs the upgrade path safely instead of conflicting with an existing install.

let your human know: **open http://localhost:5015 in a browser.** the first-run setup wizard walks them through choosing a password, setting their identity, and connecting a Gemini API key. once they've completed it, solstone is configured and ready.

if the service fails to start, check `sol service logs`.

## install an observer

solstone doesn't observe anything on its own — it needs an observer for the platform.

```bash
uname -s
```

clone the observer into solstone's observers directory and follow its INSTALL.md:

```bash
cd "$(sol root)/observers"
```

**linux:**
```bash
git clone https://github.com/solpbc/solstone-linux.git
```
then read `solstone-linux/INSTALL.md` and follow it.

**macOS:**
```bash
git clone https://github.com/solpbc/solstone-macos.git
```
then read `solstone-macos/INSTALL.md` and follow it.

## updating after a code change

```bash
git pull && make install-service
```

re-running `make install-service` handles both fresh installs and upgrades. on upgrade it runs fast install-time gates (`make install-checks` — formatting, lint, layer hygiene, mypy) first and aborts if anything fails, leaving the installed service untouched. the full test suite is no longer gated on install, because tests can flake under real service load. for a high-confidence upgrade, run `make verify && make install-service` to execute install-checks plus the test suite before touching the running service.

## done

once the observer is running, solstone observes screen and audio continuously, transcribes conversations, extracts people and projects, builds a knowledge graph, and makes everything searchable at http://localhost:5015. all data stays in `journal/` directories — one per day.

source code: https://github.com/solpbc/solstone
company: https://solpbc.org
