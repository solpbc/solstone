# installing solstone

these instructions are for a coding agent and human working together. solstone is your co-brain — your observers experience your day along with you, sol curates your memories, and your journal holds everything. open source, made by sol pbc.

**supported platforms:** linux (primary), macOS. windows is not yet supported.

the latest version of these instructions is at https://solstone.app/install.

## before you begin

### check whether solstone is already installed

```bash
sol --version 2>&1 && sol service status 2>&1
```

if `sol` isn't on PATH, the install hasn't been done yet — proceed.
if solstone is running and healthy, skip to [install an observer](#install-an-observer).

### prerequisites

linux: `uv` is the only requirement. install with `curl -LsSf https://astral.sh/uv/install.sh | sh`.

macOS: install xcode command line tools (`xcode-select --install`) and homebrew (https://brew.sh), then `brew install uv`.

## install

```bash
uv tool install solstone
```

(or `pipx install solstone` if you prefer pipx — they're equivalent for our purposes.)

`uv tool install` puts `sol` at `~/.local/bin/sol`, which most shells already have on PATH. if not: `exec $SHELL -l` or restart your shell.

## set up

```bash
sol setup
```

this runs doctor diagnostics, confirms the journal directory at `~/journal`, installs the local transcription model (~2.5 GB on linux), installs the solstone skill for claude code if you have claude configured, and starts a background service (systemd on linux, launchd on macOS) listening on http://localhost:5015.

let your human know: **open http://localhost:5015 in a browser**. the first-run wizard walks them through setting their identity and connecting a gemini API key. network access, and the password it requires, can be configured later in settings → security.

if a step has missing system libraries or python extras, `sol doctor` will tell you the exact install command to run for your platform. extras (`pdf`, `whisper`) can be added at any time with `uv tool upgrade solstone --extra pdf` or `pip install 'solstone[pdf]'`. on linux, local parakeet transcription needs `solstone[parakeet-onnx-cpu]` (or `[parakeet-onnx-cuda]` for NVIDIA GPUs); install or upgrade the same way as other extras.

if the service fails to start, check `sol service logs`.

## install an observer

solstone needs a platform observer alongside the server.

```bash
sol observer install                    # uses hostname as stream name
sol observer install laptop             # named stream
sol observer install laptop --platform linux
sol observer install --platform tmux
sol observer install --dry-run          # preview only
```

on macOS, `sol observer install --platform macos` directs you to the signed app bundle at https://solstone.app/observers.

## upgrading

```bash
uv tool upgrade solstone && sol setup
```

(or `pipx upgrade solstone && sol setup`.) the second command refreshes the runtime artifacts and reconciles the service unit if anything has changed.

## done

once the observer is running, your observers experience your day along with you, transcribe conversations, surface people and projects, build a knowledge graph, and make everything searchable at http://localhost:5015. everything stays in your journal — one folder per day.

source code: https://github.com/solpbc/solstone
company: https://solpbc.org

(running into trouble or want to develop on solstone yourself? see [CONTRIBUTING.md](CONTRIBUTING.md).)
