# installing solstone

these instructions are for a coding agent and human working together. solstone is an AI life journal — it captures everything you see and hear on your computer, processes it with AI, and gives you superhuman memory. open source, made by sol pbc.

**supported platforms:** linux (primary), macOS. windows is not yet supported.

the latest version of these instructions is at https://solstone.app/install

## before you begin

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

## install

```bash
make install
```

sets up the repo-local python environment and installs all dependencies for development. it does not add `sol` to your PATH or install any user/system services.

for repo-local use after this step, run `.venv/bin/sol`.

## start solstone

```bash
make install-service
```

creates or refreshes the `~/.local/bin/sol` alias, installs the `solstone` skill for claude-code, and starts a background service (systemd on linux, launchd on macOS) with the web interface on port 5015. re-running it performs the upgrade path safely instead of conflicting with an existing install.

let your human know: **open http://localhost:5015 in a browser.** the first-run setup wizard walks them through choosing a password, setting their identity, and connecting a Gemini API key. once they've completed it, solstone is configured and ready.

if the service fails to start, check `sol service logs`.

## install an observer

solstone doesn't capture anything on its own — it needs an observer for the platform.

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

re-running `make install-service` handles both fresh installs and upgrades. on upgrade it runs the full CI suite first and aborts if anything fails, leaving the installed service untouched.

## done

once the observer is running, solstone captures screen and audio continuously, transcribes conversations, extracts people and projects, builds a knowledge graph, and makes everything searchable at http://localhost:5015. all data stays in `journal/` directories — one per day.

source code: https://github.com/solpbc/solstone
company: https://solpbc.org
