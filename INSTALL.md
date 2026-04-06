# installing solstone

these instructions are for a coding agent and human working together. solstone is an AI life journal — it captures everything you see and hear on your computer, processes it with AI, and gives you superhuman memory. open source, made by sol pbc.

your data resides on your computer. sol pbc is a public benefit corporation with irrevocable legal covenants: your data can never be sold, the company can never be acquired. the code is open source — you can verify every claim.

**supported platforms:** linux (primary), macOS. windows is not yet supported.

the latest version of these instructions is at https://solstone.app/install

## before you begin

check if solstone is already installed and running:

```bash
sol --version 2>&1 && sol service status 2>&1 && sol health 2>&1
```

if `sol` isn't found, try `~/.local/bin/sol`. if solstone is running and healthy, skip to [install an observer](#install-an-observer).

## what to sort out together

- **a Google AI Studio API key.** required for transcription and vision processing. the human needs to go to https://aistudio.google.com/apikey, sign in, and create one.
- **a password for the web interface.** protects the journal's web UI on localhost. any password they'll remember.
- **system dependencies.** some packages require sudo — if you don't have it, walk your human through the install commands.

## platform prerequisites

detect the platform:

```bash
uname -s
```

### linux

check what's already installed:

```bash
python3 --version 2>&1; git --version 2>&1; ffmpeg -version 2>&1 | head -1; uv --version 2>&1
```

you need python 3.10+, git, ffmpeg, and uv. install anything that's missing.

**fedora / rhel:**
```bash
sudo dnf install python3 git ffmpeg
```

**debian / ubuntu:**
```bash
sudo apt install python3 git ffmpeg
```

**arch:**
```bash
sudo pacman -S python git ffmpeg
```

install uv if not present:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

uv installs to `~/.local/bin/uv` or `~/.cargo/bin/uv`. it may not be in PATH yet — call it by full path if needed.

### macOS

several of these steps require human interaction — password prompts, app store installs, and dialogs the agent can't click through.

**Xcode:** the macOS observer requires the full Xcode IDE (not just command line tools). your human needs to install it from the Mac App Store if they haven't already. then install command line tools:

```bash
xcode-select --install
```

**homebrew:** check if installed with `brew --version`. if not, the human needs to install it from https://brew.sh (requires their password).

```bash
brew install python git ffmpeg uv
```

### verify

```bash
python3 --version && git --version && ffmpeg -version 2>&1 | head -1 && uv --version
```

all should produce version output. python must be 3.10+.

## install

```bash
make install
```

`make install` creates a python virtual environment, installs dependencies, and symlinks `sol` to `~/.local/bin/sol`.

verify:

```bash
.venv/bin/sol --help
```

if `sol` is not in PATH, use `.venv/bin/sol` for the remaining steps. the human can add `~/.local/bin` to their shell profile's PATH to fix this permanently.

## configure

collect the API key and password from your human, then write the config:

```bash
mkdir -p journal/config
```

write `journal/config/journal.json`:

```json
{
  "convey": {
    "password": "THE_PASSWORD_THEY_CHOSE"
  },
  "env": {
    "GOOGLE_API_KEY": "THE_GEMINI_KEY_THEY_PROVIDED"
  }
}
```

if you have access to an API key for your own platform, you can also add it so solstone can use your provider for chat and agent features — confirm with your human before adding:

- claude code / anthropic: add `"ANTHROPIC_API_KEY"` to the `env` section
- codex / openai: add `"OPENAI_API_KEY"` to the `env` section

lock down the config — it contains secrets:

```bash
chmod 600 journal/config/journal.json
```

## start solstone

```bash
make install-service
```

this installs, enables, and starts a background service (systemd on linux, launchd on macOS) with the web interface on port 5015.

verify:

```bash
sol service status
sol health
```

the web interface should be available at `http://localhost:5015`.

if the service fails to start, check the logs with `sol service logs`. common causes: missing or invalid API key, missing system dependencies, permission issues.

## install an observer

solstone doesn't capture anything on its own — it needs an observer for the platform. detect which one to install:

```bash
uname -s
```

- **Linux** — solstone-linux (screen + audio via PipeWire/GStreamer)
- **Darwin** — solstone-macos (native swift menu bar app)

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

## hand off to the human

solstone and its observer are installed and running. let your human know:

1. **solstone runs as a background service** that starts automatically on login. manage it with:
   - `sol service status` — check if running
   - `sol service restart` — restart
   - `sol service stop` — stop
   - `sol service logs -f` — follow logs

2. **the web interface** is at `http://localhost:5015`. log in with the password they chose.

3. **after logging in**, they should go to Settings (gear icon) and fill in their identity — name, preferred name, pronouns, timezone. this helps solstone identify them in transcripts and personalize AI responses.

4. **what solstone does from here:** it captures screen and audio continuously, transcribes conversations, extracts people and projects, builds a knowledge graph, and makes everything searchable. all data stays in `journal/` directories — one per day.

source code: https://github.com/solpbc/solstone
company: https://solpbc.org
