# Installation Guide

Complete setup instructions for solstone on Linux and macOS.

## Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Git
- ffmpeg (for audio processing)

### Linux (Fedora/RHEL)

```bash
sudo dnf install python3 git ffmpeg pipewire gstreamer1-plugins-base gstreamer1-plugin-pipewire pulseaudio-utils
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Linux (Ubuntu/Debian)

```bash
sudo apt install python3 git ffmpeg pipewire gstreamer1.0-tools gstreamer1.0-pipewire pulseaudio-utils
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Linux (Arch)

```bash
sudo pacman -S python git ffmpeg pipewire gstreamer gst-plugin-pipewire libpulse
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### macOS

```bash
xcode-select --install  # Command line tools
brew install python git ffmpeg uv
```

---

## Installation

1. Clone and install:

```bash
git clone https://github.com/solpbc/solstone.git
cd solstone
make install
```

This creates an isolated virtual environment in `.venv/` for local development. Your system Python remains untouched, and no user-level CLI alias or service is installed yet.

To remove installed user/system artifacts later:

```bash
sol service stop
sol service uninstall
sol skills uninstall
python -m think.install_guard uninstall
```

To reset the repo-local development environment:

```bash
make clean-install
```

2. Your journal lives at `journal/` inside the solstone directory. It's created automatically on first run.

---

## API Keys

solstone requires API keys for AI services. All keys are configured in `journal/config/journal.json` — this is the only key configuration method.

Create the config file:

```bash
mkdir -p journal/config
cat > journal/config/journal.json << 'EOF'
{
  "convey": {},
  "env": {
    "GOOGLE_API_KEY": "your-key-here"
  }
}
EOF
chmod 600 journal/config/journal.json
```

Run `sol password set` to configure web authentication. Replace `your-key-here` with your Google AI API key.

### Google AI (Gemini) - Required

Primary backend for transcription, vision analysis, and insights.

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Add the key to the `env` section of `journal/config/journal.json`

### OpenAI (Optional)

Alternative backend for chat and agents. Add `"OPENAI_API_KEY": "your-key"` to the `env` section.

### Anthropic (Optional)

Alternative backend for chat and agents. Add `"ANTHROPIC_API_KEY": "your-key"` to the `env` section.

### Rev.ai for Imports (Optional)

For transcribing imported audio files. Sign up at [Rev.ai](https://www.rev.ai/), get a token from [Access Token](https://www.rev.ai/access_token), and add `"REVAI_ACCESS_TOKEN": "your-token"` to the `env` section.

### Example with multiple providers

```json
{
  "convey": {},
  "env": {
    "GOOGLE_API_KEY": "your-gemini-key",
    "OPENAI_API_KEY": "your-openai-key",
    "ANTHROPIC_API_KEY": "your-anthropic-key"
  }
}
```

**Important:** `journal.json` contains your API keys and credentials. It should always have restricted permissions (`chmod 600`).

---

## Cogitate CLI Binaries (Required for Tool-Using Agents)

solstone runs two distinct AI workloads, and they have different prerequisites:

- **Generate** — direct text generation (transcription, vision analysis, insights, descriptions). Uses each provider's SDK and works as soon as the API key is set. No extra binaries.
- **Cogitate** — tool-using agents (entity detection, entity assist, entity describe, and any talent under `apps/entities/talent/*.md` with `"type": "cogitate"`). solstone shells out to the provider's official CLI binary as a subprocess. The binary must be installed and on `PATH`.

Because cogitate-type talents only run when an entity-related dream completes, missing CLI binaries are invisible during initial install — generate-only paths produce transcripts and summaries, but **entities never appear**. `sol top` Agents Health flags this with messages like `"Google generate passes but Google cogitate fails: gemini CLI not installed"`. Install the binary for **each provider whose API key you configured above**. If you only set `GOOGLE_API_KEY`, you only need `gemini`.

| provider  | binary     | install                                                         |
|-----------|------------|-----------------------------------------------------------------|
| google    | `gemini`   | `npm install -g @google/gemini-cli` (Node 20+)                  |
| anthropic | `claude`   | `npm install -g @anthropic-ai/claude-code` (Node 18+)           |
| openai    | `codex`    | `npm install -g @openai/codex` (Node 16+)                       |
| ollama    | `opencode` | `curl -fsSL https://opencode.ai/install \| bash`                |

If you don't have Node.js: `brew install node` on macOS, your distro's package manager on Linux (e.g. `sudo dnf install nodejs`, `sudo apt install nodejs npm`). Verify each binary is on `PATH` after install:

```bash
gemini --version
claude --version
codex --version
opencode --version
```

After installing a CLI binary while solstone is running, restart the service so cortex picks up the new `PATH`:

```bash
sol service restart
```

`sol doctor` reports per-provider readiness, including whether the cogitate CLI is found.

---

## First Run

### Install as a Background Service

The recommended way to run solstone is through setup, which installs the runtime artifacts and starts the service:

```bash
.venv/bin/sol setup
```

This creates or refreshes the `~/.local/bin/sol` wrapper for source-checkout installs, installs the global `solstone` skill for Claude Code when Claude is configured, and installs, enables, and starts a systemd user service (Linux) or launchd agent (macOS) with convey on port 5015. After the first run, the wrapper at `~/.local/bin/sol` lets you use just `sol` from anywhere. Service installation runs only on source-checkout installs in v1; packaged installs skip the service step. Re-running it is safe. To use a custom port on the first run:

```bash
.venv/bin/sol setup --port 8000
```

Manage the service with:

```bash
sol service status     # Check if running
sol service restart    # Restart
sol service stop       # Stop
sol service logs -f    # Follow logs
sol up                 # Install + start (if not already)
sol down               # Stop
```

### Start Manually (Development)

For development, run the supervisor directly in a terminal:

```bash
sol supervisor         # Auto-selects an available port
sol supervisor 8000    # Use a specific port
```

This starts:
- **Sense** - File detection and processing dispatch
- **Callosum** - Message bus for inter-service communication
- **Cortex** - Agent execution

### Verify Services

Check that services are running:

```bash
sol health
```

---

## Web Interface

### Access the Interface

Open the Convey URL shown in the terminal (port is dynamically assigned) and log in with your password.

### Configure Your Identity

After logging in:

1. Click the **Settings** app in the left menu (gear icon)
2. Fill in your identity information:
   - **Full Name** - Your legal name
   - **Preferred Name** - How you want to be addressed
   - **Pronouns** - Select from dropdown
   - **Timezone** - Auto-detected, adjust if needed

This helps the system identify you in transcripts and personalize AI responses.

---

## Health Check

Verify everything is working:

```bash
# Check services are running
pgrep -af "sol:sense|sol:supervisor"

# Check Callosum socket exists
ls -la journal/health/callosum.sock

# View service logs
tail -f journal/health/*.log
```

See [DOCTOR.md](DOCTOR.md) for troubleshooting.

---

## Observers

Observers run alongside each platform and send observations to the solstone server. The `sol observer install` command handles the normal local setup path.

### Install

```bash
sol observer install
sol observer install laptop
sol observer install laptop --platform linux
sol observer install laptop --platform tmux
sol observer install --dry-run
```

With no name, `sol observer install` uses the machine hostname. Pass a name for a stable stream name, pass `--platform` to choose linux or tmux explicitly, and use `--dry-run` to see the plan before anything changes.

Linux installs from `https://github.com/solpbc/solstone-linux.git`; tmux installs from `https://github.com/solpbc/solstone-tmux.git`. If a system package is missing, the command reports the distro-specific install command and stops before changing observer state.

### macOS Observer

```bash
sol observer install --platform macos
```

macOS is delivered as a signed app bundle. The command directs you to https://solstone.app/observers.

For manual build-from-source troubleshooting, use each observer repo's `INSTALL.md`. The linux repo includes distro package details and desktop notes; the tmux repo covers its pure-python service path.

---

## Next Steps

- Create your first facet (project/context) in the web interface
- Review captured content in the Calendar and Transcripts apps
- Chat with the AI about your journal content

For development setup, see [AGENTS.md](../AGENTS.md).
