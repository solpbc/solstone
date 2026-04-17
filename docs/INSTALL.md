# Installation Guide

Complete setup instructions for solstone on Linux and macOS.

## Prerequisites

- Python 3.10 or later
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

To remove installed user/system artifacts:

```bash
make uninstall-service
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

## First Run

### Install as a Background Service

The recommended way to run solstone is as a system service that starts automatically on login:

```bash
make install-service
```

This creates or refreshes the `~/.local/bin/sol` alias, installs the global `solstone` skill for claude-code, and installs, enables, and starts a systemd user service (Linux) or launchd agent (macOS) with convey on port 5015. Re-running it upgrades an existing install instead of conflicting. To use a custom port:

```bash
make install-service PORT=8000
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

Observers capture screen and audio and upload to the solstone server. Each platform has its own standalone observer. Packages are not yet on PyPI — install from source.

### Linux Observer

```bash
git clone https://github.com/solpbc/solstone-linux.git
cd solstone-linux
pipx install --system-site-packages .
solstone-linux setup
solstone-linux install-service
```

`--system-site-packages` is required for PyGObject/GStreamer access.

**Note:** Activity detection (idle, lock, power save) requires GNOME desktop. Other desktops: capture works but activity-based segment boundaries won't trigger.

### tmux Observer

```bash
git clone https://github.com/solpbc/solstone-tmux.git
cd solstone-tmux
pipx install .
solstone-tmux setup
solstone-tmux install-service
```

### macOS Observer

See [solstone-macos](https://github.com/solpbc/solstone-macos) — requires Xcode (full IDE, not just CLI tools).

---

## Next Steps

- Create your first facet (project/context) in the web interface
- Review captured content in the Calendar and Transcripts apps
- Chat with the AI about your journal content

For development setup, see [AGENTS.md](../AGENTS.md).
