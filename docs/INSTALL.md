# Installation Guide

Complete setup instructions for solstone on Linux and macOS.

## Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Git
- ffmpeg (for audio processing)

### Linux (Fedora/RHEL)

```bash
sudo dnf install python3 git ffmpeg pipewire gstreamer1-plugins-base
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Linux (Ubuntu/Debian)

```bash
sudo apt install python3 git ffmpeg pipewire gstreamer1.0-tools
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Linux (Arch)

```bash
sudo pacman -S python git ffmpeg pipewire gstreamer
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

This creates an isolated virtual environment in `.venv/` and symlinks the `sol` command to `~/.local/bin/sol`. Your system Python remains untouched.

To uninstall:

```bash
make uninstall
```

2. Copy the environment template:

```bash
cp .env.example .env
```

3. Your journal lives at `journal/` inside the solstone directory. It's created automatically on first run.

---

## API Keys

solstone requires API keys for AI services. Configure these in your `.env` file.

### Google AI (Gemini) - Recommended

Primary backend for transcription, vision analysis, and insights.

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key to `.env`:

```
GOOGLE_API_KEY=your-key-here
```

### OpenAI (Optional)

Alternative backend for chat and agents.

1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key to `.env`:

```
OPENAI_API_KEY=your-key-here
```

### Anthropic (Optional)

Alternative backend for chat and agents.

1. Go to [Anthropic Console](https://console.anthropic.com/settings/keys)
2. Sign in or create an account
3. Click "Create Key"
4. Copy the key to `.env`:

```
ANTHROPIC_API_KEY=your-key-here
```

---

## Rev.ai for Imports (Optional)

For transcribing imported audio files (meetings, voice memos, etc.).

1. Sign up at [Rev.ai](https://www.rev.ai/)

2. After account creation, go to [Access Token](https://www.rev.ai/access_token)

3. Click "Generate New Access Token"

4. Add the token to `.env`:

```
REVAI_ACCESS_TOKEN=your-token-here
```

---

## First Run

### Install as a Background Service

The recommended way to run solstone is as a system service that starts automatically on login:

```bash
make install-service
```

This installs, enables, and starts a systemd user service (Linux) or launchd agent (macOS) with convey on port 5015. To use a custom port:

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

### Set a Password

Before accessing the web interface, you must configure a password.

Create the config file:

```bash
mkdir -p journal/config
cat > journal/config/journal.json << 'EOF'
{
  "convey": {
    "password": "your-password-here"
  }
}
EOF
```

Replace `your-password-here` with a secure password.

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

## Next Steps

- Create your first facet (project/context) in the web interface
- Set up a standalone observer for your platform (solstone-linux, solstone-macos)
- Review captured content in the Calendar and Transcripts apps
- Chat with the AI about your journal content

For development setup, see [AGENTS.md](../AGENTS.md).
