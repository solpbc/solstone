# Installation Guide

Complete setup instructions for solstone on Linux and macOS.

## Prerequisites

- Python 3.10 or later
- Git
- ffmpeg (for audio processing)

### Linux (Fedora/RHEL)

```bash
sudo dnf install python3 python3-pip git ffmpeg pipewire gstreamer1-plugins-base
```

### Linux (Ubuntu/Debian)

```bash
sudo apt install python3 python3-pip git ffmpeg pipewire gstreamer1.0-tools
```

### Linux (Arch)

```bash
sudo pacman -S python python-pip git ffmpeg pipewire gstreamer
```

### macOS

```bash
xcode-select --install  # Command line tools
brew install python git ffmpeg
```

**Screen/audio capture** requires [sck-cli](https://github.com/quartzjer/sck-cli):

```bash
git clone https://github.com/quartzjer/sck-cli.git
cd sck-cli
make
sudo make install
```

---

## Installation

1. Clone and install:

```bash
git clone https://github.com/solpbc/solstone.git
cd solstone
pip install -e .
```

2. Copy the environment template:

```bash
cp .env.example .env
```

3. (Optional) Set a custom journal path in `.env`:

```
JOURNAL_PATH=~/Documents/journal
```

If not set, solstone automatically uses the platform-specific default:
- Linux: `~/.local/share/solstone/journal`
- macOS: `~/Library/Application Support/solstone/journal`

The journal directory is created automatically on first use.

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

### Start the Supervisor

The supervisor manages all background services (capture, processing):

```bash
sol supervisor
```

This starts:
- **Observer** - Screen and audio capture
- **Sense** - File detection and processing dispatch
- **Callosum** - Message bus for inter-service communication

### Verify Services

In another terminal, check that services are running:

```bash
pgrep -af "sol:observer|sol:sense|sol:supervisor"
```

You should see three processes.

---

## Web Interface

### Set a Password

Before accessing the web interface, you must configure a password.

Create the config file:

```bash
mkdir -p $JOURNAL_PATH/config
cat > $JOURNAL_PATH/config/journal.json << 'EOF'
{
  "convey": {
    "password": "your-password-here"
  }
}
EOF
```

Replace `your-password-here` with a secure password.

### Access the Interface

Open http://localhost:8000/ in your browser and log in with your password.

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
pgrep -af "sol:observer|sol:sense|sol:supervisor"

# Check Callosum socket exists
ls -la $JOURNAL_PATH/health/callosum.sock

# View service logs
tail -f $JOURNAL_PATH/health/*.log
```

See [DOCTOR.md](DOCTOR.md) for troubleshooting.

---

## Next Steps

- Create your first facet (project/context) in the web interface
- Start capturing - the observer runs automatically
- Review captured content in the Calendar and Transcripts apps
- Chat with the AI about your journal content

For development setup, see [AGENTS.md](../AGENTS.md).
