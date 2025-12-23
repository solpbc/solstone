<img src="logo.png" alt="Sunstone Logo" width="300">

# Sunstone
Navigate Work Intelligently

A comprehensive Python-based AI-driven desktop journaling toolkit for multimodal capture, analysis, and intelligent navigation of workplace activities. Sunstone organizes captured data under a **journal** directory containing daily `YYYYMMDD` folders, enabling powerful temporal analysis and review.

## 🚀 Features

### Core Modules

- **Observe** 👁️👂 - Multimodal capture and analysis
  - `observe-gnome` - Screencast monitoring on Linux/GNOME
  - `observe-describe` - Analyzes visual changes with AI
  - `observe-transcribe` - Transcribes audio with AI APIs
  - `observe-sense` - Unified observation coordination
  - *Note: Requires Linux with GNOME desktop*

- **Think** 🧠 - Data processing and insights
  - `think-insight` - Generates AI-powered insights and summaries
  - `think-cluster` - Groups related content
  - `think-indexer` - Builds searchable database
  - `think-supervisor` - Orchestrates agent workflows
  - `think-dream` - Full daily data pipeline

- **Convey** 🌐 - Web interface and review
  - `convey` - Launch web UI (with `--password` for auth)
  - Review facet-scoped entities, meetings, and tasks
  - Live transcription monitoring
  - Calendar view with daily summaries

### Additional Tools

- **MCP Server** 🛰️ - `muse-mcp-tools` launches Model Context Protocol server
- **Cortex** 🧩 - `muse-cortex` provides agent-based task execution
- **Help** ❓ - `sunstone` lists all available commands

## 📦 Installation

### Basic Installation
```bash
# Install package in editable mode
make install
# or
pip install -e .
```

### Full Installation (with optional dependencies)
```bash
# Install with all optional dependencies (audio/video processing)
make full
# or
pip install -e .[full]
```

### Development Installation
```bash
# Install with development tools
make dev
# or
pip install -e .[dev]
```

## ⚙️ Configuration

Create a `.env` file in your project root or set these environment variables:

```bash
# Required
JOURNAL_PATH=/path/to/your/journal  # Where all data is stored
GOOGLE_API_KEY=your-api-key         # For Gemini AI services

# Optional
OPENAI_API_KEY=your-api-key         # For OpenAI services
ANTHROPIC_API_KEY=your-api-key      # For Claude services
```

## 🎯 Quick Start

1. **Set up environment**:
   ```bash
   cp .env.example .env  # If available
   # Edit .env with your settings
   ```

2. **Start capturing**:
   ```bash
   # Use the supervisor to run all observation
   think-supervisor
   ```

3. **Process captured data**:
   ```bash
   # Run full processing pipeline for today
   think-dream

   # Or process specific date
   think-dream 20240101
   ```

4. **Review in web UI**:
   ```bash
   # Launch web interface
   convey --password your-password
   # Open http://localhost:5000
   ```

## 🧪 Testing

```bash
# Run all tests with coverage
make test

# Run specific tests
make test-only TEST=tests/test_utils.py

# Run with verbose output
make test-verbose

# Generate HTML coverage report
make coverage
```

## 🔧 Development

```bash
# Format code
make format

# Run linting checks
make lint

# Type checking
make check

# Run all checks before committing
make check-all

# Clean build artifacts
make clean
```

See the [Makefile](Makefile) for all available commands or run `make help`.

## 📚 Documentation

- [JOURNAL.md](JOURNAL.md) - Journal directory structure and data organization
- [AGENTS.md](AGENTS.md) - Development guidelines and coding standards
- [CORTEX.md](CORTEX.md) - Agent system documentation

### Package Documentation
- [observe/README.md](observe/README.md) - Multimodal capture and analysis details
- [think/README.md](think/README.md) - Data processing and AI analysis details
- [convey/README.md](convey/README.md) - Web interface usage and features

## 🤝 Contributing

Please read [AGENTS.md](AGENTS.md) for development guidelines, coding standards, and contribution instructions.

## 📄 License

MIT License - see LICENSE file for details
