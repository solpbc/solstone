# sunstone
Navigate Work Intelligently

A comprehensive collection of multimodal AI utilities for workplace audio, visual, and cognitive processing.
Captured files are organised under a **journal** directory containing daily `YYYYMMDD` folders.

## Features

- **See** 👁️ - Screenshot capture and visual comparison with the `screen-watch` command
- **Hear** 👂 - Audio recording and transcription with the `gemini-mic` command
- **Think** 🧠 - Data analysis and AI-powered insights with the `ponder-day` command
- **Dream** 🌐 - Web interfaces for reviewing entities and meetings
- **Serve** 🛰️ - Expose search tools over MCP with the `ponder-mcp` command

## Quick Start

```bash
pip install -e .
```

Wrapper scripts `hear.sh` and `see.sh` run the audio and visual capture services
in continuous loops.

## Documentation

Formats:

- [JOURNAL.md](JOURNAL.md) - details about the structure and contents of the **journal** directory
- [CRUMBS.md](CRUMBS.md) - .crumb json files created to trace inputs for all token generation

Each package has its own README with detailed usage information:

- [hear/README.md](hear/README.md)
- [see/README.md](see/README.md)
- [think/README.md](think/README.md)
- [dream/README.md](dream/README.md)
