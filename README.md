<img src="docs/static/logo.png" alt="solstone Logo" width="300">

# solstone

Navigate Life Intelligently

solstone is an open source project maintained by sol pbc.

A Python-based desktop journaling toolkit that captures screen and audio activity, processes it with AI, and provides intelligent navigation through a web interface. All data is organized in a local journal directory with daily folders, enabling temporal analysis and full ownership of your information.

## Platform Support

- Linux (GNOME/X11 with PipeWire)
- macOS

## Key Capabilities

- **Multimodal Capture** - Screen recording and audio capture with automatic segmentation
- **AI Transcription** - Speaker-diarized audio transcription with multiple backend support
- **Visual Analysis** - AI-powered screen content categorization and extraction
- **Intelligent Insights** - Automated daily summaries, meeting detection, and entity extraction
- **Facet Organization** - Group content by project or context (work, personal, etc.)
- **Web Interface** - Review transcripts, calendar views, entity tracking, and AI chat
- **Agent System** - Extensible AI agents with MCP tool integration

## Architecture

```
observe (capture) --> think (analyze) --> convey (view)
     |                     |                   |
  raw media           JSON extracts        Flask web UI
  (flac, webm)         (jsonl)
                           |
                       muse (AI agents)
```

## Requirements

- Python 3.10 or later
- At least one AI API key (Google, OpenAI, or Anthropic)
- Platform-specific dependencies for screen/audio capture

## Quick Start

1. Install the package:
   ```bash
   pip install -e .
   ```

2. Configure environment (copy `.env.example` to `.env` and add your settings):
   ```bash
   JOURNAL_PATH=/path/to/your/journal
   GOOGLE_API_KEY=your-api-key
   ```

3. Start the supervisor (handles capture and processing):
   ```bash
   think-supervisor
   ```

4. Launch the web interface:
   ```bash
   convey
   ```

## Documentation

| Topic | Document |
|-------|----------|
| Journal structure and data formats | [docs/JOURNAL.md](docs/JOURNAL.md) |
| Capture and observation | [docs/OBSERVE.md](docs/OBSERVE.md) |
| Processing and insights | [docs/THINK.md](docs/THINK.md) |
| Web interface | [docs/CONVEY.md](docs/CONVEY.md) |
| App development | [docs/APPS.md](docs/APPS.md) |
| AI agents and Cortex | [docs/CORTEX.md](docs/CORTEX.md) |
| Message bus protocol | [docs/CALLOSUM.md](docs/CALLOSUM.md) |
| Troubleshooting | [docs/DOCTOR.md](docs/DOCTOR.md) |

## Development

See [AGENTS.md](AGENTS.md) for development guidelines, coding standards, and testing instructions.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution terms.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0-only). See [LICENSE](LICENSE) for details.
