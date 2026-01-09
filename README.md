<img src="docs/static/logo.png" alt="solstone Logo" width="300">

# solstone

Navigate Life Intelligently

solstone is an open source project maintained by [sol pbc](https://solpbc.org).

A Python-based desktop journaling toolkit that captures screen and audio activity, processes it with AI, and provides intelligent navigation through a web interface. All data is organized in a local journal directory with daily folders, enabling temporal analysis and full ownership of your information.

## Platform Support

- Linux
- macOS

## Key Capabilities

- **Multimodal Capture** - Screen recording and audio capture with automatic segmentation
- **AI Transcription** - Audio transcription with faster-whisper and voice embeddings
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

## Getting Started

See **[docs/INSTALL.md](docs/INSTALL.md)** for complete setup instructions including system dependencies, API keys, and first-run configuration.

## Documentation

| Topic | Document |
|-------|----------|
| **Installation and setup** | [docs/INSTALL.md](docs/INSTALL.md) |
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
