<img src="docs/static/logo.png" alt="solstone Logo" width="300">

# solstone

Navigate Life Intelligently

solstone is an open-source, local-first AI journaling toolkit. It captures screen and audio activity, processes it with AI agents, and presents everything through a searchable web interface. All data stays on your machine in daily journal directories — no cloud, no subscriptions, full ownership.

Python 3.10+, Linux + macOS, AGPL-3.0-only, maintained by [sol pbc](https://solpbc.org).

<img src="docs/static/screenshot-home.png" alt="solstone daily dashboard" width="800">

*The daily dashboard with facet tabs, daily goal, todos, upcoming events, and detected entities.*

## Architecture

```text
  +---------+       +----------------+       +---------+
  | observe | ----> |    journal     | ----> |  think  |
  | capture |       | YYYYMMDD/ dirs |       | process |
  +---------+       | media, jsonl,  |       | index   |
                    | entities       |       +----+----+
                    +-------+--------+            |
                            ^                     |
                            |  agent outputs      |
                       +----+----+                |
                       | cortex  | <--------------+
                       | agents  |
                       +---------+
                            |
  ==== callosum (event bus) | ==========================
                            |
                     +------+------+
                     |   convey    |
                     | web UI      |
                     +-------------+
```

- **observe** — Captures audio (PipeWire on Linux, sck-cli on macOS) and screen activity. Produces FLAC audio, WebM screen recordings, and timestamped metadata.
- **think** — Transcribes audio (faster-whisper), analyzes screen captures, extracts entities, detects meetings, and indexes everything into SQLite. Runs 30 configurable agent/generator templates from `muse/`.
- **cortex** — Orchestrates agent execution. Receives events, dispatches agents, writes results back to the journal.
- **callosum** — Async message bus connecting all services. Enables event-driven coordination between observe, think, cortex, and convey.
- **convey** — Flask-based web interface with 17 pluggable apps for navigating journal data.
- **journal** — `JOURNAL_PATH/YYYYMMDD/` daily directories. The single source of truth — transcripts, media, entities, agent outputs, and the SQLite index all live here.

## Key Features

- **Multimodal capture** — Continuous audio recording with voice activity detection, plus periodic screen capture with AI-powered categorization.
- **Transcription and speaker identification** — faster-whisper transcription with voice embeddings (resemblyzer) for speaker diarization.
- **Entity extraction** — People, projects, and concepts extracted from transcripts and tracked across time.
- **Facet organization** — Group content by project or context (e.g., work, personal, client-name) with scoped views across all apps.
- **AI agents** — 30 agent configurations for activities, meetings, scheduling, knowledge graphs, research, media analysis, and more. Extensible via the agent skill framework.
- **Searchable index** — Full-text search across transcripts, entities, and agent outputs via SQLite.
- **Local-first** — All data in daily journal directories on your filesystem. No cloud dependency. Configurable AI providers (Google Gemini, OpenAI, Anthropic).

## Web Interface

Convey currently includes 17 discoverable apps for browsing, operating, and maintaining your journal.

<img src="docs/static/screenshot-transcripts.png" alt="solstone transcript viewer" width="800">

- Daily dashboard, calendar, and scheduling.
- Transcript browser with dual-timeline navigation.
- AI chat with configurable providers.
- Entity tracking, search, and statistics.
- Speaker identification and management.
- Health monitoring, token usage, and system settings.

## Quick Start

```bash
git clone https://github.com/solpbc/solstone.git
cd solstone
make install

# Configure environment
cp .env.example .env
# Add at minimum: GOOGLE_API_KEY=your-key
# See docs/PROVIDERS.md for all supported providers

# Start the full stack
sol supervisor

# Open the web interface
# (prints URL on startup, default http://localhost:<port>)
```

See [docs/INSTALL.md](docs/INSTALL.md) for platform-specific dependencies, detailed configuration, and first-run guidance.

## CLI

solstone is operated through the unified `sol` command (33 subcommands).

```bash
sol                    # Status overview and command list
sol supervisor         # Start the full stack (capture + processing + web)
sol chat               # Interactive AI chat from the terminal
sol transcribe <file>  # Transcribe an audio file
sol indexer            # Rebuild the search index
sol screenshot /       # Capture a screenshot of the web UI
```

Run `sol help` for the full command reference.

## Documentation

| Topic | Document |
|-------|----------|
| Installation and setup | [docs/INSTALL.md](docs/INSTALL.md) |
| Journal structure and data model | [docs/JOURNAL.md](docs/JOURNAL.md) |
| Capture pipeline | [docs/OBSERVE.md](docs/OBSERVE.md) |
| Processing and agents | [docs/THINK.md](docs/THINK.md) |
| Web interface | [docs/CONVEY.md](docs/CONVEY.md) |
| App development | [docs/APPS.md](docs/APPS.md) |
| Agent runtime | [docs/CORTEX.md](docs/CORTEX.md) |
| Message bus | [docs/CALLOSUM.md](docs/CALLOSUM.md) |
| AI provider configuration | [docs/PROVIDERS.md](docs/PROVIDERS.md) |
| Troubleshooting | [docs/DOCTOR.md](docs/DOCTOR.md) |
| Project direction | [docs/ROADMAP.md](docs/ROADMAP.md) |

## Development

See [AGENTS.md](AGENTS.md) for development guidelines, coding standards, and testing instructions.

Use `make dev` to run the full stack against test fixtures, `make ci` for pre-commit checks, and `sol screenshot` for UI testing workflows.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution terms.

## License

AGPL-3.0-only. See [LICENSE](LICENSE) for details.
Maintained by [sol pbc](https://solpbc.org).
