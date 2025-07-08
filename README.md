# sunstone
Navigate Work Intelligently

A comprehensive collection of multimodal AI utilities for workplace audio, visual, and cognitive processing.
Captured files are organised under a **journal** directory containing daily `YYYYMMDD` folders.

## Features

- **See** 👁️ - Screenshot capture and visual comparison with the `screen-watch` command
  and `screen-describe`. The `gemini-see` wrapper keeps both running in a loop. The
  `reduce-screen` command condenses diff descriptions into shorter Markdown.
- **Hear** 👂 - Audio recording and transcription with `gemini-mic` and `gemini-transcribe`.
  Use `gemini-hear` to run them together or `gemini-live` for real time transcripts.
- **Think** 🧠 - Data analysis and AI-powered insights via commands like `ponder`,
  `cluster` and `process-day`.
- **Dream** 🌐 - Run `dream --password PASSWORD` for a web UI to review entities and meetings.
- **Serve** 🛰️ - `ponder-mcp` launches an OAuth-protected MCP server for searching ponders and occurrences.

## Quick Start

```bash
pip install -e .
```

Set `JOURNAL_PATH` to the folder where recordings should be stored and
`GOOGLE_API_KEY` for Gemini access.

Use `gemini-hear` together with `gemini-see` to run the audio and visual capture
services in continuous loops. `gemini-hear` wraps `gemini-mic` and
`gemini-transcribe`, while `gemini-see` wraps `screen-watch` and
`screen-describe`. Launch `dream` to review the captured data in your browser.

## Documentation

Formats:

- [JOURNAL.md](JOURNAL.md) - details about the structure and contents of the **journal** directory
- [CRUMBS.md](CRUMBS.md) - .crumb json files created to trace inputs for all token generation

Each package has its own README with detailed usage information:

- [hear/README.md](hear/README.md)
- [see/README.md](see/README.md)
- [think/README.md](think/README.md)
- [dream/README.md](dream/README.md)
