# Observe Module

Multimodal capture and AI-powered analysis of desktop activity.

## Commands

| Command | Purpose |
|---------|---------|
| `observe-gnome` | Screen and audio capture on Linux/GNOME |
| `observe-macos` | Screen and audio capture on macOS |
| `observe-transcribe` | Audio transcription with speaker diarization |
| `observe-describe` | Visual analysis of screen recordings |
| `observe-sense` | Unified observation coordination |

## Architecture

```
observe-gnome/macos (capture)
       ↓
   Raw media files (*.flac, *.webm)
       ↓
observe-sense (coordination)
   ├── observe-transcribe → audio.jsonl
   └── observe-describe → screen.jsonl
```

## Key Components

- **gnome/observer.py**, **macos/observer.py** - Platform-specific capture using native APIs
- **sense.py** - File watcher that dispatches transcription and description jobs
- **transcribe.py** - Audio processing with Whisper/Rev.ai and pyannote diarization
- **describe.py** - Vision analysis with Gemini, category-based prompts
- **categories/** - Category-specific prompts for screen content (see [SCREEN_CATEGORIES.md](SCREEN_CATEGORIES.md))

## Output Formats

See [JOURNAL.md](JOURNAL.md) for detailed extract schemas:
- Audio transcripts: `audio.jsonl` with speaker turns and timestamps
- Screen analysis: `screen.jsonl` with frame-by-frame categorization

## Configuration

Requires `JOURNAL_PATH` environment variable. API keys for transcription/vision services configured in `.env`.
