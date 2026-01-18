# Observe Module

Multimodal capture and AI-powered analysis of desktop activity.

## Commands

| Command | Purpose |
|---------|---------|
| `sol observer` | Screen and audio capture (auto-detects platform) |
| `sol observe-linux` | Screen and audio capture on Linux (direct) |
| `sol observe-macos` | Screen and audio capture on macOS (direct) |
| `sol transcribe` | Audio transcription with faster-whisper |
| `sol describe` | Visual analysis of screen recordings |
| `sol sense` | Unified observation coordination |

## Architecture

```
sol observer (platform-detected capture)
       ↓
   Raw media files (*.flac, *.webm, tmux_*.jsonl)
       ↓
sol sense (coordination)
   ├── sol transcribe → audio.jsonl
   └── sol describe → screen.jsonl
```

## Observer State Machine

The Linux observer operates in three modes based on activity:

```
          SCREENCAST
         ↗         ↘
    (screen)    (screen idle)
       ↑            ↓
     IDLE ←----→ TMUX
         (tmux active)
```

**Mode priority**: Screen activity always wins over tmux (user is physically present).

| Mode | Trigger | Captures |
|------|---------|----------|
| SCREENCAST | Screen active (not idle/locked/power-save) | Video + Audio |
| TMUX | Screen idle but tmux has recent client activity | Terminal content + Audio |
| IDLE | Both screen and tmux inactive | Audio only (if threshold met) |

**Segment boundaries** are triggered by:
- Transitions to/from SCREENCAST mode (user returns to or leaves desktop)
- Mute state changes
- 5-minute window elapsed

TMUX ↔ IDLE transitions do **not** trigger boundaries, allowing tmux segments to run the full 5-minute window like screencast segments.

## Key Components

- **observer.py** - Unified entry point with platform detection
- **linux/observer.py**, **macos/observer.py** - Platform-specific capture using native APIs
- **linux/screencast.py** - XDG Portal screencast with PipeWire + GStreamer
- **gnome/activity.py** - GNOME-specific activity detection (idle, lock, power save)
- **tmux/capture.py** - Tmux capture library (integrated into Linux observer for fallback capture)
- **sense.py** - File watcher that dispatches transcription and description jobs
- **transcribe.py** - Audio transcription with faster-whisper and sentence-level embeddings
- **describe.py** - Vision analysis with Gemini, category-based prompts
- **categories/** - Category-specific prompts for screen content (see [SCREEN_CATEGORIES.md](SCREEN_CATEGORIES.md))

## Output Formats

See [JOURNAL.md](JOURNAL.md) for detailed extract schemas:
- Audio transcripts: `audio.jsonl` with timestamps (speaker detection not included)
- Screen analysis: `screen.jsonl` with frame-by-frame categorization

## Configuration

Requires `JOURNAL_PATH` environment variable. API keys for transcription/vision services configured in `.env`.
