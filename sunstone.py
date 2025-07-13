"""List available sunstone CLI commands."""

COMMANDS = [
    ("gemini-mic", "Record audio and save FLAC files", "[-d] [-t SECONDS] [--ws-port PORT]"),
    ("gemini-transcribe", "Transcribe FLAC files using Gemini", "[-p PROMPT] [--repair DAY]"),
    ("sunstone-hear", "Run gemini-mic and gemini-transcribe continuously", ""),
    ("gemini-live", "Live transcription from WebSocket", "--ws-url URL [--whisper]"),
    ("screen-watch", "Capture screenshots and compare with cached versions", "[--min PIXELS]"),
    ("screen-describe", "Describe screenshot diffs using Gemini", "[--scan DAY] [--repair DAY]"),
    (
        "sunstone-see",
        "Run screen-watch and screen-describe in a loop",
        "[interval] [additional args]",
    ),
    ("ponder", "Summarize a day's journal with Gemini", "DAY [-f PROMPT] [-p] [-c] [--force]"),
    ("cluster", "Generate a Markdown report from JSON files", "DAY"),
    (
        "reduce-screen",
        "Summarize monitor JSON files with Gemini",
        "DAY [--prompt FILE] [--force] [-d] [--start HH:MM] [--end HH:MM]",
    ),
    ("dream", "Run the Dream review web service", "--password PASSWORD [--port PORT]"),
    ("entity-roll", "Merge entity data across days", "[--force] [--day DAY]"),
    ("journal-stats", "Scan a journal and print statistics", ""),
    (
        "dream-extract",
        "Chunk a media file into the journal",
        "MEDIA TIMESTAMP [--see BOOL] [--hear BOOL] [--see-sample SECS]",
    ),
    (
        "process-day",
        "Run daily processing tasks on a journal day",
        "[--day DAY] [--force] [--repair] [--rebuild]",
    ),
    ("ponder-mcp", "Start MCP search server", "[--port PORT] [--stdio]"),
    ("journal-index", "Index ponder markdown and occurrence files", "[--rescan] [-q QUERY]"),
    ("empty-trash", "Permanently delete trashed audio files", "[--dry-run] [--jobs N]"),
]


def main() -> None:
    """Print available sunstone commands with short descriptions."""
    print("Available commands:\n")
    for name, desc, args in COMMANDS:
        if args:
            print(f"{name} {args}\n    {desc}")
        else:
            print(f"{name}\n    {desc}")


if __name__ == "__main__":
    main()
