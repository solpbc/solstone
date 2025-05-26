# sunstone-hear

Audio recording and transcription tools powered by the Gemini API. This module detects microphone and loopback devices, records audio in chunks and transcribes speech using Gemini.

## Installation

```bash
pip install -e .
```

All Python dependencies are declared in `pyproject.toml`. System packages for sound input/output may be required.

## Usage

After setting the `GOOGLE_API_KEY` environment variable you can run the `gemini-mic` command which will capture audio and store transcriptions:

```bash
gemini-mic <save-directory> [-d] [-t SECONDS]
```

Use `-d` to enable debug mode and `-t` to adjust the speech processing interval.
