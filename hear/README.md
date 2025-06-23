# sunstone-hear

Audio recording and transcription tools powered by the Gemini API. This module detects microphone and loopback devices, records audio in chunks and transcribes speech using Gemini.

## Installation

```bash
pip install -e .
```

All Python dependencies are declared in `pyproject.toml`. System packages for sound input/output may be required.

## Usage

After setting the `GOOGLE_API_KEY` environment variable you can run the `gemini-mic` command to capture audio and `gemini-transcribe` to convert the recordings to text. These commands correspond to the `capture.py` and `transcribe.py` modules respectively:

```bash
gemini-mic <save-directory> [-d] [-t SECONDS]
gemini-transcribe <save-directory> [-p PROMPT] [-i SECONDS]
```

Use `-d` to enable debug mode and `-t` to adjust the speech processing interval.
The `gemini-transcribe` command accepts `-p` to specify a prompt file and `-i`
to control the polling interval. Crash tracebacks are logged to standard error
by default.
