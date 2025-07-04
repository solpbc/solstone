# sunstone-hear

Audio recording and transcription tools powered by the Gemini API. This module detects microphone and loopback devices, records audio in chunks and transcribes speech using Gemini.

## Installation

```bash
pip install -e .
```

All Python dependencies are declared in `pyproject.toml`. System packages for sound input/output may be required.

## Usage

After setting the `GOOGLE_API_KEY` environment variable you can run the `gemini-mic` command to capture audio and `gemini-transcribe` to convert the recordings to text.  The journal location is taken from the `JOURNAL_PATH` environment variable.  The paths for a single day are called **days**.  These commands correspond to the `capture.py` and `transcribe.py` modules respectively:

```bash
gemini-mic [-d] [-t SECONDS]
gemini-transcribe [-p PROMPT] [-i SECONDS]
```

Use `-d` to enable debug mode and `-t` to adjust the speech processing interval.
The `gemini-transcribe` command accepts `-p` to specify a prompt file and `-i` to
control the polling interval. Crash
tracebacks are logged to standard error by default.

## Architectural Overview

### Goals

`capture.py` focuses on reliably collecting audio from both the microphone and
system loopback so that the day’s conversations can be analysed later.  The
`transcribe.py` module converts those raw files into structured text that is easy
to search and reference.  Together they provide an accessible record of meetings
and on‑screen audio for downstream tools in the project.

Splitting these responsibilities into two processes improves reliability: the
lightweight recorder is isolated from the heavy work of denoising and API
requests.  Recordings can continue uninterrupted even if transcription is
stopped or moved to another machine.  The raw FLAC files remain in the journal
until you choose to delete them, so you can reprocess or re-transcribe later as
models improve.  Timestamped chunks also make it simple to align transcripts
with when events actually happened.

### Capture Workflow

The `gemini-mic` command is a wrapper around `AudioRecorder` in `capture.py`.
When launched it automatically detects the active microphone and loopback
devices using an ultrasonic tone.  Separate threads record from each device at
48 kHz and enqueue chunks.  Every `timer_interval` seconds the recorder merges
the queued buffers into a stereo FLAC file saved under the current day in the
journal.  The right channel is delayed by 100 ms so microphone and system audio
line up, and debug mode can store the individual buffers for troubleshooting.
The resulting files are named with a timestamp and the `_raw.flac` suffix.

### Transcription Workflow

`gemini-transcribe` watches the latest day directory for new `_raw.flac` files
and processes each as follows:

1. Denoise the recording with DeepFilterNet3.
2. Split stereo files into system and microphone channels and merge them with
   echo suppression.
3. Detect speech segments using Silero VAD and keep a small stash of audio so
   that speech spanning file boundaries is not lost.
4. Encode each segment to FLAC and submit the batch to the Gemini API using the
   prompt in `transcribe.txt`.
5. Save the returned JSON transcript next to the original recording and create a
   crumb entry for provenance.

Transcriptions are retried once on failure, and a `--repair` mode can process a
previous day to ensure no recordings are missed.

### WebSocket Streaming

Running `gemini-mic` with `--ws-port` starts a WebSocket server broadcasting the
new stereo chunk every second. Each message is a binary frame containing
little-endian `float32` samples interleaved as stereo pairs (`mic`, `system`)
at 16 kHz. The chunk size corresponds to the audio collected during that
one-second interval.

You can load the bytes in Python with:

```python
samples = np.frombuffer(msg, dtype=np.float32).reshape(-1, 2)
```
