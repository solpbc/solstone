import argparse
import asyncio
import io
import logging
import os
import threading
import time
from typing import Callable, Optional

import numpy as np
import soundfile as sf
import websockets
from dbus_next.aio import MessageBus
from dbus_next.constants import BusType
from google import genai
from google.genai import types
from silero_vad import load_silero_vad

from hear.audio_utils import SAMPLE_RATE, detect_speech
from think.models import GEMINI_FLASH
from think.utils import setup_cli

MODEL = GEMINI_FLASH + "-lite-preview-06-17"


async def transcribe_light(client, model: str, audio_bytes: bytes) -> str:
    parts = [
        types.Part.from_text(
            text="Please transcribe any spoken words or utterances you hear in this audio clip, accuracy is important."
        ),
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/flac"),
    ]

    contents = [types.Content(role="user", parts=parts)]

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=1024,
            system_instruction="You are a precise audio transcriptionist. Listen to the audio and write down exactly what words are spoken. Return only the spoken text with no punctuation, formatting, or additional comments.",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="text/plain",
        ),
    )
    result = response.text.strip()
    logging.info("Transcription result: %s", result)
    return result


async def identify_active_speaker(
    client,
    speaker_state,
    *,
    event_callback: Callable[[dict], None] | None = None,
    msg_id: int = 0,
) -> None:
    """Capture a screenshot and ask Gemini who is speaking, updating shared state."""
    try:
        from see.screen_dbus import take_screenshot

        screenshot_start = time.time()
        bus = await MessageBus(bus_type=BusType.SESSION).connect()
        screenshot_bytes = await take_screenshot(bus)
        screenshot_duration = time.time() - screenshot_start
        logging.debug(f"Screenshot capture took {screenshot_duration:.3f}s")
    except Exception as e:  # pragma: no cover - DBus not available in tests
        logging.error("Screenshot capture failed: %s", e)
        speaker_state["current_speaker"] = "Unknown"
        speaker_state["task_running"] = False
        return

    parts = [
        types.Part.from_text(
            text="Analyze this desktop screenshot and look for any visible meetings to identify the currently active speaker."
        ),
        types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png"),
    ]
    contents = [types.Content(role="user", parts=parts)]

    try:
        gemini_start = time.time()
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=32 + 512,
                system_instruction="Examine this screenshot looking ONLY for video conferencing apps like Zoom, Teams, Meet, or WebEx. Then if found, identify who is actively speaking right now based on visual indicators such as the speaker's frame being highlighted or outlined differently. Return only the speaker's first name, or 'Unknown' if no meeting is visible or nobody is clearly speaking.",
                thinking_config=types.ThinkingConfig(thinking_budget=512),
                response_mime_type="text/plain",
            ),
        )
        gemini_duration = time.time() - gemini_start
        logging.debug(f"Gemini speaker identification took {gemini_duration:.3f}s")
        speaker_name = response.text.strip()
        logging.info("Meeting Speaker: %s", speaker_name)
        speaker_state["current_speaker"] = speaker_name
        if event_callback:
            event_callback(
                {
                    "event": "speaker",
                    "id": msg_id,
                    "speaker": speaker_name,
                }
            )
    except Exception as e:
        logging.error("Gemini meeting request failed: %s", e)
        speaker_state["current_speaker"] = "Unknown"
    finally:
        speaker_state["task_running"] = False


async def transcribe_audio_segments(
    segments,
    client,
    speaker_state,
    *,
    event_callback: Callable[[dict], None] | None = None,
    message_id: int = 0,
) -> None:
    """Transcribe audio segments in a separate async task."""
    try:
        combined_audio = np.concatenate([seg["data"] for seg in segments])
        audio_int16 = (np.clip(combined_audio, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        sf.write(buf, audio_int16, SAMPLE_RATE, format="FLAC")

        transcribe_start = time.time()
        text = await transcribe_light(
            client,
            MODEL,
            buf.getvalue(),
        )
        transcribe_duration = time.time() - transcribe_start
        logging.debug(f"Transcription took {transcribe_duration:.3f}s")

        speaker = None
        if speaker_state is not None:
            speaker = speaker_state.get("current_speaker", "")
            prefix = f"{speaker}: " if speaker else ""
            print(f"{prefix}{text}")
        else:
            print(text)
        if event_callback:
            event_callback(
                {
                    "event": "transcription",
                    "id": message_id,
                    "text": text,
                    "speaker": speaker or "",
                }
            )

    except Exception as e:
        logging.error("Transcription error: %s", e)


async def handle_audio_message(
    msg: bytes,
    vad,
    stash: np.ndarray,
    client,
    speaker_state: dict | None,
    *,
    event_callback: Callable[[dict], None] | None = None,
    message_id: int = 0,
) -> np.ndarray:
    """Handle a single audio message from WebSocket."""
    try:
        chunk = np.frombuffer(msg, dtype=np.float32).reshape(-1, 2)
        mono = chunk.mean(axis=1)
        stash = np.concatenate((stash, mono))

        # Trigger screenshot capture if stash grows beyond 3 seconds and no task is running
        if (
            speaker_state is not None
            and len(stash) / SAMPLE_RATE > 3
            and not speaker_state.get("task_running", False)
        ):
            speaker_state["task_running"] = True
            asyncio.create_task(
                identify_active_speaker(
                    client,
                    speaker_state,
                    event_callback=event_callback,
                    msg_id=message_id,
                )
            )

        segments, stash = detect_speech(vad, "live", stash)

        # Launch transcription as async task if segments found
        if segments:
            asyncio.create_task(
                transcribe_audio_segments(
                    segments,
                    client,
                    speaker_state,
                    event_callback=event_callback,
                    message_id=message_id,
                )
            )

        return stash
    except Exception as e:
        logging.error("Error processing audio chunk: %s", e)
        return stash


async def live_loop(
    ws_url: str,
    client,
    use_speaker: bool = False,
    *,
    event_callback: Callable[[dict], None] | None = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    vad = load_silero_vad()
    stash = np.array([], dtype=np.float32)
    speaker_state = (
        {"task_running": False, "current_speaker": "Unknown"} if use_speaker else None
    )
    processed_seconds = 0.0
    max_retries = 5
    retry_delay = 2.0

    msg_id = 0
    for attempt in range(max_retries):
        if stop_event and stop_event.is_set():
            break
        try:
            logging.info(
                f"Connecting to WebSocket (attempt {attempt + 1}/{max_retries})"
            )
            async with websockets.connect(ws_url) as ws:
                logging.info("WebSocket connected successfully")
                if event_callback:
                    event_callback({"event": "joined"})
                try:
                    async for msg in ws:
                        if stop_event and stop_event.is_set():
                            await ws.close()
                            break
                        stash = await handle_audio_message(
                            msg,
                            vad,
                            stash,
                            client,
                            speaker_state,
                            event_callback=event_callback,
                            message_id=msg_id,
                        )
                        msg_id += 1
                        processed_seconds += (
                            len(msg) // 8
                        ) / SAMPLE_RATE  # Approximate calculation
                finally:
                    if event_callback:
                        event_callback({"event": "left"})
                # If we reach here, connection closed normally
                logging.info("WebSocket connection closed normally")
                break

        except (
            websockets.exceptions.ConnectionClosedError,
            websockets.exceptions.ConnectionClosedOK,
            ConnectionError,
            BrokenPipeError,
        ) as e:
            logging.warning(f"WebSocket connection error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                logging.error("Max retries reached, giving up")
                raise
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt, shutting down gracefully")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                raise


def run(
    ws_url: str,
    use_speaker: bool = True,
    callback: Callable[[dict], None] | None = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Run the live transcription loop with optional callback."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Error: GOOGLE_API_KEY not found in environment.")
    client = genai.Client(api_key=api_key)
    asyncio.run(
        live_loop(
            ws_url,
            client,
            use_speaker,
            event_callback=callback,
            stop_event=stop_event,
        )
    )


def start_thread(
    ws_url: str, callback: Callable[[dict], None], use_speaker: bool = True
) -> threading.Thread:
    """Run ``run`` in a background thread and return it."""
    stop_event = threading.Event()

    def _run() -> None:
        run(ws_url, use_speaker, callback, stop_event)

    thread = threading.Thread(target=_run, daemon=True)
    thread.stop_event = stop_event  # type: ignore[attr-defined]
    thread.start()
    return thread


def stop_thread(thread: threading.Thread | None) -> None:
    """Signal the given live thread to stop."""
    if not thread:
        return
    stop_event = getattr(thread, "stop_event", None)
    if isinstance(stop_event, threading.Event):
        stop_event.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live transcription from WebSocket")
    parser.add_argument("--ws-url", required=True, help="WebSocket URL from gemini-mic")
    parser.add_argument(
        "--speaker",
        action="store_true",
        help="Enable speaker identification (off by default)",
    )
    args = setup_cli(parser)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Error: GOOGLE_API_KEY not found in environment.")

    client = genai.Client(api_key=api_key)

    asyncio.run(live_loop(args.ws_url, client, args.speaker))


if __name__ == "__main__":
    main()
