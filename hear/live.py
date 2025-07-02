import argparse
import asyncio
import io
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets
from dotenv import load_dotenv
from google import genai
from silero_vad import load_silero_vad

from hear.audio_utils import SAMPLE_RATE, detect_speech
from hear.gemini import transcribe_segments

MODEL = "gemini-2.5-flash"


async def live_loop(ws_url: str, client, prompt_text: str, entities_text: str) -> None:
    vad = load_silero_vad()
    stash = np.array([], dtype=np.float32)
    processed_seconds = 0.0

    async with websockets.connect(ws_url) as ws:
        async for msg in ws:
            chunk = np.frombuffer(msg, dtype=np.float32).reshape(-1, 2)
            mono = chunk.mean(axis=1)
            stash = np.concatenate((stash, mono))
            segments, stash = detect_speech(vad, "live", stash)
            for seg in segments:
                start_time = processed_seconds + seg["offset"]
                start_str = time.strftime("%H:%M:%S", time.gmtime(start_time))
                audio_int16 = (np.clip(seg["data"], -1.0, 1.0) * 32767).astype(np.int16)
                buf = io.BytesIO()
                sf.write(buf, audio_int16, SAMPLE_RATE, format="FLAC")
                try:
                    result = transcribe_segments(
                        client,
                        MODEL,
                        prompt_text,
                        entities_text,
                        [{"start": start_str, "source": "mix", "bytes": buf.getvalue()}],
                    )
                    print(json.dumps(result, indent=2))
                except Exception as e:
                    logging.error("Transcription error: %s", e)
            processed_seconds += (len(mono) - len(stash)) / SAMPLE_RATE


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Live transcription from WebSocket")
    parser.add_argument("journal", type=Path, help="Journal directory containing entities.md")
    parser.add_argument("--ws-url", required=True, help="WebSocket URL from gemini-mic")
    parser.add_argument(
        "-p",
        "--prompt",
        type=Path,
        default=Path(__file__).with_name("transcribe.txt"),
        help="Path to the system prompt text",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Error: GOOGLE_API_KEY not found in environment.")

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    entities_path = args.journal / "entities.md"
    if not entities_path.is_file():
        parser.error(f"entities file not found: {entities_path}")

    prompt_text = args.prompt.read_text().strip()
    entities_text = entities_path.read_text().strip()
    client = genai.Client(api_key=api_key)

    asyncio.run(live_loop(args.ws_url, client, prompt_text, entities_text))


if __name__ == "__main__":
    main()
