#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Rev.ai transcription CLI and legacy conversion utilities.

This module provides:
- CLI for Rev.ai transcription and JSON conversion
- Legacy convert_revai_to_solstone() for per-sentence segmentation
- Backwards-compatible transcribe_file() wrapper

For the STT backend interface, see observe.transcribe.revai.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Import from the new backend module
from observe.transcribe.revai import (
    convert_to_segments,
    transcribe_file,
)


def die(msg, code=1):
    logging.error(msg)
    sys.exit(code)


def _format_timestamp(seconds: float | None) -> str:
    """Format seconds as HH:MM:SS string."""
    if seconds is None:
        return "00:00:00"
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def convert_revai_to_solstone(revai_json: dict, per_speaker: bool = False) -> list:
    """Convert Rev.ai transcript format to solstone transcript format.

    Args:
        revai_json: Dict with Rev.ai transcript structure (monologues with elements)
        per_speaker: If True, output one entry per speaker turn (monologue).
            If False (default), split into sentences within each speaker turn.

    Returns:
        List of transcript entries in solstone format
    """
    if per_speaker:
        # Use the new per-speaker conversion from backend
        segments = convert_to_segments(revai_json)
        result = []
        for seg in segments:
            entry = {
                "start": _format_timestamp(seg["start"]),
                "source": "import",
                "speaker": seg["speaker"],
                "text": seg["text"],
            }

            # Add description based on confidence
            if seg.get("confidence") is not None:
                if seg["confidence"] < 0.7:
                    entry["description"] = "low confidence"
                elif seg["confidence"] > 0.95:
                    entry["description"] = "clear"

            result.append(entry)
        return result

    # Legacy per-sentence segmentation
    result = []

    if "monologues" not in revai_json:
        return result

    for monologue in revai_json["monologues"]:
        speaker = monologue.get("speaker", 0) + 1  # Rev uses 0-based, we use 1-based
        elements = monologue.get("elements", [])

        # Build sentences from elements
        current_text = ""
        start_ts = None
        confidences = []

        for elem in elements:
            if elem["type"] == "text":
                # Track first timestamp
                if start_ts is None and elem.get("ts") is not None:
                    start_ts = elem["ts"]

                # Add word
                current_text += elem["value"]

                # Track confidence
                if elem.get("confidence") is not None:
                    confidences.append(elem["confidence"])

            elif elem["type"] == "punct":
                # Add punctuation
                current_text += elem["value"]

                # If sentence-ending punctuation, create entry
                if elem["value"] in [".", "!", "?"] and current_text.strip():
                    entry = {
                        "start": _format_timestamp(start_ts),
                        "source": "import",
                        "speaker": speaker,
                        "text": current_text.strip(),
                    }

                    # Add description based on confidence
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        if avg_confidence < 0.7:
                            entry["description"] = "low confidence"
                        elif avg_confidence > 0.95:
                            entry["description"] = "clear"

                    result.append(entry)

                    # Reset for next sentence
                    current_text = ""
                    start_ts = None
                    confidences = []

        # Handle any remaining text without sentence-ending punctuation
        if current_text.strip():
            entry = {
                "start": _format_timestamp(start_ts),
                "source": "import",
                "speaker": speaker,
                "text": current_text.strip(),
            }

            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                if avg_confidence < 0.7:
                    entry["description"] = "low confidence"
                elif avg_confidence > 0.95:
                    entry["description"] = "clear"

            result.append(entry)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Rev AI transcription CLI (high-quality + diarization). "
        "If a .json file is provided, converts it to solstone format instead of transcribing."
    )
    parser.add_argument(
        "media", help="Path to audio/video file or Rev AI JSON file to convert"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "-o", "--output", help="Write JSON to this path (default: stdout)"
    )
    parser.add_argument("--language", default="en", help="ISO code (default: en)")
    parser.add_argument(
        "--model",
        default="fusion",
        choices=["fusion", "machine", "low_cost", "human"],
        help='Rev transcriber (default: "fusion" for highest quality)',
    )
    parser.add_argument(
        "--diarization-type",
        default="premium",
        choices=["standard", "premium"],
        help='Diarization type (default: "premium")',
    )
    parser.add_argument(
        "--forced-alignment",
        action="store_true",
        help="Improve per-word timestamps where supported",
    )
    parser.add_argument(
        "--speakers-count",
        type=int,
        default=None,
        help="If known, hint total unique speakers (improves diarization)",
    )
    parser.add_argument(
        "--speaker-channels-count",
        type=int,
        default=None,
        help="If multichannel file with distinct speakers per channel (extra cost)",
    )
    parser.add_argument(
        "--remove-disfluencies",
        action="store_true",
        help="Remove ums/uhs + atmospherics (English/Spanish only)",
    )
    parser.add_argument(
        "--filter-profanity",
        action="store_true",
        help="Replace profanities with asterisks",
    )
    parser.add_argument(
        "--skip-punctuation", action="store_true", help="Disable punctuation in output"
    )
    parser.add_argument(
        "--entities",
        nargs="*",
        help="Custom vocabulary terms to improve recognition (space-separated)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.5,
        help="Seconds between status polls (default: 2.5)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60 * 30,
        help="Overall timeout in seconds (default: 30m)",
    )
    parser.add_argument(
        "--per-speaker",
        action="store_true",
        help="Output one entry per speaker turn instead of per sentence (for JSON conversion)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    media_path = Path(args.media).expanduser().resolve()
    if not media_path.exists() or not media_path.is_file():
        die(f"File not found: {media_path}")

    # Check if input is a JSON file - if so, convert instead of transcribe
    if media_path.suffix.lower() == ".json":
        logging.info(
            "Detected JSON input - converting Rev AI format to solstone format"
        )

        # Load the Rev AI JSON
        try:
            with open(media_path, "r", encoding="utf-8") as f:
                revai_data = json.load(f)
        except json.JSONDecodeError as e:
            die(f"Invalid JSON file: {e}")

        # Convert to solstone format
        solstone_transcript = convert_revai_to_solstone(
            revai_data, per_speaker=args.per_speaker
        )

        # Output the result
        if args.output:
            out_path = Path(args.output).expanduser().resolve()
            out_path.write_text(
                json.dumps(solstone_transcript, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logging.info("Wrote converted transcript to %s", out_path)
        else:
            print(json.dumps(solstone_transcript, indent=2, ensure_ascii=False))

        return

    # Otherwise, do normal transcription via the backend
    # (backend handles load_dotenv internally)

    # Build config from CLI args
    config = {
        "language": args.language,
        "model": args.model,
        "diarization_type": args.diarization_type,
        "forced_alignment": args.forced_alignment,
        "remove_disfluencies": args.remove_disfluencies,
        "filter_profanity": args.filter_profanity,
        "skip_punctuation": args.skip_punctuation,
        "poll_interval": args.poll_interval,
        "timeout": args.timeout,
    }
    if args.speakers_count is not None:
        config["speakers_count"] = args.speakers_count
    if args.speaker_channels_count is not None:
        config["speaker_channels_count"] = args.speaker_channels_count
    if args.entities:
        config["entities"] = args.entities

    try:
        transcript = transcribe_file(media_path, config)
    except ValueError as e:
        die(str(e))
    except (RuntimeError, TimeoutError) as e:
        die(str(e))

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.write_text(
            json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logging.info("Wrote %s", out_path)
    else:
        print(json.dumps(transcript, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
