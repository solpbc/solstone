# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Media file metadata detection utilities."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .utils import load_prompt


def _load_system_prompt() -> str:
    """Load the system prompt from detect_created.txt file."""
    return load_prompt("detect_created", base_dir=Path(__file__).parent).text


def _extract_metadata(path: str) -> str:
    """Return metadata for *path* using exiftool if available."""
    cmd = [
        "exiftool",
        "-all",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return proc.stdout
    except Exception as exc:  # pragma: no cover - exiftool optional
        return f"Error extracting metadata: {exc}"


def _debug_write_content(content: str, path: str) -> None:
    """Write content to a debug file in /tmp for diagnosis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gemini_debug_{timestamp}_{os.path.basename(path)}.md"
    debug_path = os.path.join("/tmp", filename)

    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Debug: Content written to {debug_path}", file=sys.stderr)


def detect_created(
    path: str, original_filename: Optional[str] = None
) -> Optional[dict]:
    """Return creation time information for *path* using configured provider.

    Parameters
    ----------
    path : str
        Path to the file to analyze
    original_filename : Optional[str]
        Original filename if path is a temporary file
    """
    metadata = _extract_metadata(path)

    # Use original filename in header if provided, otherwise use the actual path
    display_path = original_filename if original_filename else path

    lines = [
        f"# exiftool -all output for {display_path}",
        "",
    ]

    # If we have an original filename and it's different from path, add a note
    if original_filename and original_filename != path:
        lines.extend(
            [
                f"Original filename: {original_filename}",
                f"(Analyzing temporary file: {path})",
                "",
            ]
        )

    lines.append(metadata)
    markdown = "\n".join(lines)

    # Debug: write content to temp file
    _debug_write_content(markdown, path)

    from muse.models import generate

    response_text = generate(
        contents=markdown,
        context="detect.created",
        temperature=0.3,
        max_output_tokens=256,
        thinking_budget=4096,
        system_instruction=_load_system_prompt(),
        json_output=True,
    )

    try:
        result = json.loads(response_text)

        # Convert UTC to local time if needed
        if result and result.get("utc") is True:
            day = result.get("day")
            time = result.get("time")

            if day and time:
                # Parse as UTC datetime
                utc_dt = datetime.strptime(f"{day}{time}", "%Y%m%d%H%M%S")
                utc_dt = utc_dt.replace(tzinfo=timezone.utc)

                # Convert to local timezone
                local_dt = utc_dt.astimezone()

                # Update result with local time
                result["day"] = local_dt.strftime("%Y%m%d")
                result["time"] = local_dt.strftime("%H%M%S")

        return result
    except json.JSONDecodeError:
        return None


def main():
    """Main CLI entry point for detect_created utility."""
    import argparse

    from .utils import setup_cli

    parser = argparse.ArgumentParser(
        description="Detect creation time information from media file metadata"
    )
    parser.add_argument("file_path", help="Path to the media file to analyze")

    args = setup_cli(parser)

    result = detect_created(args.file_path)
    if result is not None:
        print(json.dumps(result, indent=2))
    else:
        print("Failed to detect creation time information", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
