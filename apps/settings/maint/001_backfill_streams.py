# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Backfill stream.json markers into all journal segments.

Walks every segment in the journal and determines its stream identity using
a priority cascade of available signals:

  1. Existing stream.json marker (already tagged — preserve)
  2. audio.jsonl header "stream" field
  3. audio.jsonl header "remote" field -> stream_name(remote=...)
  4. audio.jsonl header "imported" field -> detect import type from raw path
  5. imported_audio.jsonl presence -> import stream from raw path extension
  6. Import reverse index (imports/*/segments.json cross-reference)
  7. audio.jsonl header "host" field -> stream_name(host=...)
  8. Tmux-only segment (tmux_*_screen.jsonl, no audio) -> host.tmux
  9. Hostname fallback -> stream_name(host=current_or_override)

After classification, segments are grouped by stream, sorted chronologically,
and prev/seq linkages are reconstructed. Stream state files are rebuilt.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
from pathlib import Path

from think.streams import (
    _strip_hostname,
    read_segment_stream,
    stream_name,
    write_segment_stream,
)
from think.utils import day_dirs, get_journal, segment_key, setup_cli

logger = logging.getLogger(__name__)

# Signal names for reporting
SIGNAL_EXISTING = "existing_marker"
SIGNAL_AUDIO_STREAM = "audio.jsonl_stream"
SIGNAL_AUDIO_REMOTE = "audio.jsonl_remote"
SIGNAL_AUDIO_IMPORTED = "audio.jsonl_imported"
SIGNAL_IMPORTED_JSONL = "imported_audio.jsonl"
SIGNAL_IMPORT_INDEX = "import_reverse_index"
SIGNAL_AUDIO_HOST = "audio.jsonl_host"
SIGNAL_TMUX_ONLY = "tmux_only_segment"
SIGNAL_HOSTNAME_FALLBACK = "hostname_fallback"


def _read_jsonl_header(path: Path) -> dict | None:
    """Read the first line of a JSONL file as a JSON dict."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if line:
                return json.loads(line)
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _import_source_from_raw(raw_path: str) -> str:
    """Detect import source type from the raw file path extension."""
    ext = os.path.splitext(raw_path)[1].lower()
    if ext == ".m4a":
        return "apple"
    if ext in {".txt", ".md", ".pdf"}:
        return "text"
    return "audio"


def _import_source_from_mime(mime_type: str) -> str:
    """Detect import source type from MIME type."""
    if "m4a" in mime_type or "mp4" in mime_type:
        return "apple"
    if mime_type.startswith("text/"):
        return "text"
    return "audio"


def build_import_reverse_index(journal_root: Path) -> dict[tuple[str, str], dict]:
    """Build mapping from (day, segment_key) to import metadata.

    Scans imports/*/segments.json to find which segments were created by
    which import, then reads import.json for source metadata.

    Returns:
        Dict mapping (day, segment_key) to import info dict with keys:
        - source: import source type (apple, text, audio, plaud)
        - timestamp: import timestamp
    """
    index: dict[tuple[str, str], dict] = {}
    imports_dir = journal_root / "imports"
    if not imports_dir.exists():
        return index

    for import_dir in imports_dir.iterdir():
        if not import_dir.is_dir():
            continue
        timestamp = import_dir.name

        # Read segments.json to find which segments this import created
        segments_path = import_dir / "segments.json"
        if not segments_path.exists():
            continue
        try:
            data = json.loads(segments_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        day = data.get("day", "")
        segments = data.get("segments", [])
        if not day or not segments:
            continue

        # Determine import source from import.json metadata
        source = "audio"  # default
        import_json = import_dir / "import.json"
        if import_json.exists():
            try:
                meta = json.loads(import_json.read_text(encoding="utf-8"))
                filename = meta.get("original_filename", "")
                mime = meta.get("mime_type", "")
                if filename:
                    source = _import_source_from_raw(filename)
                elif mime:
                    source = _import_source_from_mime(mime)
            except (json.JSONDecodeError, OSError):
                pass

        info = {"source": source, "timestamp": timestamp}
        for seg in segments:
            index[(day, seg)] = info

    return index


def _has_tmux_only(seg_dir: Path) -> bool:
    """Check if segment has tmux screen captures but no audio files."""
    has_tmux = False
    has_audio = False
    for f in seg_dir.iterdir():
        name = f.name
        if name.startswith("tmux_") and name.endswith("_screen.jsonl"):
            has_tmux = True
        if name.endswith((".flac", ".m4a", ".ogg", ".opus")):
            has_audio = True
        if name == "audio.jsonl" or name.endswith("_audio.jsonl"):
            has_audio = True
    return has_tmux and not has_audio


def classify_segment(
    seg_dir: Path,
    day: str,
    import_index: dict[tuple[str, str], dict],
    fallback_host: str,
) -> tuple[str, str]:
    """Determine stream name for a segment using the signal cascade.

    Returns:
        Tuple of (stream_name, signal_used).
    """
    seg = seg_dir.name

    # 1. Existing stream.json
    marker = read_segment_stream(seg_dir)
    if marker and marker.get("stream"):
        return marker["stream"], SIGNAL_EXISTING

    # 2-7. Check audio.jsonl variants
    for audio_name in ["audio.jsonl"]:
        audio_path = seg_dir / audio_name
        if audio_path.exists():
            header = _read_jsonl_header(audio_path)
            if header:
                # 2. Direct stream field
                if header.get("stream"):
                    return header["stream"], SIGNAL_AUDIO_STREAM

                # 3. Remote field
                if header.get("remote"):
                    try:
                        name = stream_name(remote=header["remote"])
                        return name, SIGNAL_AUDIO_REMOTE
                    except ValueError:
                        pass

                # 4. Imported field
                if header.get("imported"):
                    raw = header.get("raw", "")
                    source = _import_source_from_raw(raw) if raw else "audio"
                    try:
                        name = stream_name(import_source=source)
                        return name, SIGNAL_AUDIO_IMPORTED
                    except ValueError:
                        pass

    # Also check *_audio.jsonl variants (but not imported_audio.jsonl — handled below)
    for f in seg_dir.iterdir():
        if (
            f.name.endswith("_audio.jsonl")
            and f.name != "audio.jsonl"
            and f.name != "imported_audio.jsonl"
        ):
            header = _read_jsonl_header(f)
            if header:
                if header.get("stream"):
                    return header["stream"], SIGNAL_AUDIO_STREAM
                if header.get("remote"):
                    try:
                        name = stream_name(remote=header["remote"])
                        return name, SIGNAL_AUDIO_REMOTE
                    except ValueError:
                        pass
                if header.get("imported"):
                    raw = header.get("raw", "")
                    source = _import_source_from_raw(raw) if raw else "audio"
                    try:
                        name = stream_name(import_source=source)
                        return name, SIGNAL_AUDIO_IMPORTED
                    except ValueError:
                        pass

    # 5. imported_audio.jsonl presence
    imported_path = seg_dir / "imported_audio.jsonl"
    if imported_path.exists():
        header = _read_jsonl_header(imported_path)
        raw = header.get("raw", "") if header else ""
        source = _import_source_from_raw(raw) if raw else "audio"
        try:
            name = stream_name(import_source=source)
            return name, SIGNAL_IMPORTED_JSONL
        except ValueError:
            pass

    # 6. Import reverse index
    seg_key = segment_key(seg) or seg
    import_info = import_index.get((day, seg_key))
    if import_info:
        try:
            name = stream_name(import_source=import_info["source"])
            return name, SIGNAL_IMPORT_INDEX
        except ValueError:
            pass

    # 7. audio.jsonl host field (checked after import signals)
    audio_path = seg_dir / "audio.jsonl"
    if audio_path.exists():
        header = _read_jsonl_header(audio_path)
        if header and header.get("host"):
            try:
                name = stream_name(host=header["host"])
                return name, SIGNAL_AUDIO_HOST
            except ValueError:
                pass

    # 8. Tmux-only segment
    if _has_tmux_only(seg_dir):
        try:
            name = stream_name(host=fallback_host, qualifier="tmux")
            return name, SIGNAL_TMUX_ONLY
        except ValueError:
            pass

    # 9. Hostname fallback
    try:
        name = stream_name(host=fallback_host)
        return name, SIGNAL_HOSTNAME_FALLBACK
    except ValueError:
        # Absolute last resort — should never happen with a valid hostname
        return fallback_host.lower(), SIGNAL_HOSTNAME_FALLBACK


def _infer_fallback_host(journal_root: Path, override: str | None) -> str:
    """Determine the best fallback hostname.

    Checks existing stream state files for a host field before falling
    back to socket.gethostname(). Domain suffixes are stripped so dots
    remain reserved for stream qualifiers.
    """
    if override:
        return _strip_hostname(override)

    # Check existing streams/ state files for a host hint
    streams_dir = journal_root / "streams"
    if streams_dir.exists():
        for state_file in streams_dir.glob("*.json"):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if state.get("type") == "observer" and state.get("host"):
                    return _strip_hostname(state["host"])
            except (json.JSONDecodeError, OSError):
                continue

    return _strip_hostname(socket.gethostname())


def backfill_streams(journal_root: Path, fallback_host: str, verbose: bool) -> None:
    """Classify all segments, build linkage chains, and write markers."""
    print(f"Fallback host: {fallback_host}")
    print(f"Journal: {journal_root}\n")

    # Phase 1: Build import reverse index
    import_index = build_import_reverse_index(journal_root)
    if import_index:
        print(f"Import index: {len(import_index)} segments from imports/\n")

    # Phase 2: Walk all segments and classify
    classified: list[dict] = []  # {day, segment, seg_dir, stream, signal}
    signal_counts: dict[str, int] = {}
    stream_counts: dict[str, int] = {}

    days = day_dirs()
    for day in sorted(days):
        day_dir = Path(days[day])
        for seg_dir in sorted(day_dir.iterdir()):
            if not seg_dir.is_dir() or not segment_key(seg_dir.name):
                continue

            # Skip empty segments (no content files at all)
            has_content = any(
                f.is_file() and f.name != "stream.json" for f in seg_dir.iterdir()
            )
            if not has_content:
                continue

            name, signal = classify_segment(seg_dir, day, import_index, fallback_host)

            classified.append(
                {
                    "day": day,
                    "segment": seg_dir.name,
                    "seg_dir": seg_dir,
                    "stream": name,
                    "signal": signal,
                }
            )
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
            stream_counts[name] = stream_counts.get(name, 0) + 1

            if verbose:
                tag = "*" if signal != SIGNAL_EXISTING else " "
                print(f"  {tag} {day}/{seg_dir.name} -> {name} ({signal})")

    print(f"Classified {len(classified)} segments into {len(stream_counts)} streams\n")

    # Signal breakdown
    print("Signal breakdown:")
    for signal, count in sorted(signal_counts.items(), key=lambda x: -x[1]):
        print(f"  {signal:<30} {count:>5}")
    print()

    # Stream breakdown
    print("Stream breakdown:")
    for name, count in sorted(stream_counts.items(), key=lambda x: -x[1]):
        print(f"  {name:<30} {count:>5}")
    print()

    # Phase 3: Build linkage chains
    streams: dict[str, list[dict]] = {}
    for entry in classified:
        streams.setdefault(entry["stream"], []).append(entry)

    for name in streams:
        streams[name].sort(key=lambda e: (e["day"], e["segment"]))

    # Assign seq and prev pointers
    writes_needed = 0
    skipped_existing = 0
    linkage_fixed = 0

    for name, entries in streams.items():
        for i, entry in enumerate(entries):
            entry["seq"] = i + 1
            if i == 0:
                entry["prev_day"] = None
                entry["prev_segment"] = None
            else:
                entry["prev_day"] = entries[i - 1]["day"]
                entry["prev_segment"] = entries[i - 1]["segment"]

            # Check if write is needed
            existing = read_segment_stream(entry["seg_dir"])
            if existing:
                same_stream = existing.get("stream") == name
                same_seq = existing.get("seq") == entry["seq"]
                same_prev = (
                    existing.get("prev_day") == entry["prev_day"]
                    and existing.get("prev_segment") == entry["prev_segment"]
                )
                if same_stream and same_seq and same_prev:
                    skipped_existing += 1
                    entry["action"] = "skip"
                    continue
                elif same_stream:
                    linkage_fixed += 1
                    entry["action"] = "fix_linkage"
                else:
                    entry["action"] = "write"
                    writes_needed += 1
            else:
                entry["action"] = "write"
                writes_needed += 1

    print(
        f"Actions: {writes_needed} writes, {linkage_fixed} linkage fixes, "
        f"{skipped_existing} already correct\n"
    )

    if writes_needed == 0 and linkage_fixed == 0:
        print("Nothing to do — all segments already have correct stream markers.")
        return

    # Phase 4: Write markers
    written = 0
    fixed = 0
    for name, entries in streams.items():
        for entry in entries:
            if entry["action"] == "skip":
                continue
            write_segment_stream(
                entry["seg_dir"],
                entry["stream"],
                entry["prev_day"],
                entry["prev_segment"],
                entry["seq"],
            )
            if entry["action"] == "write":
                written += 1
            else:
                fixed += 1

    print(f"Wrote {written} new markers, fixed {fixed} linkages")

    # Phase 5: Rebuild stream state files
    streams_dir = journal_root / "streams"
    streams_dir.mkdir(parents=True, exist_ok=True)

    rebuilt = 0
    for name, entries in streams.items():
        if not entries:
            continue
        last = entries[-1]

        # Try to preserve type/host/platform from existing state
        existing_state_path = streams_dir / f"{name}.json"
        existing_state = {}
        if existing_state_path.exists():
            try:
                with open(existing_state_path, "r", encoding="utf-8") as f:
                    existing_state = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        # Infer type from stream name
        if name.startswith("import."):
            stream_type = "import"
        elif "." in name and name.endswith(".tmux"):
            stream_type = "observer"
        else:
            stream_type = existing_state.get("type", "observer")

        state = {
            "name": name,
            "type": stream_type,
            "host": existing_state.get(
                "host", fallback_host if stream_type == "observer" else None
            ),
            "platform": existing_state.get("platform"),
            "created_at": existing_state.get("created_at", 0),
            "last_day": last["day"],
            "last_segment": last["segment"],
            "seq": last["seq"],
        }

        state_path = streams_dir / f"{name}.json"
        tmp_path = state_path.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
        os.rename(str(tmp_path), str(state_path))
        rebuilt += 1

    print(f"Rebuilt {rebuilt} stream state files in {streams_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--host",
        help="Override fallback hostname (default: from existing streams or socket.gethostname())",
    )

    args = setup_cli(parser)
    journal_root = Path(get_journal())

    fallback_host = _infer_fallback_host(journal_root, args.host)
    backfill_streams(journal_root, fallback_host, args.verbose)


if __name__ == "__main__":
    main()
