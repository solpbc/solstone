# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import argparse
import datetime as dt
import json
import logging
import os
import queue
import re
import threading
import time
from pathlib import Path

from think.callosum import CallosumConnection
from think.detect_created import detect_created
from think.importers.audio import _get_audio_duration, prepare_audio_segments
from think.importers.shared import (
    _get_relative_path,
    _is_in_imports,
    _setup_import,
)
from think.importers.text import _read_transcript, process_transcript
from think.importers.utils import save_import_segments
from think.streams import stream_name, update_stream, write_segment_stream
from think.utils import day_path, get_journal, get_rev, segment_key, setup_cli

logger = logging.getLogger(__name__)

TIME_RE = re.compile(r"\d{8}_\d{6}")

# Importer tract state
_callosum: CallosumConnection | None = None
_message_queue: queue.Queue | None = None
_import_id: str | None = None
_current_stage: str = "initialization"
_start_time: float = 0.0
_stage_start_time: float = 0.0
_stages_run: list[str] = []
_status_thread: threading.Thread | None = None
_status_running: bool = False


def _set_stage(stage: str) -> None:
    """Update current stage and track timing."""
    global _current_stage, _stage_start_time
    _current_stage = stage
    _stage_start_time = time.monotonic()
    if stage not in _stages_run:
        _stages_run.append(stage)
    logger.debug(f"Stage changed to: {stage}")


def _status_emitter() -> None:
    """Background thread that emits status events every 5 seconds."""
    while _status_running:
        if _callosum and _import_id:
            elapsed_ms = int((time.monotonic() - _start_time) * 1000)
            stage_elapsed_ms = int((time.monotonic() - _stage_start_time) * 1000)
            _callosum.emit(
                "importer",
                "status",
                import_id=_import_id,
                stage=_current_stage,
                elapsed_ms=elapsed_ms,
                stage_elapsed_ms=stage_elapsed_ms,
            )
        time.sleep(5)


def _format_timestamp_display(timestamp: str) -> str:
    """Format timestamp for human-readable display."""
    try:
        dt_obj = dt.datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        return dt_obj.strftime("%a %b %d %Y, %-I:%M %p")
    except ValueError:
        return timestamp


def _run_sync(backend_name: str, *, dry_run: bool = True) -> None:
    """Run sync for a named backend and print results."""
    from think.importers.plaud import format_size
    from think.importers.sync import get_syncable_backends, load_sync_state

    journal_root = Path(get_journal())

    # Find the requested backend
    backends = get_syncable_backends()
    backend = None
    for b in backends:
        if b.name == backend_name:
            backend = b
            break

    if backend is None:
        available = ", ".join(b.name for b in backends) or "(none)"
        raise SystemExit(
            f"Unknown sync backend: {backend_name}\nAvailable backends: {available}"
        )

    mode = "save" if not dry_run else "catalog"
    print(f"Syncing {backend_name} ({mode} mode)...")
    print()

    try:
        result = backend.sync(journal_root, dry_run=dry_run)
    except ValueError as e:
        raise SystemExit(str(e))
    except RuntimeError as e:
        raise SystemExit(f"Sync failed: {e}")

    total = result.get("total", 0)
    imported = result.get("imported", 0)
    available = result.get("available", 0)
    skipped = result.get("skipped", 0)
    downloaded = result.get("downloaded", 0)
    errors = result.get("errors", [])

    # Print summary
    print()
    print(f"  Total recordings:    {total}")
    print(f"  Already imported:    {imported}")
    print(f"  Available to import: {available}")
    if skipped:
        print(f"  Skipped:             {skipped} (trashed/short)")

    if downloaded > 0:
        print(f"  Downloaded + imported: {downloaded}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for err in errors:
            print(f"    - {err}")

    # In dry-run mode, show available files
    if dry_run and available > 0:
        state = load_sync_state(journal_root, backend_name)
        if state:
            files = state.get("files", {})
            avail_files = [
                (fid, info)
                for fid, info in files.items()
                if info.get("status") == "available"
            ]
            if avail_files:
                print()
                print("Available recordings:")
                for _fid, info in avail_files:
                    name = info.get("filename", "unnamed")
                    size = info.get("filesize", 0)
                    print(f"  - {name} ({format_size(size)})")
                print()
                print("Run with --save to download and import these files:")
                print(f"  sol import --sync {backend_name} --save")

    if not dry_run and available == 0 and downloaded == 0:
        print()
        print("Everything is up to date.")


def main() -> None:
    global _callosum, _message_queue, _import_id, _current_stage, _start_time
    global _stage_start_time, _stages_run, _status_thread, _status_running

    parser = argparse.ArgumentParser(description="Import a media file into the journal")
    parser.add_argument("media", nargs="?", help="Path to audio or text file")
    parser.add_argument(
        "--timestamp", help="Timestamp YYYYMMDD_HHMMSS for journal entry"
    )
    parser.add_argument(
        "--facet",
        type=str,
        default=None,
        help="Facet name for this import",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default=None,
        help="Contextual setting description to store with import metadata",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Import source type (apple, plaud, audio, text, or a file importer name). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-import by deleting existing import directory",
    )
    parser.add_argument(
        "--auto",
        nargs="?",
        const=True,
        default=None,
        help="Auto-accept detected timestamp. Optionally provide guidance text for the LLM (e.g., --auto 'timestamps are Pacific time').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without writing to the journal",
    )
    parser.add_argument(
        "--backends",
        action="store_true",
        help="List syncable importer backends",
    )
    parser.add_argument(
        "--sync",
        type=str,
        metavar="BACKEND",
        help="Sync catalog from a backend (e.g., plaud). Shows status by default.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="With --sync: download and import new files (default is dry-run)",
    )
    parser.add_argument(
        "--list-importers",
        action="store_true",
        help="List available file importers",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (file importers only)",
    )
    args, extra = setup_cli(parser, parse_known=True)
    if extra and not args.timestamp:
        args.timestamp = extra[0]

    if args.backends:
        from think.importers.sync import get_syncable_backends

        backends = get_syncable_backends()
        if backends:
            print("Syncable backends:")
            for b in backends:
                print(f"  {b.name}")
        else:
            print("No syncable backends available")
        return

    if args.list_importers:
        from think.importers.file_importer import get_file_importers

        importers = get_file_importers()
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "name": imp.name,
                            "display_name": imp.display_name,
                            "file_patterns": imp.file_patterns,
                            "description": imp.description,
                        }
                        for imp in importers
                    ]
                )
            )
        elif importers:
            print("File importers:")
            for imp in importers:
                patterns = ", ".join(imp.file_patterns)
                print(f"  {imp.name:<12} {imp.display_name} ({patterns})")
                print(f"               {imp.description}")
        else:
            print("No file importers available")
        return

    if args.sync:
        _run_sync(args.sync, dry_run=not args.save)
        return

    if not args.media:
        parser.error("the following arguments are required: media")

    args.media = os.path.expanduser(args.media)

    _file_importer = None
    import_source = None

    # Track detection result for metadata
    detection_result = None

    # --- File importer detection (before timestamp resolution) ---
    if args.source:
        from think.importers.file_importer import (
            FILE_IMPORTER_REGISTRY,
            get_file_importer,
        )

        if args.source in FILE_IMPORTER_REGISTRY:
            _file_importer = get_file_importer(args.source)
            if _file_importer is None:
                raise SystemExit(f"Failed to load file importer: {args.source}")
            import_source = args.source

    # Also try file importer detection for directories
    if _file_importer is None and os.path.isdir(args.media):
        from think.importers.file_importer import detect_file_importer

        detected = detect_file_importer(Path(args.media))
        if detected is not None:
            _file_importer = detected
            import_source = detected.name

    # Try file importer detection for unknown file extensions
    if _file_importer is None and not args.source:
        _ext = os.path.splitext(args.media)[1].lower()
        if _ext not in {".m4a", ".txt", ".md", ".pdf"}:
            from think.importers.file_importer import detect_file_importer

            detected = detect_file_importer(Path(args.media))
            if detected is not None:
                _file_importer = detected
                import_source = detected.name

    # --- Timestamp resolution ---
    if _file_importer is not None and not args.timestamp:
        # File importers don't need an external timestamp — auto-generate for metadata
        args.timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    elif not args.timestamp:
        # If no timestamp provided, detect it
        # Pass the original filename for better detection
        detection_result = detect_created(
            args.media,
            original_filename=os.path.basename(args.media),
            guidance=args.auto if isinstance(args.auto, str) else None,
        )
        if (
            detection_result
            and detection_result.get("day")
            and detection_result.get("time")
        ):
            detected_timestamp = f"{detection_result['day']}_{detection_result['time']}"
            display = _format_timestamp_display(detected_timestamp)
            if args.auto:
                print(
                    f"Detected timestamp: {detected_timestamp} ({display}) — auto-importing"
                )
                args.timestamp = detected_timestamp
            else:
                print(f"Detected timestamp: {detected_timestamp} ({display})")
                print("\nRun:")
                print(f"  sol import {args.media} --timestamp {detected_timestamp}")
                return
        else:
            raise SystemExit(
                "Could not detect timestamp. Please provide --timestamp YYYYMMDD_HHMMSS"
            )

    if not TIME_RE.fullmatch(args.timestamp):
        raise SystemExit("timestamp must be in YYYYMMDD_HHMMSS format")

    base_dt = dt.datetime.strptime(args.timestamp, "%Y%m%d_%H%M%S")
    day = base_dt.strftime("%Y%m%d")

    # --- Derive import_source for non-file-importer paths ---
    if import_source is None:
        if args.source:
            import_source = args.source
        else:
            _ext = os.path.splitext(args.media)[1].lower()
            if _ext == ".m4a":
                import_source = "apple"
            elif _ext in {".txt", ".md", ".pdf"}:
                import_source = "text"
            else:
                import_source = "audio"

    stream = stream_name(import_source=import_source)

    if args.dry_run and _file_importer is not None:
        preview = _file_importer.preview(Path(args.media))
        if args.json:
            print(
                json.dumps(
                    {
                        "importer": _file_importer.name,
                        "source": args.media,
                        "date_range": list(preview.date_range),
                        "item_count": preview.item_count,
                        "entity_count": preview.entity_count,
                        "summary": preview.summary,
                    }
                )
            )
        else:
            print()
            print(f"  Importer:   {_file_importer.display_name}")
            print(f"  Source:     {args.media}")
            print(f"  Date range: {preview.date_range[0]} — {preview.date_range[1]}")
            print(f"  Items:      {preview.item_count}")
            print(f"  Entities:   {preview.entity_count}")
            print(f"  Summary:    {preview.summary}")
            print()
        return

    if args.dry_run:
        from think.importers.plaud import format_size

        # Print summary without writing anything
        file_size = os.path.getsize(args.media)
        display = _format_timestamp_display(args.timestamp)

        print()
        print(f"  File:       {args.media}")
        print(f"  Size:       {format_size(file_size)}")
        print(f"  Timestamp:  {args.timestamp} ({display})")
        print(f"  Source:     {import_source}")
        print(f"  Stream:     {stream}")
        print(f"  Target day: {day}")

        ext = os.path.splitext(args.media)[1].lower()
        if ext in {".txt", ".md", ".pdf"}:
            from think.detect_transcript import detect_transcript_segment
            from think.importers.text import _time_to_seconds

            text = _read_transcript(args.media)
            chars = len(text)
            lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
            print()
            print(f"  Content:    {chars:,} characters, {lines:,} lines")

            # Run segmentation to preview what would be created
            start_time = base_dt.strftime("%H:%M:%S")
            if args.verbose:
                print()
                print("  Segmenting transcript...")
            segments = detect_transcript_segment(text, start_time)
            if segments:
                keys = []
                for idx, (start_at, seg_text) in enumerate(segments):
                    time_part = start_at.replace(":", "")
                    start_seconds = _time_to_seconds(start_at)
                    if idx + 1 < len(segments):
                        next_start_at, _ = segments[idx + 1]
                        duration = _time_to_seconds(next_start_at) - start_seconds
                    else:
                        duration = 5
                    duration = max(1, duration)
                    key = f"{time_part}_{duration}"
                    keys.append(key)

                print(f"  Segments:   {len(segments)}")
                print(f"  Keys:       {', '.join(keys)}")

                if args.verbose:
                    print()
                    for idx, (key, (start_at, seg_text)) in enumerate(
                        zip(keys, segments), 1
                    ):
                        seg_lines = seg_text.count("\n") + (
                            1 if seg_text and not seg_text.endswith("\n") else 0
                        )
                        print(
                            f"  Segment {idx}: {key} "
                            f"({seg_lines:,} lines, {len(seg_text):,} chars)"
                        )
            else:
                print()
                print("  Segments:   segmentation failed")
        else:
            duration = _get_audio_duration(args.media)
            if duration is not None:
                segment_duration = 300
                num_segments = int(
                    (duration + segment_duration - 1) // segment_duration
                )
                if num_segments == 0:
                    num_segments = 1

                keys = []
                for i in range(num_segments):
                    ts = base_dt + dt.timedelta(minutes=i * 5)
                    keys.append(f"{ts.strftime('%H%M%S')}_{segment_duration}")

                if duration < 60:
                    dur_str = f"{duration:.0f} seconds"
                elif duration < 3600:
                    dur_str = f"{duration / 60:.1f} minutes"
                else:
                    dur_str = f"{duration / 3600:.1f} hours"

                print()
                print(f"  Duration:   {dur_str}")
                print(f"  Segments:   {num_segments} (5-minute chunks)")
                print(f"  Keys:       {', '.join(keys)}")
            else:
                print()
                print("  Duration:   unknown (ffprobe failed)")
        print()
        return

    # Check if file needs setup (not already in imports/)
    needs_setup = _file_importer is None and not _is_in_imports(args.media)

    # Copy to imports/ if file is not already there
    if needs_setup:
        args.media = _setup_import(
            args.media,
            args.timestamp,
            args.facet,
            args.setting,
            detection_result,
            force=args.force,
        )
        print("Starting import...")

    logger.info(f"Using provided timestamp: {args.timestamp}")
    day_dir = str(day_path(day))

    # Initialize importer tract state
    _import_id = args.timestamp
    _start_time = time.monotonic()
    _stage_start_time = _start_time
    _current_stage = "initialization"
    _stages_run = ["initialization"]

    # Start Callosum connection with message queue for receiving events
    _message_queue = queue.Queue()
    _callosum = CallosumConnection(defaults={"rev": get_rev()})
    _callosum.start(callback=lambda msg: _message_queue.put(msg))

    # Start status emitter thread
    _status_running = True
    _status_thread = threading.Thread(target=_status_emitter, daemon=True)
    _status_thread.start()

    # Emit started event
    ext = os.path.splitext(args.media)[1].lower()
    _callosum.emit(
        "importer",
        "started",
        import_id=_import_id,
        input_file=os.path.basename(args.media),
        file_type=ext.lstrip("."),
        day=day,
        facet=args.facet,
        setting=args.setting,
        options={},
        stage=_current_stage,
        stream=stream,
    )

    # Track all created files and processing metadata
    all_created_files: list[str] = []
    created_segments: list[str] = []
    journal_root = Path(get_journal())
    processing_results = {
        "processed_timestamp": args.timestamp,
        "target_day": base_dt.strftime("%Y%m%d"),
        "target_day_path": day_dir,
        "input_file": args.media,
        "processing_started": dt.datetime.now().isoformat(),
        "facet": args.facet,
        "setting": args.setting,
        "outputs": [],
    }

    # Get parent directory for saving metadata
    media_path = Path(args.media)
    import_dir = media_path.parent
    failed_segments: list[str] = []

    try:
        if _file_importer is not None:
            # File importer processing — structured file/directory imports
            _set_stage("importing")

            result = _file_importer.process(
                Path(args.media), journal_root, facet=args.facet
            )

            all_created_files.extend(result.files_created)
            processing_results["outputs"].append(
                {
                    "type": "file_import",
                    "importer": _file_importer.name,
                    "description": result.summary,
                    "files": result.files_created,
                    "entries_written": result.entries_written,
                    "entities_seeded": result.entities_seeded,
                    "count": len(result.files_created),
                }
            )

            if result.errors:
                logger.warning(
                    "%d errors during %s import: %s",
                    len(result.errors),
                    _file_importer.name,
                    result.errors,
                )

            # Emit callosum events for file imports
            _callosum.emit(
                "importer",
                "file_imported",
                import_id=_import_id,
                importer=_file_importer.name,
                entries_written=result.entries_written,
                entities_seeded=result.entities_seeded,
                files_created=len(result.files_created),
                errors=len(result.errors),
                stream=stream,
            )

            logger.info(
                "%s import complete: %d entries, %d entities, %d files",
                _file_importer.display_name,
                result.entries_written,
                result.entities_seeded,
                len(result.files_created),
            )
            if args.json:
                print(
                    json.dumps(
                        {
                            "importer": _file_importer.name,
                            "entries_written": result.entries_written,
                            "entities_seeded": result.entities_seeded,
                            "files_created": result.files_created,
                            "errors": result.errors,
                            "summary": result.summary,
                        }
                    )
                )

        elif ext in {".txt", ".md", ".pdf"}:
            # Text transcript processing — no observe pipeline
            _set_stage("segmenting")

            created_files = process_transcript(
                args.media,
                day_dir,
                base_dt,
                import_id=args.timestamp,
                stream=stream,
                facet=args.facet,
                setting=args.setting,
            )
            all_created_files.extend(created_files)
            processing_results["outputs"].append(
                {
                    "type": "transcript",
                    "format": "imported_audio.jsonl",
                    "description": "Transcript segments",
                    "files": created_files,
                    "count": len(created_files),
                }
            )

            # Extract segment keys for text imports
            for file_path in created_files:
                seg = segment_key(file_path)
                if seg and seg not in created_segments:
                    created_segments.append(seg)

            # Write stream markers for text import segments
            for seg in created_segments:
                try:
                    seg_dir = day_path(day) / stream / seg
                    result = update_stream(stream, day, seg, type="import", host=None)
                    write_segment_stream(
                        seg_dir,
                        stream,
                        result["prev_day"],
                        result["prev_segment"],
                        result["seq"],
                    )
                except Exception as e:
                    logger.warning(f"Failed to write stream identity: {e}")

            # Save segment list for tracking (same as audio path)
            save_import_segments(journal_root, args.timestamp, created_segments, day)

            # Emit observe.observed for text imports (already processed)
            for seg in created_segments:
                _callosum.emit(
                    "observe", "observed", segment=seg, day=day, stream=stream
                )
                logger.info(f"Emitted observe.observed for segment: {day}/{seg}")

        else:
            # Audio processing via observe pipeline
            _set_stage("segmenting")

            # Prepare audio segments (slice into 5-minute chunks)
            segments = prepare_audio_segments(
                args.media,
                day_dir,
                base_dt,
                args.timestamp,
                stream,
            )

            if not segments:
                raise RuntimeError("No segments created from audio file")

            # Track created files and segment keys, write stream markers
            for seg_key, seg_dir, files in segments:
                created_segments.append(seg_key)
                for f in files:
                    all_created_files.append(str(seg_dir / f))
                try:
                    result = update_stream(
                        stream, day, seg_key, type="import", host=None
                    )
                    write_segment_stream(
                        seg_dir,
                        stream,
                        result["prev_day"],
                        result["prev_segment"],
                        result["seq"],
                    )
                except Exception as e:
                    logger.warning(f"Failed to write stream identity: {e}")

            # Save segment list for tracking
            save_import_segments(journal_root, args.timestamp, created_segments, day)

            processing_results["outputs"].append(
                {
                    "type": "audio_segments",
                    "description": "Audio segments queued for transcription",
                    "segments": created_segments,
                    "count": len(created_segments),
                }
            )

            # Build meta dict for observe.observing events
            meta: dict[str, str] = {"import_id": args.timestamp, "stream": stream}
            if args.facet:
                meta["facet"] = args.facet
            if args.setting:
                meta["setting"] = args.setting

            # Emit observe.observing per segment to trigger sense.py transcription
            for seg_key, seg_dir, files in segments:
                _callosum.emit(
                    "observe",
                    "observing",
                    segment=seg_key,
                    day=day,
                    files=files,
                    meta=meta,
                    stream=stream,
                )
                logger.info(f"Emitted observe.observing for segment: {day}/{seg_key}")

            # Wait for transcription to complete
            _set_stage("transcribing")
            pending = set(created_segments)
            completed_count = 0
            segment_timeout = 600  # 10 minutes since last progress
            last_progress = time.monotonic()
            transcribe_start = time.monotonic()

            logger.info(f"Waiting for {len(pending)} segments to complete")

            while pending:
                # Check for timeout since last progress
                stall_duration = time.monotonic() - last_progress
                if stall_duration > segment_timeout:
                    total_elapsed = int(time.monotonic() - transcribe_start)
                    timed_out = sorted(pending)
                    logger.error(
                        f"Transcription stalled: no progress for "
                        f"{int(stall_duration)}s ({total_elapsed}s total). "
                        f"{completed_count}/{len(created_segments)} segments "
                        f"completed, {len(timed_out)} still pending: {timed_out}"
                    )
                    failed_segments.extend(timed_out)
                    break

                # Poll for observe.observed events from message queue
                try:
                    msg = _message_queue.get(timeout=5.0)
                except queue.Empty:
                    continue

                tract = msg.get("tract")
                event = msg.get("event")
                seg = msg.get("segment")

                if tract == "observe" and event == "observed" and seg in pending:
                    pending.discard(seg)
                    completed_count += 1
                    last_progress = time.monotonic()
                    if msg.get("error"):
                        errors = msg.get("errors", [])
                        logger.warning(
                            f"Segment {seg} failed: {errors} ({len(pending)} remaining)"
                        )
                        failed_segments.append(seg)
                    else:
                        logger.info(
                            f"Segment {seg} transcribed "
                            f"({completed_count}/{len(created_segments)} done, "
                            f"{len(pending)} remaining)"
                        )
                elif tract == "observe" and event == "status":
                    last_progress = time.monotonic()

            if failed_segments:
                logger.warning(
                    f"{len(failed_segments)} of {len(created_segments)} "
                    f"segments failed: {failed_segments}"
                )
            else:
                total_elapsed = int(time.monotonic() - transcribe_start)
                logger.info(
                    f"All {len(created_segments)} segments "
                    f"transcribed successfully ({total_elapsed}s)"
                )

        # Complete processing metadata
        processing_results["processing_completed"] = dt.datetime.now().isoformat()
        processing_results["total_files_created"] = len(all_created_files)
        processing_results["all_created_files"] = all_created_files
        processing_results["segments"] = created_segments
        if failed_segments:
            processing_results["failed_segments"] = failed_segments

        imported_path = import_dir / "imported.json"
        if _file_importer is None:
            # Write imported.json with all processing metadata
            try:
                with open(imported_path, "w", encoding="utf-8") as f:
                    json.dump(processing_results, f, indent=2)
                logger.info(f"Saved import processing metadata: {imported_path}")
            except Exception as e:
                logger.warning(f"Failed to save imported.json: {e}")

            # Update import.json with processing summary if it exists
            import_metadata_path = import_dir / "import.json"
            if import_metadata_path.exists():
                try:
                    with open(import_metadata_path, "r", encoding="utf-8") as f:
                        import_meta = json.load(f)
                    import_meta["processing_completed"] = processing_results[
                        "processing_completed"
                    ]
                    import_meta["total_files_created"] = processing_results[
                        "total_files_created"
                    ]
                    import_meta["imported_json_path"] = str(imported_path)
                    import_meta["segments"] = created_segments
                    with open(import_metadata_path, "w", encoding="utf-8") as f:
                        json.dump(import_meta, f, indent=2)
                    logger.info(f"Updated import metadata: {import_metadata_path}")
                except Exception as e:
                    logger.warning(f"Failed to update import metadata: {e}")

        # Emit completed event
        duration_ms = int((time.monotonic() - _start_time) * 1000)
        output_files_relative = [_get_relative_path(f) for f in all_created_files]
        metadata_file_relative = _get_relative_path(str(imported_path))

        _callosum.emit(
            "importer",
            "completed",
            import_id=_import_id,
            stage=_current_stage,
            duration_ms=duration_ms,
            total_files_created=len(all_created_files),
            output_files=output_files_relative,
            metadata_file=metadata_file_relative,
            stages_run=_stages_run,
            segments=created_segments,
            stream=stream,
        )

    except Exception as e:
        duration_ms = int((time.monotonic() - _start_time) * 1000)
        partial_outputs = [_get_relative_path(f) for f in all_created_files]
        imported_path = import_dir / "imported.json"

        error_results = {
            **processing_results,  # Include all the metadata we have
            "processing_failed": dt.datetime.now().isoformat(),
            "error": str(e),
            "error_stage": _current_stage,
            "duration_ms": duration_ms,
            "total_files_created": len(all_created_files),
            "all_created_files": all_created_files,
            "stages_run": _stages_run,
        }

        if _file_importer is None:
            # Write error state to imported.json for persistent failure tracking
            try:
                with open(imported_path, "w", encoding="utf-8") as f:
                    json.dump(error_results, f, indent=2)
                logger.info(f"Saved error state: {imported_path}")
            except Exception as write_err:
                logger.warning(f"Failed to write error state: {write_err}")

        # Emit error event
        if _callosum:
            _callosum.emit(
                "importer",
                "error",
                import_id=_import_id,
                stage=_current_stage,
                error=str(e),
                duration_ms=duration_ms,
                partial_outputs=partial_outputs,
            )

        logger.error(f"Import failed: {e}")
        raise

    finally:
        # Stop status thread and Callosum connection
        _status_running = False
        if _status_thread:
            _status_thread.join(timeout=6)
        if _callosum:
            _callosum.stop()


if __name__ == "__main__":
    main()
