# macOS Observer Implementation TODO

This document tracks the remaining work to complete the macOS observer integration using sck-cli and ScreenCaptureKit.

## Phase 1: Activity Detection (activity.py)

### 1.1 Implement `get_idle_time_ms()`
- [ ] Import PyObjC Quartz framework
- [ ] Use `CGEventSourceSecondsSinceLastEventType(1, kCGAnyInputEventType)`
- [ ] Convert seconds to milliseconds
- [ ] Add error handling for API failures
- [ ] Test on macOS system

**Example:**
```python
from Quartz import CGEventSourceSecondsSinceLastEventType, kCGAnyInputEventType

def get_idle_time_ms() -> int:
    seconds = CGEventSourceSecondsSinceLastEventType(1, kCGAnyInputEventType)
    return int(seconds * 1000)
```

### 1.2 Implement `is_screen_locked()`
- [ ] Research best approach for screen lock detection:
  - Option A: Query CGSessionCopyCurrentDictionary for kCGSSessionOnConsoleKey
  - Option B: Use `ioreg -c IOHIDSystem | grep HIDIdleTime`
  - Option C: Check display sleep state as proxy
- [ ] Implement chosen method with PyObjC
- [ ] Add fallback if primary method unavailable
- [ ] Test with actual screen lock/unlock cycles
- [ ] Handle edge cases (fast user switching, etc.)

### 1.3 Implement `is_power_save_active()`
- [ ] Investigate IOKit display state query
- [ ] Check NSScreen APIs for display power state
- [ ] Alternative: subprocess call to `pmset -g` or `system_profiler`
- [ ] Return True if displays are sleeping/powered off
- [ ] Test with display sleep/wake

### 1.4 Implement `get_monitor_geometries()`
- [ ] Import Cocoa NSScreen framework
- [ ] Get all screens: `NSScreen.screens()`
- [ ] For each screen:
  - Extract frame geometry: `screen.frame()`
  - Get device description for unique ID: `screen.deviceDescription()`
  - Handle NSScreenNumber, display ID, etc.
- [ ] Compute union bounding box of all monitors
- [ ] Calculate midlines (union_mid_x, union_mid_y)
- [ ] Assign position labels based on intersection with midlines:
  - "center", "left", "right", "top", "bottom", "left-top", etc.
- [ ] Return list of dicts: `[{"id": "...", "box": [x1,y1,x2,y2], "position": "..."}]`
- [ ] Test with single monitor, dual monitor, triple monitor setups
- [ ] Handle coordinate system (Cocoa uses bottom-left origin)

### 1.5 Implement `get_monitor_metadata_string()`
- [ ] Call `get_monitor_geometries()`
- [ ] Format as: `"0:center,0,0,1920,1080 1:right,1920,0,3840,1080"`
- [ ] Test output format matches GNOME format exactly

## Phase 2: ScreenCaptureKit Manager (screencapture.py)

### 2.1 Implement `start()`
- [ ] Validate sck-cli is available in PATH or at specified path
- [ ] Build command: `[sck_cli_path, str(output_base), "-r", str(frame_rate), "-l", str(duration)]`
- [ ] Launch subprocess with `subprocess.Popen()`
- [ ] Store process handle and output_base in instance variables
- [ ] Add stderr/stdout capture for debugging
- [ ] Return success/failure
- [ ] Test with various parameters

### 2.2 Implement `stop()`
- [ ] Check if process exists and is running
- [ ] Send SIGTERM to process
- [ ] Wait with timeout (5 seconds) for graceful shutdown
- [ ] If timeout, send SIGKILL as fallback
- [ ] Clear process handle and output_base
- [ ] Log any stderr output from process
- [ ] Test graceful and forced shutdown scenarios

### 2.3 Implement `is_running()`
- [ ] Check if `self.process` is not None
- [ ] Use `self.process.poll()` to check if still running
- [ ] Return True if running, False otherwise

### 2.4 Implement `finalize()`
- [ ] Check if temp files exist: `temp_base.mov`, `temp_base.m4a`
- [ ] If files missing, log error and return failure
- [ ] Add monitor metadata to video file using one of:
  - Option A: ffmpeg: `ffmpeg -i input.mov -metadata title="..." -c copy output.mov`
  - Option B: PyObjC AVFoundation APIs to modify metadata in-place
- [ ] Atomically rename temp video file to final path: `os.replace()`
- [ ] Atomically rename temp audio file to final path: `os.replace()`
- [ ] Return tuple of (video_success, audio_success)
- [ ] Handle errors gracefully (log but don't crash)
- [ ] Test with actual sck-cli output files

### 2.5 Implement `get_output_size()`
- [ ] Check if `self.current_output_base` is set
- [ ] Build path to .mov file
- [ ] Use `os.path.getsize()` to get file size
- [ ] Return 0 if file doesn't exist or error
- [ ] Used for health check file growth verification

## Phase 3: Main Observer (observer.py)

### 3.1 Implement `setup()`
- [ ] Verify sck-cli is available in PATH
- [ ] Create ScreenCaptureKitManager instance
- [ ] Initialize Callosum connection
- [ ] Start Callosum connection
- [ ] Log initialization success
- [ ] Return True on success, False on failure

### 3.2 Implement `check_activity_status()`
- [ ] Call `get_idle_time_ms()` from activity module
- [ ] Call `is_screen_locked()` from activity module
- [ ] Cache values in instance variables for status events
- [ ] Determine if idle: `(idle_time > IDLE_THRESHOLD_MS) or screen_locked`
- [ ] Set `self.cached_is_active = not is_idle`
- [ ] Return activity status

### 3.3 Implement `handle_boundary()`
- [ ] Get timestamp parts and calculate duration
- [ ] Get day directory path
- [ ] If capture running:
  - Stop sck-cli via `self.screencapture.stop()`
  - Build temp base path (e.g., `.120000`)
  - Build final paths with duration (e.g., `120000_300_screen.mov`, `120000_300_audio.m4a`)
  - Queue for finalization: `self.pending_finalization = (temp_base, final_video, final_audio)`
  - Clear state variables
- [ ] Reset timing: `self.start_at = time.time()`, `self.start_at_mono = time.monotonic()`
- [ ] If active and screen not locked:
  - Call `initialize_capture()`
- [ ] Build list of files that were captured
- [ ] Emit Callosum event: `self.callosum.emit("observe", "observing", segment="...", files=[...])`
- [ ] Log boundary handling

### 3.4 Implement `initialize_capture()`
- [ ] Get timestamp for filename
- [ ] Get day directory path
- [ ] Build temp output base: `day_dir / f".{time_part}"` (hidden file)
- [ ] Call `self.screencapture.start(output_base, self.interval, frame_rate=1.0)`
- [ ] If success:
  - Set `self.capture_running = True`
  - Set `self.current_output_base = output_base`
  - Set `self.last_video_size = 0`
  - Log capture start
  - Return True
- [ ] Else:
  - Log failure
  - Return False

### 3.5 Implement `emit_status()`
- [ ] Build capture info dict:
  - If capturing: `{"recording": True, "file": "...", "window_elapsed_seconds": ...}`
  - Else: `{"recording": False}`
- [ ] Build activity info dict: `{"active": ..., "idle_time_ms": ..., "screen_locked": ...}`
- [ ] Emit via Callosum: `self.callosum.emit("observe", "status", capture=..., activity=...)`

### 3.6 Implement `finalize_capture()`
- [ ] Check if temp files exist
- [ ] If missing, log warning and return
- [ ] Get monitor metadata string: `get_monitor_metadata_string()`
- [ ] Call `self.screencapture.finalize(temp_base, final_video, final_audio, monitor_metadata)`
- [ ] Log success/failure
- [ ] Return finalization status

### 3.7 Implement `main_loop()`
- [ ] Check initial activity status
- [ ] If active and not locked, start initial capture
- [ ] Main loop while `self.running`:
  - Sleep for CHUNK_DURATION (5 seconds)
  - Process pending finalization if queued
  - Check activity status
  - Detect activation edge: `is_active and not self.capture_running`
  - Calculate elapsed time since window start (monotonic)
  - Check for boundary: `elapsed >= self.interval or activation_edge`
  - If boundary, call `handle_boundary(is_active)`
  - Track if capture files are growing (for health reporting via status event)
  - Emit status event with `screencast.files_growing` field (supervisor derives health from this)
- [ ] Call `shutdown()` after loop exits

### 3.8 Implement `shutdown()`
- [ ] If capture running:
  - Stop capture
  - Wait briefly (1 second) for files to be written
  - Build final paths
  - Call `finalize_capture()` for current capture
- [ ] If pending finalization exists:
  - Wait briefly
  - Call `finalize_capture()` for pending
- [ ] Stop Callosum connection
- [ ] Log shutdown complete

### 3.9 Wire up CLI arguments
- [ ] Add `--sck-cli-path` argument support
- [ ] Pass to ScreenCaptureKitManager constructor
- [ ] Test CLI invocation: `observe-macos --interval 300`

## Phase 4: Testing & Integration

### 4.1 Manual Testing
- [ ] Install PyObjC dependencies: `pip install -e ".[macos]"`
- [ ] Build and install sck-cli to PATH
- [ ] Run observer: `observe-macos --interval 60` (use 1 min for faster testing)
- [ ] Verify files created in journal directory
- [ ] Test activity detection (go idle, lock screen, etc.)
- [ ] Test window boundaries and file naming
- [ ] Test graceful shutdown (Ctrl-C)
- [ ] Verify Callosum events emitted
- [ ] Verify health files touched

### 4.2 Multi-Monitor Testing
- [ ] Test with single monitor
- [ ] Test with dual monitors (side-by-side)
- [ ] Test with three monitors
- [ ] Verify monitor metadata in video files
- [ ] Test monitor arrangement changes during capture

### 4.3 Edge Cases
- [ ] Test rapid screen lock/unlock
- [ ] Test system sleep/wake
- [ ] Test display disconnect/reconnect
- [ ] Test sck-cli crash/failure during capture
- [ ] Test disk full scenario
- [ ] Test very short capture durations
- [ ] Test very long capture durations (>5 min)

### 4.4 Integration with Downstream Tools
- [ ] Verify observe-describe works with .mov files
- [ ] Verify observe-sense works with .m4a audio
- [ ] Test or update tools expecting .flac to handle .m4a
- [ ] Test monitor metadata parsing from video titles
- [ ] Verify think-indexer handles new file formats

## Phase 5: sck-cli Enhancements

### 5.1 High Priority: Monitor Metadata Capture
**Investigate:**
- [ ] Can `SCShareableContent` provide display arrangement/geometry?
- [ ] Research `SCDisplay` properties (displayID, width, height, frame)
- [ ] Can we get display position in global coordinate space?
- [ ] Can we distinguish between primary and secondary displays?

**Implement:**
- [ ] Add code to capture monitor geometry at capture start
- [ ] Store in video metadata (QuickTime user data or title field)
- [ ] Format: `"0:center,0,0,1920,1080 1:right,1920,0,3840,1080"`
- [ ] Test with multiple monitor configurations
- [ ] Document in sck-cli README

**Benefits:**
- Enables per-monitor analysis in observe-describe
- Matches GNOME screencast format for compatibility
- Essential for downstream processing

### 5.2 High Priority: Temp File Support
**Investigate:**
- [ ] Add CLI flag: `--temp` or `--hidden`
- [ ] When enabled, write to `.{basename}.mov` and `.{basename}.m4a`
- [ ] Python wrapper then renames after completion

**Implement:**
- [ ] Add flag to ArgumentParser
- [ ] Modify output path construction in SCKShot.swift
- [ ] Test that files are hidden on macOS (ls -a shows them)
- [ ] Document flag in README

**Benefits:**
- Prevents file watchers from triggering on incomplete files
- Cleaner integration with Sunstone's workflow
- Matches GNOME observer pattern

### 5.3 High Priority: Graceful Shutdown
**Investigate:**
- [ ] Verify current SIGTERM/SIGINT handling
- [ ] Ensure VideoWriter.finish() is called on interrupt
- [ ] Ensure AudioWriter finishes both tracks properly
- [ ] Test file validity after various interrupt scenarios

**Implement:**
- [ ] Add proper signal handlers if missing
- [ ] Ensure clean shutdown path exercises all finish() methods
- [ ] Test: Start capture, wait 5 sec, send SIGTERM, verify files valid
- [ ] Test: Start capture, wait 30 sec, send SIGINT, verify files valid

**Benefits:**
- Ensures data integrity
- Critical for reliable operation
- Prevents corrupt files on shutdown

### 5.4 Medium Priority: Multi-Display Support
**Investigate:**
- [ ] Currently captures "first" display - which one exactly?
- [ ] Can we capture multiple displays simultaneously?
- [ ] Would require multiple StreamOutput/VideoWriter instances
- [ ] Or capture combined virtual display space?

**Implement:**
- [ ] Add flag: `--display <id>` or `--display all`
- [ ] Allow specifying which display(s) to capture
- [ ] If "all", consider whether to:
  - Create separate files per display, or
  - Capture combined virtual space (current behavior?)
- [ ] Document display selection in README

**Benefits:**
- Flexibility for multi-monitor setups
- May reduce file size if only one display active
- Future-proofing

### 5.5 Medium Priority: Exit Code Validation
**Investigate:**
- [ ] Add validation before exit:
  - Check output files exist
  - Check files have non-zero size
  - Check video file is valid (can open with AVFoundation)
  - Check audio file is valid

**Implement:**
- [ ] Return exit code 0 only if all validations pass
- [ ] Return exit code 1 if capture failed
- [ ] Return exit code 2 if files missing/corrupt
- [ ] Log specific error messages to stderr

**Benefits:**
- Python wrapper can detect failures reliably
- Better error handling and debugging
- Prevents silent failures

### 5.6 Medium Priority: Metadata Embedding
**Investigate:**
- [ ] What metadata can be embedded in .mov container?
- [ ] QuickTime user data atoms for custom fields?
- [ ] Standard fields: title, comment, creation date, etc.
- [ ] Can we store capture settings (frame rate, duration, display ID)?

**Implement:**
- [ ] Add custom metadata fields:
  - Capture frame rate
  - Capture duration (planned)
  - Display ID(s) captured
  - Monitor geometry string
  - sck-cli version
- [ ] Use AVFoundation APIs to write metadata
- [ ] Test metadata survives file copy/move
- [ ] Document metadata fields

**Benefits:**
- Self-documenting files
- Enables smarter downstream processing
- Helpful for debugging capture issues

### 5.7 Low Priority: Frame Timestamp Accuracy
**Investigate:**
- [ ] Verify CMSampleBuffer presentation timestamps are accurate
- [ ] Test frame extraction at specific timestamps
- [ ] Ensure timestamps align with audio timestamps
- [ ] Test with different frame rates

**Implement:**
- [ ] Add logging of frame timestamps if not already present
- [ ] Validate timestamps against wall clock
- [ ] Document timestamp behavior in README

**Benefits:**
- Important for visual analysis alignment
- Ensures audio/video sync
- Critical for accurate playback

### 5.8 Low Priority: Output Path Validation
**Investigate:**
- [ ] Add validation before capture starts:
  - Parent directory exists
  - Parent directory is writable
  - Output files don't already exist
  - Sufficient disk space available

**Implement:**
- [ ] Add pre-flight checks in run() method
- [ ] Print clear error messages for each failure case
- [ ] Exit early if validation fails
- [ ] Document error messages

**Benefits:**
- Better user experience
- Prevents wasted capture attempts
- Clearer error messages

## Phase 6: Documentation & Polish

### 6.1 Documentation
- [ ] Add docstring examples to all public functions
- [ ] Create observe/macos/README.md with:
  - Installation instructions (PyObjC, sck-cli)
  - Usage examples
  - Configuration options
  - Troubleshooting guide
- [ ] Update main README.md to mention macOS support
- [ ] Document differences from GNOME observer

### 6.2 Code Quality
- [ ] Run `make format` to format all new code
- [ ] Run `make lint` and fix any issues
- [ ] Add type hints to all function signatures
- [ ] Add logging at appropriate levels (INFO, DEBUG, WARNING, ERROR)

### 6.3 Error Handling
- [ ] Review all TODO implementations for error handling
- [ ] Add try/except blocks where needed
- [ ] Ensure errors are logged with context
- [ ] Ensure errors don't crash the observer (graceful degradation)

## Notes

### Differences from GNOME Observer
- **Audio**: sck-cli provides synchronized .m4a instead of separate AudioRecorder
- **Format**: .mov video instead of .webm (or optionally convert with ffmpeg)
- **Activity APIs**: PyObjC instead of DBus
- **Subprocess**: Manages external sck-cli process instead of direct API calls
- **No RMS threshold**: Audio always captured when recording (rely on VAD post-processing)

### Key Design Decisions
- Use sck-cli's native audio to avoid synchronization complexity
- Mirror GNOME observer architecture for consistency
- Use PyObjC for native system APIs (parallels DBus approach)
- Accept .m4a format (update downstream tools if needed)
- Temp file pattern (`.HHMMSS`) prevents premature file watcher triggers

### Dependencies
- sck-cli must be built and available in PATH (or specified via --sck-cli-path)
- PyObjC frameworks required: core, Cocoa, Quartz
- Optional: ffmpeg for video metadata manipulation (if not using PyObjC)
- Optional: ffmpeg for .mov → .webm conversion (if desired)

### Testing Strategy
1. Start with activity.py (testable independently)
2. Then screencapture.py (can test with mock sck-cli)
3. Then observer.py (integration testing)
4. Finally sck-cli enhancements (separate repo)

### Future Enhancements
- Consider adding VAD post-processing to match GNOME's threshold logic
- Consider .mov → .webm conversion for format consistency
- Consider .m4a → .flac conversion if downstream tools require it
- Add observe-macos-test command for validation
- Add metrics/telemetry for capture success rates
