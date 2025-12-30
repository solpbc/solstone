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

## Phase 2: ScreenCaptureKit Manager (screencapture.py)

**Note:** sck-cli now provides multi-display capture with JSONL metadata output to stdout.
Display geometry is parsed from sck-cli output - no PyObjC monitor detection needed.

### 2.1 JSONL Parsing (DONE)
- [x] Parse sck-cli stdout for display geometry
- [x] Extract displayID, x, y, width, height per display
- [x] Use `assign_monitor_positions()` to compute position labels
- [x] Build DisplayInfo objects with position, displayID, temp_path

### 2.2 Implement `start()` (DONE)
- [x] Build command with frame rate and duration
- [x] Launch subprocess and capture stdout
- [x] Parse JSONL for display and audio info
- [x] Return list of DisplayInfo and AudioInfo

### 2.3 Implement `stop()` (DONE)
- [x] Send SIGTERM to process
- [x] Wait with timeout for graceful shutdown
- [x] SIGKILL as fallback

### 2.4 Implement `finalize()` (DONE)
- [x] Simple file rename (no metadata embedding needed)
- [x] Rename per-display: `temp_displayID.mov` -> `HHMMSS_LEN_position_displayID_screen.mov`
- [x] Rename audio: `temp.m4a` -> `HHMMSS_LEN_audio.m4a`

### 2.5 Implement `get_output_size()` (DONE)
- [x] Sum sizes of all display video files
- [x] Used for health check file growth verification

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
  - Get captured displays/audio from manager
  - Build finalization list: (temp_path, final_path) tuples
  - Queue for finalization: `self.pending_finalization = [...]`
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
- [ ] Store returned displays and audio in instance variables
- [ ] Set `self.capture_running = True`
- [ ] Initialize file size tracking
- [ ] Log capture start with display info
- [ ] Return True on success, False on failure

### 3.5 Implement `emit_status()`
- [ ] Build capture info dict:
  - If capturing: `{"recording": True, "displays": [...], "window_elapsed_seconds": ...}`
  - Else: `{"recording": False}`
- [ ] Build activity info dict: `{"active": ..., "idle_time_ms": ..., "screen_locked": ...}`
- [ ] Emit via Callosum: `self.callosum.emit("observe", "status", capture=..., activity=...)`

### 3.6 Implement `finalize_screencast()` (DONE)
- [x] Simple file rename using os.replace()
- [x] Log success/failure

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
  - Emit status event with `files_growing` field
- [ ] Call `shutdown()` after loop exits

### 3.8 Implement `shutdown()`
- [ ] If capture running:
  - Stop capture
  - Wait briefly (1 second) for files to be written
  - Build finalization list
  - Process finalizations
- [ ] If pending finalization exists:
  - Wait briefly
  - Process pending finalizations
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

### 4.2 Multi-Monitor Testing
- [ ] Test with single monitor (position should be "center")
- [ ] Test with dual monitors (side-by-side)
- [ ] Test with three monitors
- [ ] Verify per-display files with position labels
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
- [ ] Verify observe-sense dispatches .mov to describe and .m4a to transcribe
- [ ] Test parse_screen_filename() with new displayID format
- [ ] Verify think-indexer handles new file formats

## Phase 5: sck-cli Enhancements

### 5.1 Multi-Display Support (DONE in sck-cli)
- [x] Captures all displays simultaneously
- [x] Creates `<base>_<displayID>.mov` per display
- [x] Outputs JSONL with display geometry to stdout

### 5.2 Temp File Support
- [ ] Add CLI flag: `--temp` or `--hidden`
- [ ] When enabled, write to `.{basename}_<displayID>.mov` and `.{basename}.m4a`
- [ ] Python wrapper then renames after completion
- [ ] Prevents file watchers from triggering on incomplete files

### 5.3 Graceful Shutdown
- [ ] Verify current SIGTERM/SIGINT handling
- [ ] Ensure VideoWriter.finish() is called on interrupt
- [ ] Ensure AudioWriter finishes both tracks properly
- [ ] Test file validity after various interrupt scenarios

### 5.4 Exit Code Validation
- [ ] Add validation before exit:
  - Check output files exist
  - Check files have non-zero size
  - Check video file is valid
  - Check audio file is valid
- [ ] Return exit code 0 only if all validations pass
- [ ] Return exit code 1 if capture failed
- [ ] Return exit code 2 if files missing/corrupt

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

### Architecture Changes from Original Plan
- **No PyObjC monitor detection needed**: sck-cli provides display geometry via JSONL stdout
- **No metadata embedding**: Position/displayID encoded in filename instead
- **Multi-display from day one**: sck-cli captures all displays automatically
- **DisplayInfo dataclass**: Mirrors GNOME's StreamInfo pattern

### File Naming Convention
- **Video**: `HHMMSS_LEN_position_displayID_screen.mov` (e.g., `120000_300_center_1_screen.mov`)
- **Audio**: `HHMMSS_LEN_audio.m4a` (e.g., `120000_300_audio.m4a`)
- **Temp files**: `.HHMMSS_displayID.mov`, `.HHMMSS.m4a` (hidden during capture)

### Differences from GNOME Observer
- **Audio**: sck-cli provides synchronized .m4a instead of separate AudioRecorder
- **Format**: .mov video instead of .webm
- **Activity APIs**: PyObjC instead of DBus
- **Subprocess**: Manages external sck-cli process instead of direct API calls
- **Connector ID**: Uses numeric displayID instead of connector names like "DP-3"
- **No RMS threshold**: Audio always captured when recording

### Dependencies
- sck-cli must be built and available in PATH (or specified via --sck-cli-path)
- PyObjC frameworks required: core, Cocoa, Quartz (for activity detection only)
- observe.utils.assign_monitor_positions for position label computation

### Testing Strategy
1. Start with activity.py (testable independently)
2. Then screencapture.py (can test with mock sck-cli or real capture)
3. Then observer.py (integration testing)
4. Finally sck-cli enhancements (separate repo)
