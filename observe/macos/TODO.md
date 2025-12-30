# macOS Observer Implementation TODO

This document tracks the remaining work to complete the macOS observer integration using sck-cli and ScreenCaptureKit.

## Phase 1: Activity Detection (activity.py) - DONE

### 1.1 Implement `get_idle_time_ms()` (DONE)
- [x] Import PyObjC Quartz framework
- [x] Use `CGEventSourceSecondsSinceLastEventType(1, kCGAnyInputEventType)`
- [x] Convert seconds to milliseconds
- [x] Add error handling for API failures
- [x] Test on macOS system

### 1.2 Implement `is_screen_locked()` (DONE)
- [x] Used CGSessionCopyCurrentDictionary for kCGSSessionOnConsoleKey
- [x] Add error handling
- [x] Test on macOS system

### 1.3 Implement `is_power_save_active()` (DONE)
- [x] Used CGDisplayIsAsleep(CGMainDisplayID())
- [x] Add error handling
- [x] Test on macOS system

### 1.4 Implement `is_output_muted()` (DONE)
- [x] Used osascript to query volume settings
- [x] Add error handling and timeout
- [x] Test on macOS system

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

## Phase 3: Main Observer (observer.py) - DONE

### 3.1 Implement `setup()` (DONE)
- [x] Verify sck-cli is available in PATH via shutil.which()
- [x] Initialize Callosum connection
- [x] Start Callosum connection
- [x] Log initialization success
- [x] Return True on success, False on failure

### 3.2 Implement `check_activity_status()` (DONE)
- [x] Call `get_idle_time_ms()` from activity module
- [x] Call `is_screen_locked()` from activity module
- [x] Call `is_output_muted()` from activity module
- [x] Cache values in instance variables for status events
- [x] Determine if idle: `(idle_time > IDLE_THRESHOLD_MS) or screen_locked`
- [x] Return activity status

### 3.3 Implement `handle_boundary()` (DONE)
- [x] Get timestamp parts and calculate duration
- [x] Stop capture if running
- [x] Check audio threshold (3-chunk RMS logic) before saving audio
- [x] Build finalization list and queue
- [x] Reset timing for new window
- [x] Start new capture if active and screen not locked
- [x] Emit Callosum observing event with saved files

### 3.4 Implement `initialize_capture()` (DONE)
- [x] Get timestamp for filename
- [x] Build temp output base (hidden file)
- [x] Start sck-cli via ScreenCaptureKitManager
- [x] Store displays and audio info
- [x] Initialize file size tracking
- [x] Log capture start with display info

### 3.5 Implement `emit_status()` (DONE)
- [x] Build capture info dict with recording status, displays, elapsed time, files_growing
- [x] Build activity info dict with active, idle_time_ms, screen_locked, output_muted
- [x] Emit via Callosum

### 3.6 Implement `finalize_screencast()` (DONE)
- [x] Simple file rename using os.replace()
- [x] Log success/failure

### 3.7 Implement `main_loop()` (DONE)
- [x] Check initial activity status
- [x] Start initial capture if active
- [x] Main loop with CHUNK_DURATION sleep intervals
- [x] Process pending finalizations
- [x] Check activity status and detect activation edge
- [x] Detect mute state transitions (triggers boundary like GNOME)
- [x] Handle window boundaries
- [x] Track file growth for health reporting
- [x] Emit status events

### 3.8 Implement `shutdown()` (DONE)
- [x] Stop capture if running
- [x] Check audio threshold for final segment
- [x] Finalize all pending captures
- [x] Stop Callosum connection

### 3.9 Implement `_check_audio_threshold()` (DONE)
- [x] Decode m4a with PyAV
- [x] Split into 5-second chunks
- [x] Compute RMS per chunk
- [x] Count threshold hits (same MIN_HITS_FOR_SAVE = 3 as GNOME)
- [x] Return True if enough voice activity

### 3.10 Wire up CLI arguments (DONE)
- [x] Pass --sck-cli-path to ScreenCaptureKitManager

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

## Phase 5: sck-cli (DONE)

All sck-cli requirements are met:
- [x] Multi-display capture with per-display files
- [x] JSONL metadata output to stdout
- [x] Temp file support (Python passes hidden path like `.HHMMSS`)
- [x] Graceful SIGTERM/SIGINT handling (verified)
- [x] File validation done in Python's `finalize()`

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
