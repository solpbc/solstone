# macOS Observer TODO

Tracks remaining work for the macOS observer integration.

## Completed

- **Phase 1: Activity Detection** (`activity.py`) - All done
- **Phase 2: ScreenCaptureKit Manager** (`screencapture.py`) - All done
- **Phase 3: Main Observer** (`observer.py`) - All done
- **Phase 5: sck-cli** - All requirements met

## Phase 4: Testing & Integration

### 4.1 Manual Testing
- [x] Install PyObjC dependencies (now automatic via `pip install -e .`)
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

## Phase 6: Documentation & Polish

### 6.1 Documentation
- [ ] Create observe/macos/README.md with installation and usage
- [ ] Update main README.md to mention macOS support
- [ ] Document differences from Linux observer

### 6.2 Code Quality
- [x] Run `make format` and `make lint`
- [x] Add type hints to function signatures
- [x] Proper logging at appropriate levels

---

## Reference

### File Naming Convention
- **Video**: `HHMMSS_LEN_position_displayID_screen.mov` (e.g., `120000_300_center_1_screen.mov`)
- **Audio**: `HHMMSS_LEN_audio.m4a` (e.g., `120000_300_audio.m4a`)
- **Temp files**: `.HHMMSS_displayID.mov`, `.HHMMSS.m4a` (hidden during capture)

### Differences from Linux Observer
- **Audio threshold**: macOS checks at boundary (post-capture), Linux checks real-time
- **Format**: .mov video instead of .webm, .m4a audio instead of .flac
- **Activity APIs**: PyObjC/Quartz instead of DBus
- **Capture**: External sck-cli process instead of GStreamer/PipeWire
- **Connector ID**: Numeric displayID instead of connector names like "DP-3"
- **No tmux mode**: macOS observer only has screencast/idle modes

### Dependencies
- sck-cli must be built and available in PATH (or specified via --sck-cli-path)
- PyObjC frameworks (core, Cocoa, Quartz) - installed automatically on macOS via pip
- observe.utils.assign_monitor_positions for position label computation
