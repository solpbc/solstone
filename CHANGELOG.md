# Changelog

All notable changes to solstone (the Python package) will be documented in this file.

Format adapted from [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), aligned with `cmo/brand/changelog-voice.md`.

## [0.3.4] - 2026-05-16

### Added
- a fresh journal now opens with a useful set of starred apps in the nav rail instead of a blank one. if you've already arranged your own starred apps, your choices are left exactly as they are.

### Changed
- the deprecated `precision` setting for parakeet transcription has been removed. `quantization` (auto, fp32, or int8) is the setting to use. if your journal config still carries the old `precision` line it's now simply ignored, with no change to how transcription runs.

### Fixed
- browsing back from the all-facets entity edit view now returns you to the entity you were looking at, in the same facet. before, back could land you on a different view.
- the bundled `transcripts read` documentation now shows the correct options. the previous example listed the wrong units for `--start` and `--length`, so following it as written would have failed.

## [0.3.3] - 2026-05-16

### Added
- a validate button now sits next to the gemini api key on the setup page, so you can confirm the key works before finalizing.

### Changed
- the setup page is reworked: cleaner typography, retention preferences as three explicit choices (always keep, keep for a set number of days, don't retain), enter-to-submit from any field, and your journal version and path surfaced up top.
- a fresh `sol setup` now installs the solstone bundle into all three coding-agent configs (claude, codex, gemini) at once, and lands the per-talent skill files in your journal so sol's sub-agents can find them.

### Fixed
- the setup page works end-to-end on a fresh install. earlier builds had a silent javascript bug that left the validate button, retention radios, and finalize submit unresponsive.
- on macos, your local timezone now resolves correctly on first setup. earlier installs could land in utc because the resolver missed where macos stores zone data.
