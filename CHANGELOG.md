# Changelog

All notable changes to solstone (the Python package) will be documented in this file.

Format adapted from [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), aligned with `cmo/brand/changelog-voice.md`.

## [0.3.3] - 2026-05-16

### Added
- a validate button now sits next to the gemini api key on the setup page, so you can confirm the key works before finalizing.

### Changed
- the setup page is reworked: cleaner typography, retention preferences as three explicit choices (always keep, keep for a set number of days, don't retain), enter-to-submit from any field, and your journal version and path surfaced up top.
- a fresh `sol setup` now installs the solstone bundle into all three coding-agent configs (claude, codex, gemini) at once, and lands the per-talent skill files in your journal so sol's sub-agents can find them.

### Fixed
- the setup page works end-to-end on a fresh install. earlier builds had a silent javascript bug that left the validate button, retention radios, and finalize submit unresponsive.
- on macos, your local timezone now resolves correctly on first setup. earlier installs could land in utc because the resolver missed where macos stores zone data.
