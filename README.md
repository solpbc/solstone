# solstone

the solstone app takes in what you see and hear, and it all goes into your journal: an open source, local-first memory the agents you use can work from. on your devices, always private, only yours.

solstone is a personal memory platform, all yours, in two parts you own: **the solstone app** and your **journal**, the memory. from your journal emerges a knowledge graph of every person and project in your life, proactive meeting prep, automatic to-do tracking, and full-text search across your whole life. you stop managing your memory and start being present. your journal is always private, only yours.

open source. self-host it, or let [sol pbc](https://solpbc.org) operate it for you.

> **looking for the journal's code?** it moved to
> **[solstone-journal](https://github.com/solpbc/solstone-journal)**. this repo is now
> an index of the whole solstone family — there's no code here, pick a repo below.

## the family

| repo | what it is |
|------|-----------|
| **[solstone-journal](https://github.com/solpbc/solstone-journal)** | the journal that holds and keeps your memories. |
| **[solstone-macos](https://github.com/solpbc/solstone-macos)** | the solstone macos app. |
| **[solstone-linux](https://github.com/solpbc/solstone-linux)** | the linux desktop app. |
| **[solstone-windows](https://github.com/solpbc/solstone-windows)** | the windows app. |
| **[solstone-android](https://github.com/solpbc/solstone-android)** | the android app. |
| **[solstone-swift](https://github.com/solpbc/solstone-swift)** | the ios app. |
| **[solstone-tmux](https://github.com/solpbc/solstone-tmux)** | the tmux app. |

## platform metadata & publish rail

This repository generates, validates, signs, and publishes the platform-level `platform.json` release metadata manifest and manages the atomic `latest` pointer across release lanes.

> **Note**: This repository does not install Solstone or replace the POSIX bootstrap installer (`https://solstone.app/install.sh`). Production key cutover, component native releases, live publication, installer integration, and URL cutovers are handled in downstream and deployment stages.

For authoritative definitions, refer to:
- Native component pins: `pins/`
- Minimum installer revision floor: `compat/minimum_installer_revision`
- Component handler contracts: `contracts/`
- Target architecture and package mappings: `src/solstone_platform/targets.py`

### Development & Verification Commands

```bash
# Build (compile Python sources)
make build

# Run unit and regression test suite
make test

# Lint source files and check headers
make lint

# Verify formatting and encoding
make format

# Run full hermetic CI gate (offline, non-root)
make ci

# Run CLI directly
PYTHONPATH=src python3 -m solstone_platform.cli --help
```

## why trust it with your life

solstone is built by [sol pbc](https://solpbc.org), a public benefit corporation. the
benefit purpose and the data covenants are written into our articles of incorporation,
not a privacy policy: your data is never sold, licensed, or used for anything but
serving you. they can't be amended without the founder's personal signature, and after
him the language can only get stronger, never weaker. or skip the trust question
entirely and run it yourself.

[product site](https://solstone.app) · [company](https://solpbc.org) · bluesky [@solstone.app](https://bsky.app/profile/solstone.app)
