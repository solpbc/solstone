# solstone

the solstone app takes in what you share with it, and it all goes into your journal: an open source, local-first memory the agents you use can work from. on your devices, always private, only yours.

solstone is a personal memory platform, all yours, in two parts you own: **the solstone app** and your **journal**, the memory. from your journal emerges a knowledge graph of every person and project in your life, proactive meeting prep, automatic to-do tracking, and full-text search across your whole life. you stop managing your memory and start being present. your journal is always private, only yours.

open source and self-hostable, with optional services operated by [sol pbc](https://solpbc.org).

> **looking for the journal's code?** it moved to
> **[solstone-journal](https://github.com/solpbc/solstone-journal)**. this repo is now
> the family index and the home of solstone's platform release metadata and installer rail.

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

## platform metadata & installer rail

this repository generates, validates, signs, and publishes the platform-level `platform.json` release metadata manifest, manages the atomic `latest` pointer across release lanes, and provides the POSIX platform installer build rail (`install.sh.in` -> `install.sh`).

for linux and macos installation and verification, plus linux recovery, read [INSTALL.md](INSTALL.md).

save the canonical installer and inspect its options:

```sh
curl -fsSL https://solstone.app/install.sh -o install.sh
sh install.sh --help
```

check that the download succeeded before running the script. `https://solstone.app/platform-install.sh` remains a byte-identical compatibility URL. on macos, this installer acquires the native apps; it never installs a standalone journal runtime.

for authoritative definitions, refer to:
- native component pins: `pins/`
- current installer revision: `compat/installer_revision`
- minimum installer revision floor: `compat/minimum_installer_revision`
- component handler contracts: `contracts/`
- target architecture and package mappings: `src/solstone_platform/targets.py`

### development & verification commands

```bash
# Build (compile Python sources)
make build

# Build the production POSIX platform installer from the checked-in public pin
make build-installer

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

### recutting the linux platform catalogue

when a published linux component should become the platform default, prepare the
new catalogue directly from its signed release. the
[preparation path](src/solstone_platform/recut.py) validates the downloaded
artifacts, requires every unselected component entry to remain exact, and records
the `latest` version and ETag that publication is allowed to replace.

```bash
export SOLSTONE_R2_ENDPOINT="https://<account>.r2.cloudflarestorage.com"
export SOLSTONE_R2_BUCKET="<bucket>"
export AWS_ACCESS_KEY_ID="<access-key-id>"
export AWS_SECRET_ACCESS_KEY="<secret-access-key>"

PYTHONPATH=src python3 -m solstone_platform.cli prepare-recut \
  --version 2.0.1 \
  --replace journal=2.0.12 \
  --out /var/tmp/solstone-release/platform-2.0.1
```

inspect `platform.json` and `recut-receipt.json`, and compare the candidate's
`components` object with the signed base named in the receipt before signing.
signing remains a separate, explicit production step. publish the
signed candidate with its receipt:

```bash
SOLSTONE_PLATFORM_PRODUCTION=ack PYTHONPATH=src python3 -m solstone_platform.cli publish \
  --manifest /var/tmp/solstone-release/platform-2.0.1/platform.json \
  --signature /var/tmp/solstone-release/platform-2.0.1/platform.json.minisig \
  --journal-dir /var/tmp/solstone-release/platform-2.0.1/journal \
  --desktop-dir /var/tmp/solstone-release/platform-2.0.1/desktop \
  --tmux-dir /var/tmp/solstone-release/platform-2.0.1/tmux \
  --bootstrap-file /var/tmp/solstone-release/platform-2.0.1/bootstrap/solstone-journal-2.0.12-install.sh \
  --expected-latest-receipt /var/tmp/solstone-release/platform-2.0.1/recut-receipt.json \
  --acknowledge-production
```

if `latest` changed to a different release after preparation, publication
refuses without adopting the new base. prepare a new candidate from the current
release instead. on a retry, the
[publisher](src/solstone_platform/publish.py) reuses an immutable object only
when its bytes and metadata match, and uses at most one conditional write to
advance `latest`.

## why trust it with your life

solstone is built by [sol pbc](https://solpbc.org), a public benefit corporation. the
benefit purpose and the data covenants are written into our articles of incorporation,
not a privacy policy: your data is never sold, licensed, or used for anything but
serving you. they can't be amended without the founder's personal signature, and after
him the language can only get stronger, never weaker. or skip the trust question
entirely and run it yourself.

[product site](https://solstone.app) · [company](https://solpbc.org) · bluesky [@solstone.app](https://bsky.app/profile/solstone.app)
