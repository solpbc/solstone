# Solstone Platform

This repository defines the `platform.json` release metadata schema, generator, detached Minisign signer, destination adapters, and atomic publish rail for the Solstone platform.

## Key Principles & Guardrails

- **Scope**: This codebase generates, signs, and publishes multi-component `platform.json` release manifests, manages the atomic `latest` pointer, and maintains the POSIX platform installer template and build rail (`install.sh.in` -> `install.sh`). The platform installer is not yet published; `https://solstone.app/install.sh` remains the journal bootstrap.
- **Source of Truth**:
  - Embedded release pins: `pins/journal.pub`, `pins/desktop.pub`, `pins/tmux.pub`.
  - Production platform pin: `pins/platform.pub` + `pins/platform.keyid`. The private key never enters this repository; production signing still requires an explicitly supplied matching key and acknowledgement.
  - Compatibility floor: `compat/minimum_installer_revision`.
  - Installer revision: `compat/installer_revision`.
  - Component contracts: `contracts/journal.v1.json`, `contracts/desktop.v1.json`, `contracts/tmux.v1.json`.
  - Target architecture mappings: `src/solstone_platform/targets.py`.
- **Offline & Hermetic**: Tests and CI must not contact `updates.solstone.app`, Cloudflare R2, or other remote endpoints; use loopback fixtures for transport coverage.
- **Tooling Constraints**: Pure Python 3.12 standard library + host `minisign` binary. No third-party pip dependencies (`boto3`, `jsonschema`, etc.).
- **Security**: No secrets or private keys are ever stored in the repository. Ephemeral keys used in tests are created under temporary directories with `0700` permissions and removed when the fixture context exits.
