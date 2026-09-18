# Solstone Platform

This repository defines the `platform.json` release metadata schema, generator, detached Minisign signer, destination adapters, and atomic publish rail for the Solstone platform.

## Key Principles & Guardrails

- **Scope**: This codebase generates, signs, and publishes multi-component `platform.json` release manifests and manages the atomic `latest` pointer. It is **not** the POSIX bootstrap installer (`install.sh`), does not install system services, and does not run installer receipts.
- **Source of Truth**:
  - Embedded release pins: `pins/journal.pub`, `pins/desktop.pub`, `pins/tmux.pub`.
  - Production platform pin: **Deliberately absent** (`pins/platform.pub` does not exist; production signing requires explicitly supplied keys with acknowledgement).
  - Compatibility floor: `compat/minimum_installer_revision`.
  - Component contracts: `contracts/journal.v1.json`, `contracts/desktop.v1.json`, `contracts/tmux.v1.json`.
  - Target architecture mappings: `src/solstone_platform/targets.py`.
- **Offline & Hermetic**: CI and testing run completely offline. Never contact `updates.solstone.app`, Cloudflare R2, or remote endpoints during tests or CI.
- **Tooling Constraints**: Pure Python 3.12 standard library + host `minisign` binary. No third-party pip dependencies (`boto3`, `jsonschema`, etc.).
- **Security**: No secrets or private keys are ever stored in the repository. Ephemeral keys used in tests are created under temporary directories with `0700` permissions and cleaned up immediately.
