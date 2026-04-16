# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""
Utility functions for safely saving and verifying voiceprint .npz files.

This module introduces a file-locking mechanism to prevent race conditions
during concurrent read-modify-write operations on voiceprint files,
and adds an integrity check after writing.
"""

import fcntl
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def save_voiceprints_safely(
    npz_path: Path, embeddings: np.ndarray, metadata: dict
) -> None:
    """
    Safely saves voiceprint data to an NPZ file with file locking and integrity check.

    Acquires an exclusive lock on the file, writes to a temporary file, renames it
    atomically, and then performs an integrity check by attempting to reload the file.
    If the integrity check fails, the file is logged as corrupt and optionally quarantined.

    Args:
        npz_path: The final path to the .npz file.
        embeddings: The numpy array of embeddings to save.
        metadata: The metadata dictionary to save.

    Raises:
        Exception: If the file locking or atomic rename fails, or if the integrity check fails.
    """
    lock_path = npz_path.with_suffix(".lock")
    tmp_path = npz_path.with_name(npz_path.stem + ".tmp.npz")

    # Ensure the directory exists
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Open and acquire an exclusive lock.
        # Use 'w' mode for the lock file to create it if it doesn't exist.
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX)  # Acquire exclusive lock

                # Save to a temporary file first
                # allow_pickle=False is generally safer, but metadata might contain complex types.
                # For voiceprints, assume numpy types are okay. If issues arise, revisit this.
                np.savez_compressed(tmp_path, embeddings=embeddings, metadata=metadata)

                # Atomically rename the temporary file to the final destination
                # This operation is generally atomic on most file systems.
                if tmp_path.exists():
                    tmp_path.rename(npz_path)
                else:
                    # This should ideally not happen if np.savez_compressed succeeded
                    raise FileNotFoundError(
                        f"Temporary voiceprint file not found: {tmp_path}"
                    )

                # --- Integrity Check ---
                try:
                    # Attempt to load the file to verify its integrity
                    # allow_pickle=False is a good security practice if possible,
                    # but might need to be True if metadata contains arbitrary Python objects.
                    # For now, assume standard numpy savz_compressed data.
                    with np.load(npz_path, allow_pickle=False) as data:
                        # Basic check: ensure expected keys exist
                        if "embeddings" not in data or "metadata" not in data:
                            raise ValueError(
                                "Missing 'embeddings' or 'metadata' keys in loaded NPZ."
                            )
                    logger.info(
                        f"Successfully wrote and verified voiceprint file: {npz_path}"
                    )

                except (FileNotFoundError, ValueError, np.lib.npyio.NpzFile) as e:
                    logger.error(
                        f"CRITICAL: Integrity check failed for voiceprint file {npz_path} after write: {e}"
                    )
                    # Optional: Quarantine the corrupt file
                    # It's often better to log and alert than to auto-delete/rename immediately,
                    # unless a robust rollback strategy is in place.
                    # corrupt_path = npz_path.with_suffix(".npz.corrupt")
                    # if npz_path.exists():
                    #     npz_path.rename(corrupt_path)
                    #     logger.warning(f"Corrupt file quarantined to: {corrupt_path}")
                    # else:
                    #     logger.error(f"Corrupt file {npz_path} not found for quarantining.")
                    raise  # Re-raise the exception to signal failure

            except Exception as e:
                # Clean up the temporary file if it exists and an error occurred before rename
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError as rm_err:
                        logger.error(
                            f"Failed to clean up temporary file {tmp_path}: {rm_err}"
                        )
                raise e  # Re-raise the original exception

            finally:
                # Release the lock, regardless of success or failure
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                except OSError as unlock_err:
                    logger.error(f"Failed to release lock on {lock_path}: {unlock_err}")

    except OSError as e:
        # Handle errors related to opening/locking the lock file itself
        logger.error(f"Failed to acquire or manage lock file {lock_path}: {e}")
        raise e
