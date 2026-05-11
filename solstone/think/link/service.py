# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""link relay service runtime.

Registered with solstone's supervisor via `think/sol_cli.py` COMMANDS (see `sol link`);
the supervisor launches this as a subprocess alongside callosum, cortex,
convey, etc. Service lifecycle:

  start → load state + CA → ensure account_token (enroll once) →
    open listen WS to spl-relay → accept tunnel pairs → pipe raw bytes to
    Convey's secure listener on 127.0.0.1:7657. On disconnect, reconnect
    with exponential backoff.

Exits on SIGINT/SIGTERM with a clean close of the listen WS and all
in-flight tunnel WSes.

Callosum events are emitted on the `link` tract:
  enrolled     first-run account-token mint
  connecting   opening listen WS
  connected    listen WS open (service is reachable)
  disconnect   listen WS closed (about to reconnect)
  tunnel_pair  incoming tunnel (paired device dialed in)
  tunnel_close tunnel closed
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any

from solstone.think.callosum import CallosumConnection

from .ca import load_or_generate_ca
from .paths import (
    LinkState,
    ca_dir,
    load_account_token,
    relay_url,
    save_account_token,
)
from .relay_client import RelayClient

log = logging.getLogger("link.service")


async def run_service() -> None:
    """Build the relay client and run it until signaled."""
    state = LinkState.load_or_create()
    ca = load_or_generate_ca(ca_dir())
    token = load_account_token()

    callosum = CallosumConnection()
    callosum.start()

    def emit(event: str, fields: dict[str, Any]) -> None:
        try:
            callosum.emit("link", event, **fields)
        except Exception:
            log.debug("callosum emit failed", exc_info=True)

    client = RelayClient(
        instance_id=state.instance_id,
        home_label=state.home_label,
        relay_endpoint=relay_url(),
        account_token=token,
        on_account_token=save_account_token,
        ca_pubkey_spki_pem=ca.pubkey_spki_pem,
        callosum_emit=emit,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with _suppress_not_implemented():
            loop.add_signal_handler(sig, stop_event.set)

    run_task = asyncio.create_task(client.run(), name="link-relay-client")
    try:
        await stop_event.wait()
    finally:
        log.info("link service stopping")
        await client.stop()
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass
        callosum.stop()


class _suppress_not_implemented:
    """Context manager that swallows NotImplementedError for Windows/TTYs."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, _exc: Any, _tb: Any) -> bool:
        return exc_type is NotImplementedError


def main() -> None:
    """CLI entry point for `sol link` — starts the service."""
    import argparse

    from solstone.think.utils import require_solstone, setup_cli

    parser = argparse.ArgumentParser(description="solstone link tunnel service")
    args = setup_cli(parser)
    require_solstone()

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        asyncio.run(run_service())
    except KeyboardInterrupt:
        log.info("link service interrupted")


if __name__ == "__main__":
    main()
