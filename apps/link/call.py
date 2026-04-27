# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI commands for the link tunnel service.

Auto-discovered by ``think.call`` and mounted as ``sol call link ...``.
"""

from __future__ import annotations

import datetime as dt
import socket
import time

import typer

from convey.utils import relative_time
from think.link.auth import AuthorizedClients
from think.link.ca import generate_nonce, load_or_generate_ca
from think.link.nonces import NONCE_TTL_SECONDS, NonceStore
from think.link.paths import (
    LinkState,
    authorized_clients_path,
    ca_dir,
    load_account_token,
    nonces_path,
    relay_url,
)
from think.utils import require_solstone

app = typer.Typer(
    help="Link — tunnel service for reaching this solstone from paired phones."
)


@app.callback()
def _require_up() -> None:
    require_solstone()


def _authorized() -> AuthorizedClients:
    return AuthorizedClients(authorized_clients_path())


def _nonces() -> NonceStore:
    return NonceStore(nonces_path())


def _detect_lan_ip() -> str | None:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        finally:
            sock.close()
    except OSError:
        return None


def _relative_time(iso: str | None) -> str:
    if not iso:
        return "never"
    try:
        then = dt.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.UTC)
    except ValueError:
        return iso
    now = dt.datetime.now(dt.UTC)
    delta_seconds = max(0, (now - then).total_seconds())
    return f"{relative_time(delta_seconds)} ago"


@app.command()
def pair(
    device_label: str = typer.Option(
        ..., "--device-label", help="Label for the phone being paired"
    ),
    convey_host: str = typer.Option(
        "",
        "--convey-host",
        help="Override host[:port] for the pair URL (default: auto-detect LAN IP)",
    ),
    convey_port: int = typer.Option(
        0,
        "--convey-port",
        help="Override convey port (default: read from service port file or 5015)",
    ),
    timeout_seconds: int = typer.Option(
        NONCE_TTL_SECONDS,
        "--timeout",
        help="How long to wait for the phone before giving up",
    ),
) -> None:
    """Mint a one-shot nonce, print the pair URL + QR-ready payload, wait for completion."""
    from think.utils import read_service_port

    value = generate_nonce()
    _nonces().add(value, device_label)
    ca_fp = load_or_generate_ca(ca_dir()).fingerprint_sha256()

    host = convey_host or _detect_lan_ip() or "localhost"
    port = convey_port or read_service_port("convey") or 5015
    base = f"http://{host}:{port}"
    url = f"{base}/app/link/pair?token={value}"

    typer.echo(f"Pair code: {value} (expires in 5 minutes)")
    typer.echo(f"Pair URL: {url}")
    typer.echo(f"CA fingerprint: sha256:{ca_fp}")
    typer.echo(f"Device: {device_label}")
    typer.echo("")
    typer.echo("Waiting for phone…")

    # Poll authorized_clients.json for a new entry.
    authorized = _authorized()
    before = {e.fingerprint for e in authorized.snapshot()}
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        time.sleep(1.0)
        current = authorized.snapshot()
        new_entries = [e for e in current if e.fingerprint not in before]
        if new_entries:
            entry = new_entries[-1]
            typer.echo(f"Paired: {entry.device_label}")
            typer.echo(f"  fingerprint: {entry.fingerprint}")
            typer.echo(f"  paired_at:   {entry.paired_at}")
            raise typer.Exit(0)
        # Also detect nonce consumption — if the nonce is gone/used, assume
        # the pair route fired but we missed the device (rare).
        nonce_entry = _nonces().peek(value)
        if nonce_entry and nonce_entry.used:
            typer.echo(
                "Pair request completed; device should appear in `sol call link list`."
            )
            raise typer.Exit(0)
    typer.echo("Timed out. Pair code expired.")
    raise typer.Exit(2)


@app.command("list")
def list_devices() -> None:
    """Print every paired device with its last-seen time."""
    entries = _authorized().snapshot()
    if not entries:
        typer.echo("No phones paired yet.")
        return
    for entry in entries:
        short_fp = entry.fingerprint.replace("sha256:", "")[:16]
        typer.echo(
            f"- {entry.device_label}"
            f" — added {_relative_time(entry.paired_at)}"
            f" — last seen {_relative_time(entry.last_seen_at)}"
            f" [{short_fp}]"
        )


@app.command()
def unpair(
    target: str = typer.Argument(
        ..., help="Device label or fingerprint (sha256:<hex>)"
    ),
) -> None:
    """Revoke a paired device. Next reconnect from that device fails at TLS handshake."""
    authorized = _authorized()
    if target.startswith("sha256:"):
        removed = authorized.remove(target)
        if not removed:
            typer.echo(f"No paired device with fingerprint {target}")
            raise typer.Exit(1)
        typer.echo("Unpaired.")
        return
    entry = authorized.find_by_label(target)
    if entry is None:
        typer.echo(f"No paired device with label {target!r}")
        raise typer.Exit(1)
    authorized.remove(entry.fingerprint)
    typer.echo("Unpaired.")


@app.command()
def status() -> None:
    """Report enrollment, listen-WS state, active tunnel count, relay endpoint."""
    state = LinkState.load_or_create()
    token = load_account_token()
    url = relay_url()
    entries = _authorized().snapshot()
    typer.echo(f"Instance ID:   {state.instance_id}")
    typer.echo(f"Home label:    {state.home_label}")
    typer.echo(f"Relay URL:     {url}")
    typer.echo(f"Enrolled:      {'yes' if token else 'no'}")
    typer.echo(f"Paired devices: {len(entries)}")
    # Listen-WS state and active-tunnel count live in the service process
    # memory — surfaced via callosum events rather than polled here. The
    # convey /app/link/api/status route is the live vantage.
    typer.echo("Listen-WS state: (query convey /app/link/api/status for live state)")
