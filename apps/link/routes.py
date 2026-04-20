# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""link app routes — pair ceremony + paired-device dashboard.

All user-facing work for the spl tunnel integration happens here. The
protocol-level code (TLS, framing, mux) lives in `think/link/`; this
module is the HTTP surface that mobiles and the convey UI hit.

Routes:

  GET  /link                    dashboard (paired devices + pair button)
  GET  /link/qr.png             QR image for an active nonce (via ?token=)
  POST /link/pair-start         generate a new nonce + return QR payload
  POST /link/pair               mobile posts CSR + nonce; we sign + attest
  POST /link/unpair             remove a fingerprint (immediate revocation)
  GET  /link/api/devices        JSON list of paired devices for JS polling
  GET  /link/api/status         service status (for dashboard refresh)

The pair hop is plain HTTP on convey's existing listener — there is no
separate port. Integrity is provided by the CA-fingerprint pinned in the
QR, not by transport TLS. A MITM on the LAN can observe the nonce but
cannot forge a cert signed by the pinned CA.
"""

from __future__ import annotations

import datetime as dt
import logging
import socket
from typing import Any

from cryptography.hazmat.primitives import serialization
from flask import Blueprint, jsonify, request

from think.link.auth import AuthorizedClients, ClientEntry
from think.link.ca import generate_nonce, load_or_generate_ca, mint_attestation, sign_csr
from think.link.nonces import NonceStore
from think.link.paths import (
    LinkState,
    authorized_clients_path,
    ca_dir,
    load_account_token,
    nonces_path,
    relay_url,
)

logger = logging.getLogger(__name__)

link_bp = Blueprint(
    "app:link",
    __name__,
    url_prefix="/app/link",
)


def _authorized() -> AuthorizedClients:
    return AuthorizedClients(authorized_clients_path())


def _nonces() -> NonceStore:
    return NonceStore(nonces_path())


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _lan_pair_base_url() -> str:
    """Best-effort LAN URL for the convey host — used in the QR payload."""
    host = request.host
    scheme = "http" if not request.is_secure else "https"
    # If the request came in on localhost, substitute a routable LAN IP so
    # the phone's QR scan works. If we can't find one, fall back to host.
    try:
        hostname, _, port = host.partition(":")
        if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            lan_ip = _detect_lan_ip()
            if lan_ip:
                host = f"{lan_ip}:{port}" if port else lan_ip
    except Exception:
        logger.debug("lan ip detection failed", exc_info=True)
    return f"{scheme}://{host}"


def _detect_lan_ip() -> str | None:
    """Pick a reasonable LAN-facing IPv4 by opening a UDP socket.

    No packets are sent — we just read what src address the kernel would
    pick for a route to an external host. Returns None on any error.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        finally:
            sock.close()
    except OSError:
        return None


def _ca_fingerprint() -> str:
    ca = load_or_generate_ca(ca_dir())
    return ca.fingerprint_sha256()


def _is_lan_accessible() -> bool:
    """Check whether convey is bound to a non-loopback interface.

    Used to drive the "enable LAN access" nudge on /link. Best-effort: the
    signal is the Host header the dashboard loaded under.
    """
    hostname, _, _ = request.host.partition(":")
    if hostname in ("localhost", "127.0.0.1", "::1"):
        return bool(_detect_lan_ip())
    return True


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------


@link_bp.route("/api/devices")
def api_devices() -> Any:
    """JSON list of paired devices — used by the dashboard JS."""
    entries = _authorized().snapshot()
    devices = [_entry_to_json(e) for e in entries]
    return jsonify({"devices": devices})


@link_bp.route("/api/status")
def api_status() -> Any:
    """Snapshot of link-service state for the dashboard header."""
    state = LinkState.load_or_create()
    token = load_account_token()
    ca_fp = _ca_fingerprint() if ca_dir().exists() else None
    return jsonify(
        {
            "instance_id": state.instance_id,
            "home_label": state.home_label,
            "enrolled": token is not None,
            "relay_url": relay_url(),
            "ca_fingerprint": ca_fp,
            "lan_accessible": _is_lan_accessible(),
        }
    )


# ---------------------------------------------------------------------------
# pair ceremony
# ---------------------------------------------------------------------------


@link_bp.route("/pair-start", methods=["POST"])
def pair_start() -> Any:
    """Generate a single-use 5-minute nonce and return QR-ready payload."""
    payload = request.get_json(silent=True) or {}
    device_label = str(payload.get("device_label") or "").strip() or "unnamed device"

    nonce = generate_nonce()
    _nonces().add(nonce, device_label)

    ca_fp = _ca_fingerprint()
    pair_url = f"{_lan_pair_base_url()}/app/link/pair?token={nonce}"
    # The QR payload is a stable shape the mobile can parse — keep it
    # versioned so future mobiles stay backward-compatible.
    qr_payload = {
        "v": 1,
        "kind": "spl-pair",
        "pair_url": pair_url,
        "ca_fingerprint": ca_fp,
        "expires_in": 300,
        "device_label": device_label,
    }
    return jsonify(
        {
            "nonce": nonce,
            "pair_url": pair_url,
            "qr_payload": qr_payload,
            "ca_fingerprint": ca_fp,
            "expires_in": 300,
            "device_label": device_label,
        }
    )


@link_bp.route("/pair", methods=["POST"])
def pair() -> Any:
    """Mobile pair endpoint — accepts CSR + nonce, signs + mints attestation.

    Query: `?token=<nonce>` (the nonce minted by /pair-start).
    Body  (JSON):
        {
          "csr":          "<PEM>",      // required
          "device_label": "<string>",   // optional (falls back to nonce label)
          "nonce":        "<hex>"       // optional: may be in body instead of query
        }

    Response on success (200):
        {
          "client_cert":       "<PEM>",
          "ca_chain":          ["<PEM>", ...],
          "instance_id":       "<uuid>",
          "home_label":        "<string>",
          "home_attestation":  "<JWT>",
          "fingerprint":       "sha256:<hex>"
        }
    """
    body = request.get_json(silent=True) or {}
    nonce_value = request.args.get("token") or body.get("nonce")
    csr_pem = body.get("csr")
    device_label = str(body.get("device_label") or "").strip() or "unnamed device"

    if not isinstance(nonce_value, str) or not isinstance(csr_pem, str):
        return jsonify({"error": "missing fields (nonce + csr required)"}), 400

    consumed = _nonces().consume(nonce_value)
    if consumed is None:
        return jsonify({"error": "nonce expired or used"}), 410

    effective_label = device_label or (consumed.device_label or "unnamed device")

    ca = load_or_generate_ca(ca_dir())
    try:
        client_cert_pem, fingerprint = sign_csr(ca, csr_pem, effective_label)
    except Exception as exc:
        logger.info("pair: bad csr: %s", exc)
        return jsonify({"error": f"bad csr: {exc}"}), 400

    state = LinkState.load_or_create()
    authorized = _authorized()
    authorized.add(
        fingerprint=fingerprint,
        device_label=effective_label,
        instance_id=state.instance_id,
        paired_at=_utc_now_iso(),
    )
    attestation = mint_attestation(ca, state.instance_id, fingerprint)
    ca_chain_pem = ca.cert.public_bytes(serialization.Encoding.PEM).decode("ascii")
    return jsonify(
        {
            "client_cert": client_cert_pem,
            "ca_chain": [ca_chain_pem],
            "instance_id": state.instance_id,
            "home_label": state.home_label,
            "home_attestation": attestation,
            "fingerprint": fingerprint,
        }
    )


@link_bp.route("/unpair", methods=["POST"])
def unpair() -> Any:
    """Revoke a paired device by label or fingerprint.

    Body (JSON): {"fingerprint": "sha256:..."} or {"device_label": "..."}
    """
    body = request.get_json(silent=True) or {}
    fingerprint = body.get("fingerprint")
    device_label = body.get("device_label")
    if not isinstance(fingerprint, str):
        if not isinstance(device_label, str):
            return jsonify({"error": "fingerprint or device_label required"}), 400
        entry = _authorized().find_by_label(device_label)
        if entry is None:
            return jsonify({"error": "no paired device with that label"}), 404
        fingerprint = entry.fingerprint

    removed = _authorized().remove(fingerprint)
    if not removed:
        return jsonify({"error": "fingerprint not paired"}), 404
    return jsonify({"unpaired": fingerprint})


def _entry_to_json(entry: ClientEntry) -> dict[str, Any]:
    short_fp = entry.fingerprint.replace("sha256:", "")[:16]
    return {
        "fingerprint": entry.fingerprint,
        "fingerprint_short": short_fp,
        "device_label": entry.device_label,
        "paired_at": entry.paired_at,
        "last_seen_at": entry.last_seen_at,
    }


# ---------------------------------------------------------------------------
# helpers for the workspace template
# ---------------------------------------------------------------------------


@link_bp.app_context_processor
def _inject_link_helpers() -> dict[str, Any]:
    """Make `url_for` to link endpoints easy from templates."""
    return {}
