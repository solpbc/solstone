# link service

The home-side tunnel endpoint for the spl protocol — solstone's long-term home for this code.

**Forked from [`github.com/solpbc/spl`](https://github.com/solpbc/spl) `home/` on 2026-04-20.**
The two copies are now fully independent: no pip dep, no submodule, no sync scripts.
The `spl` repo's `home/` continues as the open-source reference implementation of the protocol; this module is the canonical production implementation.

## layout

| File | Purpose |
|------|---------|
| `service.py` | Entry point + runtime. `sol link` runs `main()` here. |
| `relay_client.py` | Listen-WS + per-tunnel raw byte pipe to Convey's secure listener. Spawns a task per incoming tunnel. |
| `ca.py` | Local CA lifecycle + CSR signing + home-attestation minting. |
| `auth.py` | `authorized_clients.json` reader/writer with mtime-reload and last-seen tracking. |
| `nonces.py` | Pair-ceremony nonce store (shared between CLI and convey pair route). |
| `paths.py` | Journal-path helpers + `SOL_LINK_RELAY_URL` resolution. |

TLS termination, multiplexing, and inline WSGI dispatch now live in
`solstone/convey/secure_listener/`, because Convey owns both listening ports:
the DL web port and the PL secure-listener port 7657.

## naming

- **link** — user-facing and architecturally-visible names: service name, convey app, `sol call link`, `journal/link/`, `/link` route.
- **spl** — protocol-level constructs: wire-format frames, JWT claim schemas, reset reason codes. These reference the external stable spl protocol and keep that name.

## privacy

No payload bytes are ever logged. The link service is the spl-relay-tunnel client: it opens the outbound relay WebSocket, accepts incoming tunnels, and pipes bytes to Convey's secure listener on `127.0.0.1:7657`. The CA private key never leaves `journal/link/ca/private.pem`; account tokens live in `journal/link/tokens/` and device tokens live on the phone.
