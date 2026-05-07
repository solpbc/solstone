# link service

The home-side tunnel endpoint for the spl protocol — solstone's long-term home for this code.

**Forked from [`github.com/solpbc/spl`](https://github.com/solpbc/spl) `home/` on 2026-04-20.**
The two copies are now fully independent: no pip dep, no submodule, no sync scripts.
The `spl` repo's `home/` continues as the open-source reference implementation of the protocol; this module is the canonical production implementation.

## layout

| File | Purpose |
|------|---------|
| `service.py` | Entry point + runtime. `sol link` runs `main()` here. |
| `relay_client.py` | Listen-WS + per-tunnel TLS pump. Spawns a task per incoming tunnel. |
| `tcp_pipe.py` | Byte pump that opens a loopback TCP connection to convey and forwards stream bytes both ways. |
| `tls_adapter.py` | pyOpenSSL memory-BIO adapter. Runs TLS 1.3 over opaque byte streams. |
| `ca.py` | Local CA lifecycle + CSR signing + home-attestation minting. |
| `auth.py` | `authorized_clients.json` reader/writer with mtime-reload and last-seen tracking. |
| `nonces.py` | Pair-ceremony nonce store (shared between CLI and convey pair route). |
| `mux.py` | Multiplex state machine per stream. |
| `framing.py` | Wire-frame encode/decode. |
| `paths.py` | Journal-path helpers + `SOL_LINK_RELAY_URL` resolution. |

## naming

- **link** — user-facing and architecturally-visible names: service name, convey app, `sol call link`, `journal/link/`, `/link` route.
- **spl** — protocol-level constructs: wire-format frames, JWT claim schemas, reset reason codes. These reference the external stable spl protocol and keep that name.

## privacy

No payload bytes are ever logged. The tunnel pump only emits rendezvous metadata (tunnel_id, stream_id, bytes_in, bytes_out, closed_reason) to logs and callosum. The CA private key never leaves `journal/link/ca/private.pem`; account tokens live in `journal/link/tokens/` and device tokens live on the phone.
