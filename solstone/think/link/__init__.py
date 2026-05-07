# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""link service — solstone's tunnel endpoint.

Forked from github.com/solpbc/spl home/ on 2026-04-20; the two copies are
now fully independent. The spl repo is the open-source protocol reference;
this is the canonical production implementation.

The service opens a listen WebSocket to spl-relay and terminates TLS 1.3
inside the solstone process. Paired devices reach convey's real routes
through the tunnel — there is no separate in-tunnel test app.

The wire protocol and frame types keep the "spl" name (spl_frame,
SplTunnelFrame, etc.); everything user-facing, architecturally-visible,
or journal-facing is "link" (service name, apps/link, journal/link,
sol call link, /link).
"""

__version__ = "0.1.0"

from .service import main  # noqa: E402 — re-exported so `sol link` can import it

__all__ = ["main"]
