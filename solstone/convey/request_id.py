# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Request ID stamping for Convey responses."""

from __future__ import annotations

import logging

from flask import Flask, Response, g

from solstone.convey.utils import generate_request_id

logger = logging.getLogger(__name__)


def install_request_id_stamper(app: Flask) -> None:
    @app.before_request
    def _stamp_request_id() -> None:
        try:
            g.request_id = generate_request_id()
        except Exception:
            g.request_id = ""
            logger.warning("request id generation failed", exc_info=True)

    @app.after_request
    def _stamp_request_id_header(response: Response) -> Response:
        response.headers["X-Solstone-Request-Id"] = getattr(g, "request_id", "")
        return response


__all__ = [
    "install_request_id_stamper",
]
