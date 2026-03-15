# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Welcome-mat client for support.solpbc.org.

Implements the full DPoP + self-signed access token auth flow per the
welcome-mat spec and the extro-support SKILL.md interface contract.

All cryptographic operations use the ``cryptography`` library (already a
solstone dependency).  The keypair, access token, and cached TOS are
persisted in the journal's app storage directory.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

logger = logging.getLogger(__name__)

DEFAULT_PORTAL_URL = "https://support.solpbc.org"

# ---------------------------------------------------------------------------
# Base64url helpers
# ---------------------------------------------------------------------------


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    s += "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def _jwt_encode(header: dict, payload: dict, private_key: rsa.RSAPrivateKey) -> str:
    """Create a signed JWT (RS256)."""
    h = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    p = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{h}.{p}".encode()
    sig = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{h}.{p}.{_b64url_encode(sig)}"


def _sha256_b64url(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return _b64url_encode(hashlib.sha256(data).digest())


# ---------------------------------------------------------------------------
# JWK / Thumbprint
# ---------------------------------------------------------------------------


def _public_key_jwk(key: rsa.RSAPublicKey) -> dict:
    """Export RSA public key as a JWK dict."""
    numbers = key.public_numbers()
    e_bytes = numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, "big")
    n_bytes = numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")
    return {
        "kty": "RSA",
        "e": _b64url_encode(e_bytes),
        "n": _b64url_encode(n_bytes),
    }


def _jwk_thumbprint(jwk: dict) -> str:
    """RFC 7638 JWK thumbprint (SHA-256, base64url)."""
    # Canonical JSON: alphabetical keys
    canonical = json.dumps(
        {"e": jwk["e"], "kty": "RSA", "n": jwk["n"]},
        separators=(",", ":"),
        sort_keys=True,
    )
    return _sha256_b64url(canonical)


# ---------------------------------------------------------------------------
# Portal client
# ---------------------------------------------------------------------------


class PortalClient:
    """Welcome-mat client for the support portal.

    Parameters
    ----------
    portal_url:
        Base URL of the portal (no trailing slash).
    storage_dir:
        Directory for keypair, token cache, and TOS cache.
    handle:
        Agent handle for registration.  Defaults to the machine hostname.
    anonymous:
        If True, generate a random handle and don't persist the keypair.
    """

    def __init__(
        self,
        portal_url: str = DEFAULT_PORTAL_URL,
        storage_dir: Path | None = None,
        handle: str | None = None,
        anonymous: bool = False,
    ) -> None:
        self.portal_url = portal_url.rstrip("/")
        self.anonymous = anonymous
        self._handle = handle

        if storage_dir is None:
            from apps.utils import get_app_storage_path

            storage_dir = get_app_storage_path("support", "portal", ensure_exists=True)

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._private_key: rsa.RSAPrivateKey | None = None
        self._access_token: str | None = None
        self._tos_text: str | None = None
        self._jwk: dict | None = None
        self._thumbprint: str | None = None

        self._load_state()

    # -- Persistence ---------------------------------------------------------

    @property
    def _keypair_path(self) -> Path:
        return self.storage_dir / "keypair.pem"

    @property
    def _token_path(self) -> Path:
        return self.storage_dir / "token.json"

    @property
    def _tos_cache_path(self) -> Path:
        return self.storage_dir / "tos.txt"

    @property
    def handle(self) -> str:
        if self._handle:
            return self._handle
        import socket

        hostname = socket.gethostname().lower().replace("_", "-")[:48]
        # Ensure valid handle format
        handle = "".join(c for c in hostname if c.isalnum() or c in ".-")
        handle = handle.strip(".-") or "solstone"
        self._handle = f"solstone-{handle}"
        return self._handle

    def _load_state(self) -> None:
        """Load persisted keypair and token."""
        if self.anonymous:
            return

        if self._keypair_path.is_file():
            pem = self._keypair_path.read_bytes()
            self._private_key = serialization.load_pem_private_key(pem, password=None)
            pub = self._private_key.public_key()
            self._jwk = _public_key_jwk(pub)
            self._thumbprint = _jwk_thumbprint(self._jwk)

        if self._token_path.is_file():
            try:
                data = json.loads(self._token_path.read_text())
                self._access_token = data.get("access_token")
                self._handle = data.get("handle", self._handle)
            except (json.JSONDecodeError, OSError):
                pass

        if self._tos_cache_path.is_file():
            try:
                self._tos_text = self._tos_cache_path.read_text()
            except OSError:
                pass

    def _save_keypair(self) -> None:
        if self.anonymous or self._private_key is None:
            return
        pem = self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        self._keypair_path.write_bytes(pem)
        self._keypair_path.chmod(0o600)

    def _save_token(self) -> None:
        if self.anonymous:
            return
        data = {"access_token": self._access_token, "handle": self._handle}
        self._token_path.write_text(json.dumps(data))

    def _save_tos(self, tos_text: str) -> None:
        self._tos_text = tos_text
        if not self.anonymous:
            self._tos_cache_path.write_text(tos_text)

    # -- Key management ------------------------------------------------------

    def _ensure_keypair(self) -> None:
        """Generate RSA-4096 keypair if we don't have one."""
        if self._private_key is not None:
            return

        logger.info("Generating RSA-4096 keypair for support portal registration")
        self._private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
        )
        pub = self._private_key.public_key()
        self._jwk = _public_key_jwk(pub)
        self._thumbprint = _jwk_thumbprint(self._jwk)
        self._save_keypair()

    # -- DPoP proof creation -------------------------------------------------

    def _create_dpop_proof(
        self,
        method: str,
        url: str,
        access_token: str | None = None,
    ) -> str:
        """Create a DPoP proof JWT per RFC 9449."""
        assert self._private_key is not None
        assert self._jwk is not None

        header = {
            "typ": "dpop+jwt",
            "alg": "RS256",
            "jwk": self._jwk,
        }
        payload: dict[str, Any] = {
            "jti": str(uuid.uuid4()),
            "htm": method,
            "htu": url.split("?")[0],  # strip query/fragment
            "iat": int(time.time()),
        }
        if access_token is not None:
            payload["ath"] = _sha256_b64url(access_token)

        return _jwt_encode(header, payload, self._private_key)

    # -- Access token creation -----------------------------------------------

    def _create_access_token(self, tos_text: str) -> str:
        """Create a self-signed wm+jwt access token."""
        assert self._private_key is not None
        assert self._thumbprint is not None

        header = {"typ": "wm+jwt", "alg": "RS256"}
        payload = {
            "jti": str(uuid.uuid4()),
            "tos_hash": _sha256_b64url(tos_text),
            "aud": self.portal_url,
            "cnf": {"jkt": self._thumbprint},
            "iat": int(time.time()),
        }
        return _jwt_encode(header, payload, self._private_key)

    # -- TOS signing ---------------------------------------------------------

    def _sign_tos(self, tos_text: str) -> str:
        """Sign TOS text with RS256 and return base64url signature."""
        assert self._private_key is not None
        sig = self._private_key.sign(
            tos_text.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return _b64url_encode(sig)

    # -- HTTP helpers --------------------------------------------------------

    def _http(self) -> httpx.Client:
        return httpx.Client(timeout=30.0)

    def _authed_headers(self, method: str, url: str) -> dict[str, str]:
        """Return Authorization + DPoP headers for an authenticated request."""
        assert self._access_token is not None
        return {
            "Authorization": f"DPoP {self._access_token}",
            "DPoP": self._create_dpop_proof(method, url, self._access_token),
        }

    def _authed_request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        params: dict | None = None,
        retry_on_tos: bool = True,
    ) -> httpx.Response:
        """Make an authenticated request, handling TOS re-consent."""
        url = f"{self.portal_url}{path}"
        headers = self._authed_headers(method, url)

        with self._http() as client:
            resp = client.request(
                method, url, headers=headers, json=json_body, params=params
            )

        if resp.status_code == 401 and retry_on_tos:
            try:
                body = resp.json()
            except Exception:
                body = {}
            if body.get("error") == "tos_changed":
                logger.info("TOS changed — re-registering")
                self.register()
                return self._authed_request(
                    method, path, json_body=json_body, params=params, retry_on_tos=False
                )

        return resp

    # -- Public API ----------------------------------------------------------

    @property
    def is_registered(self) -> bool:
        return self._access_token is not None and self._private_key is not None

    @property
    def cached_tos(self) -> str | None:
        """Return locally cached TOS text, or None if not cached."""
        return self._tos_text

    def fetch_tos(self) -> str:
        """Fetch the current TOS from the portal."""
        url = f"{self.portal_url}/tos"
        with self._http() as client:
            resp = client.get(url, headers={"Accept": "text/plain"})
            resp.raise_for_status()
        tos_text = resp.text
        self._save_tos(tos_text)
        return tos_text

    def register(self) -> dict[str, Any]:
        """Run the full welcome-mat registration flow.

        1. Ensure keypair exists
        2. Fetch TOS
        3. Sign TOS
        4. Create access token
        5. POST /api/signup
        """
        self._ensure_keypair()

        tos_text = self.fetch_tos()
        tos_signature = self._sign_tos(tos_text)
        access_token = self._create_access_token(tos_text)

        url = f"{self.portal_url}/api/signup"
        dpop_proof = self._create_dpop_proof("POST", url)

        body = {
            "tos_signature": tos_signature,
            "access_token": access_token,
            "handle": self.handle,
        }

        with self._http() as client:
            resp = client.post(
                url,
                headers={"DPoP": dpop_proof, "Content-Type": "application/json"},
                json=body,
            )

        if resp.status_code == 409:
            # Handle already taken — append random suffix
            import random
            import string

            suffix = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=4)
            )
            self._handle = f"{self.handle}-{suffix}"
            return self.register()

        resp.raise_for_status()
        data = resp.json()

        self._access_token = data["access_token"]
        self._handle = data.get("handle", self._handle)
        self._save_token()

        logger.info("Registered with support portal as %s", self._handle)
        return data

    def ensure_registered(self) -> None:
        """Register if not already registered."""
        if not self.is_registered:
            self.register()

    # -- Tickets -------------------------------------------------------------

    def create_ticket(
        self,
        *,
        product: str = "solstone",
        subject: str,
        description: str,
        severity: str = "medium",
        category: str | None = None,
        user_email: str | None = None,
        user_context: dict | str | None = None,
    ) -> dict[str, Any]:
        """Create a support ticket."""
        self.ensure_registered()
        body: dict[str, Any] = {
            "product": product,
            "subject": subject,
            "description": description,
            "severity": severity,
        }
        if category:
            body["category"] = category
        if user_email:
            body["user_email"] = user_email
        if user_context:
            body["user_context"] = (
                json.dumps(user_context)
                if isinstance(user_context, dict)
                else user_context
            )

        resp = self._authed_request("POST", "/api/tickets", json_body=body)
        resp.raise_for_status()
        return resp.json()

    def list_tickets(
        self,
        *,
        status: str | None = None,
        product: str | None = None,
        severity: str | None = None,
    ) -> list[dict[str, Any]]:
        """List tickets (own tickets for user accounts)."""
        self.ensure_registered()
        params: dict[str, str] = {}
        if status:
            params["status"] = status
        if product:
            params["product"] = product
        if severity:
            params["severity"] = severity

        resp = self._authed_request("GET", "/api/tickets", params=params)
        resp.raise_for_status()
        return resp.json()

    def get_ticket(self, ticket_id: int) -> dict[str, Any]:
        """Get a single ticket with message thread."""
        self.ensure_registered()
        resp = self._authed_request("GET", f"/api/tickets/{ticket_id}")
        resp.raise_for_status()
        return resp.json()

    def reply_to_ticket(self, ticket_id: int, content: str) -> dict[str, Any]:
        """Add a message to a ticket."""
        self.ensure_registered()
        resp = self._authed_request(
            "POST", f"/api/tickets/{ticket_id}/messages", json_body={"content": content}
        )
        resp.raise_for_status()
        return resp.json()

    # -- Knowledge Base ------------------------------------------------------

    def search_articles(self, query: str | None = None) -> list[dict[str, Any]]:
        """Search published KB articles."""
        self.ensure_registered()
        params: dict[str, str] = {}
        if query:
            params["q"] = query

        resp = self._authed_request("GET", "/api/articles", params=params)
        resp.raise_for_status()
        return resp.json()

    def get_article(self, slug: str) -> dict[str, Any]:
        """Read a single KB article."""
        self.ensure_registered()
        resp = self._authed_request("GET", f"/api/articles/{slug}")
        resp.raise_for_status()
        return resp.json()

    # -- Announcements -------------------------------------------------------

    def list_announcements(self) -> list[dict[str, Any]]:
        """List active announcements."""
        self.ensure_registered()
        resp = self._authed_request("GET", "/api/announcements")
        resp.raise_for_status()
        return resp.json()

    # -- Health --------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Check portal health (no auth needed)."""
        with self._http() as client:
            resp = client.get(f"{self.portal_url}/api/health")
            resp.raise_for_status()
        return resp.json()


# -- Module-level convenience ------------------------------------------------


def get_client(
    portal_url: str | None = None,
    anonymous: bool = False,
) -> PortalClient:
    """Get a portal client using journal settings for configuration.

    Reads ``support.portal_url`` from journal config if *portal_url* is None.
    """
    if portal_url is None:
        portal_url = _get_portal_url_from_settings()
    return PortalClient(portal_url=portal_url, anonymous=anonymous)


def _get_portal_url_from_settings() -> str:
    """Read portal URL from journal config, falling back to default."""
    try:
        from think.utils import get_journal

        config_path = Path(get_journal()) / "config" / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            support = config.get("support", {})
            url = support.get("portal_url")
            if url:
                return url
    except Exception:
        pass
    return DEFAULT_PORTAL_URL


def is_enabled() -> bool:
    """Check if the support agent is enabled in settings."""
    try:
        from think.utils import get_journal

        config_path = Path(get_journal()) / "config" / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            support = config.get("support", {})
            return support.get("enabled", True)
    except Exception:
        pass
    return True
