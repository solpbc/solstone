# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""AWS SigV4 S3/R2 destination adapter with strict key confinement and credential isolation."""

from dataclasses import dataclass
import datetime
import hashlib
import hmac
import os
from typing import Any, Optional
import urllib.error
import urllib.parse
import urllib.request

from solstone_platform.destination import (
    CasResult,
    GetResult,
    PutResult,
    ResultStatus,
)
from solstone_platform.redact import redact_sensitive_text
from solstone_platform.refusals import (
    HTTP_3XX,
    HTTP_403,
    HTTP_404,
    HTTP_412,
    HTTP_429,
    HTTP_5XX,
    HTTP_ETAG_MALFORMED,
    HTTP_TIMEOUT,
    HTTP_TRUNCATED_2XX,
    LANE_INVALID,
    Refusal,
    SCHEMA_INVALID,
    UNSAFE_FILENAME,
)


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def get_signature_key(key: str, date_stamp: str, region_name: str, service_name: str) -> bytes:
    """Derive AWS SigV4 signing key."""
    k_date = _sign(("AWS4" + key).encode("utf-8"), date_stamp)
    k_region = hmac.new(k_date, region_name.encode("utf-8"), hashlib.sha256).digest()
    k_service = hmac.new(k_region, service_name.encode("utf-8"), hashlib.sha256).digest()
    return hmac.new(k_service, b"aws4_request", hashlib.sha256).digest()


def compute_sigv4_headers(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
    region: str,
    service: str,
    access_key: str,
    secret_key: str,
    session_token: Optional[str] = None,
    amz_datetime: Optional[str] = None,
) -> dict[str, str]:
    """Compute AWS SigV4 canonical request, string to sign, and Authorization header."""
    parsed = urllib.parse.urlparse(url)
    uri = urllib.parse.quote(parsed.path) if parsed.path else "/"

    # Query string canonicalization
    query_parts = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query_parts.sort(key=lambda x: x[0])
    canonical_querystring = "&".join(f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(v, safe='')}" for k, v in query_parts)

    dt = amz_datetime or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    date_stamp = dt[:8]

    # Lowercase headers dictionary
    req_headers: dict[str, str] = {}
    for k, v in headers.items():
        req_headers[k.lower()] = " ".join(v.split())

    if "host" not in req_headers:
        req_headers["host"] = parsed.netloc
    if "x-amz-date" not in req_headers:
        req_headers["x-amz-date"] = dt

    if session_token:
        req_headers["x-amz-security-token"] = session_token

    payload_hash = hashlib.sha256(body).hexdigest()
    if service == "s3" or "x-amz-content-sha256" in req_headers:
        req_headers["x-amz-content-sha256"] = payload_hash

    # Canonical headers
    sorted_header_names = sorted(req_headers.keys())
    canonical_headers = "".join(f"{name}:{req_headers[name]}\n" for name in sorted_header_names)
    signed_headers = ";".join(sorted_header_names)

    canonical_request = f"{method}\n{uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    canonical_request_hash = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()

    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = f"AWS4-HMAC-SHA256\n{dt}\n{credential_scope}\n{canonical_request_hash}"

    signing_key = get_signature_key(secret_key, date_stamp, region, service)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    auth_header = f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
    req_headers["authorization"] = auth_header
    return req_headers


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Refuse 3xx redirect to prevent leaking Authorization credentials
        raise Refusal(HTTP_3XX, f"HTTP redirect {code} refused to prevent credential leakage")


@dataclass
class R2Config:
    endpoint: str
    bucket: str
    region: str = "auto"
    key_prefix: str = ""
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None

    @classmethod
    def from_env(cls) -> "R2Config":
        return cls(
            endpoint=os.environ.get("SOLSTONE_R2_ENDPOINT", ""),
            bucket=os.environ.get("SOLSTONE_R2_BUCKET", ""),
            region=os.environ.get("SOLSTONE_R2_REGION", "auto"),
            key_prefix=os.environ.get("SOLSTONE_R2_KEY_PREFIX", ""),
            access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            session_token=os.environ.get("AWS_SESSION_TOKEN"),
        )


class R2Destination:
    """R2 / S3 Destination adapter implementing atomic publish semantics."""

    def __init__(self, config: R2Config) -> None:
        if config.key_prefix and config.key_prefix.strip("/"):
            raise Refusal(UNSAFE_FILENAME, f"non-empty key_prefix '{config.key_prefix}' rejected on canonical R2Destination")
        self.config = config
        self.opener = urllib.request.build_opener(NoRedirectHandler)
        self.network_sentinel: list[tuple[str, str, dict[str, str]]] = []

    def _validate_key_confinement(self, key: str) -> None:
        """Enforce strict canonical key confinement."""
        if ".." in key or "//" in key or "\\" in key:
            raise Refusal(UNSAFE_FILENAME, f"key '{key}' contains illegal traversal sequence")

        parts = key.split("/")
        if len(parts) == 3 and parts[0] == "solstone" and parts[2] == "latest":
            lane = parts[1]
            if lane not in ("release", "staging", "dev"):
                raise Refusal(LANE_INVALID, f"invalid lane in key: {lane}")
        elif len(parts) == 4 and parts[0] == "solstone":
            lane = parts[1]
            if lane not in ("release", "staging", "dev"):
                raise Refusal(LANE_INVALID, f"invalid lane in key: {lane}")
            filename = parts[3]
            if not filename or "/" in filename:
                raise Refusal(UNSAFE_FILENAME, f"invalid filename in key: {key}")
        elif len(parts) == 4 and parts[0] == "solstone-journal":
            lane = parts[1]
            if lane not in ("release", "staging", "dev"):
                raise Refusal(LANE_INVALID, f"invalid lane in key: {lane}")
            filename = parts[3]
            if not (filename.startswith("solstone-journal-") and filename.endswith("-install.sh")):
                raise Refusal(UNSAFE_FILENAME, f"invalid solstone-journal bootstrap filename: {filename}")
        else:
            raise Refusal(UNSAFE_FILENAME, f"key '{key}' does not match allowed platform release pattern")

    def _admit_request(self, method: str, key: str, headers: dict[str, str]) -> None:
        """Lowest transport seam admission filter."""
        self.network_sentinel.append((method, key, dict(headers)))
        self._validate_key_confinement(key)

        is_latest = key.endswith("/latest") and key.startswith("solstone/")
        if method == "GET":
            pass
        elif method == "PUT":
            if_none_match = headers.get("If-None-Match") or headers.get("if-none-match")
            if_match = headers.get("If-Match") or headers.get("if-match")

            if not is_latest:
                # Immutable keys: strictly require If-None-Match: * and no If-Match
                if if_none_match != "*" or if_match is not None:
                    raise Refusal(UNSAFE_FILENAME, f"immutable key '{key}' requires exactly If-None-Match: * and no If-Match")
            else:
                # Latest pointer: require exactly If-Match XOR If-None-Match: *
                has_inm = (if_none_match == "*")
                has_im = bool(if_match and if_match.strip())
                if not (has_inm ^ has_im):
                    raise Refusal(UNSAFE_FILENAME, f"latest pointer '{key}' requires exactly If-Match XOR If-None-Match: *")
        else:
            # Reject DELETE, POST, HEAD, LIST, and any other method
            raise Refusal(UNSAFE_FILENAME, f"HTTP method '{method}' is not admitted on R2Destination")


    def _make_url(self, key: str) -> str:
        base = self.config.endpoint.rstrip("/")
        bucket = self.config.bucket
        return f"{base}/{bucket}/{key.lstrip('/')}"

    def _send_request(
        self,
        method: str,
        key: str,
        body: bytes = b"",
        headers: Optional[dict[str, str]] = None,
    ) -> tuple[int, dict[str, str], bytes]:
        req_headers = dict(headers or {})
        self._admit_request(method, key, req_headers)
        url = self._make_url(key)

        signed_headers = compute_sigv4_headers(
            method=method,
            url=url,
            headers=req_headers,
            body=body,
            region=self.config.region,
            service="s3",
            access_key=self.config.access_key_id or "",
            secret_key=self.config.secret_access_key or "",
            session_token=self.config.session_token,
        )

        req = urllib.request.Request(url, data=body if method in ("PUT", "POST") else None, headers=signed_headers, method=method)
        try:
            with self.opener.open(req, timeout=30) as resp:
                status = resp.status
                if 300 <= status < 400:
                    raise Refusal(HTTP_3XX, f"HTTP redirect {status} refused")
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                resp_body = resp.read()
                # Check for truncated body if content-length header given
                if "content-length" in resp_headers:
                    try:
                        expected_len = int(resp_headers["content-length"])
                        if len(resp_body) < expected_len:
                            raise Refusal(HTTP_TRUNCATED_2XX, f"truncated response: received {len(resp_body)} of {expected_len} bytes")
                    except ValueError:
                        pass
                return status, resp_headers, resp_body
        except urllib.error.HTTPError as err:
            if 300 <= err.code < 400:
                raise Refusal(HTTP_3XX, f"HTTP redirect {err.code} refused to prevent credential leakage")
            resp_headers = {k.lower(): v for k, v in err.headers.items()}
            err_body = err.read()
            return err.code, resp_headers, err_body
        except urllib.error.URLError as err:
            raise Refusal(HTTP_TIMEOUT, f"network error connecting to destination: {redact_sensitive_text(str(err))}") from err

    def get(self, key: str) -> GetResult:
        status, headers, body = self._send_request("GET", key)
        ct = headers.get("content-type")
        cc = headers.get("cache-control")
        if status == 200:
            etag = headers.get("etag")
            if not etag or not etag.strip():
                return GetResult(status=ResultStatus.MALFORMED_ETAG, detail="missing or empty ETag in 200 response")
            return GetResult(
                status=ResultStatus.OK,
                body=body,
                etag=etag,
                content_type=ct,
                cache_control=cc,
            )
        if status == 404:
            return GetResult(status=ResultStatus.ABSENT, detail="404 Not Found")
        if status == 403:
            return GetResult(status=ResultStatus.FORBIDDEN, detail="403 Forbidden")
        if status == 429:
            return GetResult(status=ResultStatus.RATE_LIMITED, detail="429 Rate Limited")
        if 300 <= status < 400:
            return GetResult(status=ResultStatus.REDIRECT_REFUSED, detail=f"HTTP {status} redirect")
        if 500 <= status < 600:
            return GetResult(status=ResultStatus.SERVER_ERROR, detail=f"HTTP {status}")
        return GetResult(status=ResultStatus.INDETERMINATE, detail=f"HTTP {status}")

    def put_if_absent(
        self,
        key: str,
        body: bytes,
        content_type: str,
        cache_control: str,
    ) -> PutResult:
        headers = {
            "Content-Type": content_type,
            "Cache-Control": cache_control,
            "If-None-Match": "*",
        }
        status, resp_headers, _ = self._send_request("PUT", key, body=body, headers=headers)
        if status in (200, 201, 204):
            etag = resp_headers.get("etag")
            if not etag or not etag.strip():
                return PutResult(status=ResultStatus.MALFORMED_ETAG, detail="missing or empty ETag in PUT response")
            return PutResult(status=ResultStatus.OK, etag=etag)
        if status == 412:
            return PutResult(status=ResultStatus.PRECONDITION_FAILED, detail="412 object already exists")
        if status == 403:
            return PutResult(status=ResultStatus.FORBIDDEN, detail="403 Forbidden")
        if status == 429:
            return PutResult(status=ResultStatus.RATE_LIMITED, detail="429 Rate Limited")
        if 300 <= status < 400:
            return PutResult(status=ResultStatus.REDIRECT_REFUSED, detail=f"HTTP {status} redirect")
        if 500 <= status < 600:
            return PutResult(status=ResultStatus.SERVER_ERROR, detail=f"HTTP {status}")
        return PutResult(status=ResultStatus.INDETERMINATE, detail=f"HTTP {status}")

    def compare_and_swap(
        self,
        key: str,
        body: bytes,
        expected_etag: str,
        content_type: str,
        cache_control: str,
    ) -> CasResult:
        headers = {
            "Content-Type": content_type,
            "Cache-Control": cache_control,
        }
        if expected_etag in ("", "*", None):
            headers["If-None-Match"] = "*"
        else:
            headers["If-Match"] = expected_etag

        status, resp_headers, _ = self._send_request("PUT", key, body=body, headers=headers)
        if status in (200, 201, 204):
            etag = resp_headers.get("etag")
            if not etag or not entertain_etag(etag):
                return CasResult(status=ResultStatus.MALFORMED_ETAG, detail="missing or empty ETag in CAS response")
            return CasResult(status=ResultStatus.OK, etag=etag)
        if status == 412:
            return CasResult(status=ResultStatus.PRECONDITION_FAILED, detail="412 CAS precondition failed")
        if status == 403:
            return CasResult(status=ResultStatus.FORBIDDEN, detail="403 Forbidden")
        if status == 429:
            return CasResult(status=ResultStatus.RATE_LIMITED, detail="429 Rate Limited")
        if 300 <= status < 400:
            return CasResult(status=ResultStatus.REDIRECT_REFUSED, detail=f"HTTP {status} redirect")
        if 500 <= status < 600:
            return CasResult(status=ResultStatus.SERVER_ERROR, detail=f"HTTP {status}")
        return CasResult(status=ResultStatus.INDETERMINATE, detail=f"HTTP {status}")


def entertain_etag(etag: Optional[str]) -> bool:
    return bool(etag and etag.strip())
