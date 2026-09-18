# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Abstract destination adapter protocol and structured operation results."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Protocol


class ResultStatus(Enum):
    OK = auto()
    ABSENT = auto()               # 404 Not Found
    PRECONDITION_FAILED = auto()  # 412 (put_if_absent target exists, or CAS mismatch)
    FORBIDDEN = auto()            # 403 Forbidden
    RATE_LIMITED = auto()         # 429 Too Many Requests
    SERVER_ERROR = auto()         # 5xx
    TIMEOUT = auto()              # Socket / connection timeout
    TRUNCATED_2XX = auto()        # Incomplete payload
    MALFORMED_ETAG = auto()       # Invalid ETag header
    REDIRECT_REFUSED = auto()     # 3xx redirect refused
    LOST_BEFORE_COMMIT = auto()
    LOST_AFTER_COMMIT = auto()
    INDETERMINATE = auto()        # Ambiguous status requiring authoritative re-read


@dataclass(frozen=True)
class GetResult:
    status: ResultStatus
    body: Optional[bytes] = None
    etag: Optional[str] = None
    content_type: Optional[str] = None
    cache_control: Optional[str] = None
    detail: Optional[str] = None

    def is_ok(self) -> bool:
        return self.status == ResultStatus.OK

    def is_absent(self) -> bool:
        return self.status == ResultStatus.ABSENT


@dataclass(frozen=True)
class PutResult:
    status: ResultStatus
    etag: Optional[str] = None
    detail: Optional[str] = None

    def is_ok(self) -> bool:
        return self.status == ResultStatus.OK

    def is_precondition_failed(self) -> bool:
        return self.status == ResultStatus.PRECONDITION_FAILED


@dataclass(frozen=True)
class CasResult:
    status: ResultStatus
    etag: Optional[str] = None
    detail: Optional[str] = None

    def is_ok(self) -> bool:
        return self.status == ResultStatus.OK

    def is_precondition_failed(self) -> bool:
        return self.status == ResultStatus.PRECONDITION_FAILED


class Destination(Protocol):
    """Destination storage interface for platform release objects."""

    def get(self, key: str) -> GetResult:
        ...

    def put_if_absent(
        self,
        key: str,
        body: bytes,
        content_type: str,
        cache_control: str,
    ) -> PutResult:
        ...

    def compare_and_swap(
        self,
        key: str,
        body: bytes,
        expected_etag: str,
        content_type: str,
        cache_control: str,
    ) -> CasResult:
        ...
