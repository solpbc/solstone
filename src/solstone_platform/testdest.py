# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Hermetic in-memory fixture destination adapter for loopback testing."""

from dataclasses import dataclass
import hashlib
from typing import Callable, Optional

from solstone_platform.destination import (
    CasResult,
    GetResult,
    PutResult,
    ResultStatus,
)
from solstone_platform.pins import MinisignPin, PinSet
from solstone_platform.refusals import Refusal, UNSAFE_FILENAME

FIXTURE_BUILD_TOKEN = object()


@dataclass
class StoredObject:
    body: bytes
    etag: str
    content_type: str
    cache_control: str


@dataclass
class LedgerEntry:
    method: str
    key: str
    content_type: Optional[str] = None
    cache_control: Optional[str] = None


class FixtureDestination:
    """Non-subclassable in-memory destination adapter for test fixtures."""

    def __init_subclass__(cls, **kwargs):
        raise TypeError("FixtureDestination is final and cannot be subclassed")

    def __init__(
        self,
        pinset: Optional[PinSet] = None,
        platform_pin: Optional[MinisignPin] = None,
    ) -> None:
        self._build_token = FIXTURE_BUILD_TOKEN
        self.pinset = pinset
        self.platform_pin = platform_pin
        self._post_capture_hook: Optional[Callable[[], None]] = None
        self.cas_pre_hook: Optional[Callable[[], None]] = None

        self.objects: dict[str, StoredObject] = {}
        self.ledger: list[LedgerEntry] = []
        self.network_sentinel: list[Any] = []

        self.fail_next_get: Optional[ResultStatus] = None
        self.get_overrides: dict[str, GetResult] = {}
        self.fail_next_put: Optional[ResultStatus] = None
        self.fail_next_cas: Optional[ResultStatus] = None
        self.fail_next_put_ok_no_object: bool = False
        self.fail_next_put_wrong_bytes: Optional[bytes] = None
        self.fail_next_put_wrong_content_type: Optional[str] = None
        self.fail_next_put_wrong_cache_control: Optional[str] = None

    def _make_etag(self, body: bytes) -> str:
        return f'"{hashlib.sha256(body).hexdigest()[:16]}"'

    def list_keys(self) -> list[str]:
        """Diagnostic key listing. Platform publisher MUST NOT call this method."""
        return sorted(self.objects.keys())

    def get(self, key: str) -> GetResult:
        self.ledger.append(LedgerEntry(method="get", key=key))
        if key in self.get_overrides:
            return self.get_overrides.pop(key)

        if self.fail_next_get:
            status = self.fail_next_get
            self.fail_next_get = None
            return GetResult(status=status, detail="simulated get failure")

        if key not in self.objects:
            return GetResult(status=ResultStatus.ABSENT, detail="404 Not Found")

        obj = self.objects[key]
        return GetResult(
            status=ResultStatus.OK,
            body=obj.body,
            etag=obj.etag,
            content_type=obj.content_type,
            cache_control=obj.cache_control,
        )

    def put_if_absent(
        self,
        key: str,
        body: bytes,
        content_type: str,
        cache_control: str,
    ) -> PutResult:
        self.ledger.append(LedgerEntry(method="put_if_absent", key=key, content_type=content_type, cache_control=cache_control))
        if self.fail_next_put:
            status = self.fail_next_put
            self.fail_next_put = None
            return PutResult(status=status, detail="simulated put failure")

        if self.fail_next_put_ok_no_object:
            self.fail_next_put_ok_no_object = False
            return PutResult(status=ResultStatus.OK, etag=self._make_etag(body))

        if key in self.objects:
            return PutResult(status=ResultStatus.PRECONDITION_FAILED, detail="object already exists")

        store_body = body
        if self.fail_next_put_wrong_bytes is not None:
            store_body = self.fail_next_put_wrong_bytes
            self.fail_next_put_wrong_bytes = None

        store_ct = content_type
        if self.fail_next_put_wrong_content_type is not None:
            store_ct = self.fail_next_put_wrong_content_type
            self.fail_next_put_wrong_content_type = None

        store_cc = cache_control
        if self.fail_next_put_wrong_cache_control is not None:
            store_cc = self.fail_next_put_wrong_cache_control
            self.fail_next_put_wrong_cache_control = None

        etag = self._make_etag(store_body)
        self.objects[key] = StoredObject(
            body=store_body,
            etag=etag,
            content_type=store_ct,
            cache_control=store_cc,
        )
        return PutResult(status=ResultStatus.OK, etag=etag)

    def compare_and_swap(
        self,
        key: str,
        body: bytes,
        expected_etag: str,
        content_type: str,
        cache_control: str,
    ) -> CasResult:
        self.ledger.append(LedgerEntry(method="compare_and_swap", key=key, content_type=content_type, cache_control=cache_control))
        # CAS is only permitted on latest pointer
        parts = key.split("/")
        if not (len(parts) == 3 and parts[0] == "solstone" and parts[2] == "latest"):
            raise Refusal(UNSAFE_FILENAME, f"compare_and_swap is strictly confined to latest pointer, rejected key: {key}")

        if self.cas_pre_hook:
            hook = self.cas_pre_hook
            self.cas_pre_hook = None
            hook()

        if self.fail_next_cas:
            status = self.fail_next_cas
            self.fail_next_cas = None
            return CasResult(status=status, detail="simulated CAS failure")

        existing = self.objects.get(key)
        if expected_etag in ("", "*", None):
            # Atomic creation: expect absent
            if existing is not None:
                return CasResult(status=ResultStatus.PRECONDITION_FAILED, detail="expected absent, but key exists")
        else:
            if existing is None:
                return CasResult(status=ResultStatus.PRECONDITION_FAILED, detail="expected existing object, but key absent")
            if existing.etag != expected_etag:
                return CasResult(status=ResultStatus.PRECONDITION_FAILED, detail=f"etag mismatch: expected {expected_etag}, current {existing.etag}")

        new_etag = self._make_etag(body)
        self.objects[key] = StoredObject(
            body=body,
            etag=new_etag,
            content_type=content_type,
            cache_control=cache_control,
        )
        return CasResult(status=ResultStatus.OK, etag=new_etag)

