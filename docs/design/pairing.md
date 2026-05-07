# Wave 5 pairing server

## 1. Summary

Wave 5 adds a self-hosted iOS pairing flow to the existing Convey process: one JSON API blueprint at `/api/pairing/*` plus one owner-facing HTML blueprint at `/app/pairing/*`, both defined in `solstone/convey/pairing.py`. This follows the Wave 2 / Wave 3 root-blueprint pattern rather than adding an `solstone/apps/pairing/` package (`solstone/convey/voice.py:27-197`, `solstone/convey/push.py:24-127`, `solstone/convey/__init__.py:110-169`).

The server mints one-time `ptk_...` pairing tokens in memory, accepts ssh-ed25519 public keys from the iOS client, stores paired devices in `journal/config/paired_devices.json`, and returns a one-time `dsk_...` bearer session key whose hash is the only value persisted at rest. The companion iOS contract is fixed: `solstone://pair?token=...&host=...`, `POST {host}/api/pairing/confirm`, then `Authorization: Bearer <session_key>` on subsequent API calls.

The design keeps config defaults compatible with existing journals by using both a new `pairing` block in `solstone/think/journal_default.json` and in-code defaults in `solstone/think/pairing/config.py`, because `get_config()` does not merge defaults once `config/journal.json` exists (`solstone/think/journal_default.json:1-53`, `solstone/think/utils.py:557-588`, `tests/conftest.py:77-84`). It also promotes `cryptography` to a direct dependency: the repo already imports it directly in the link pairing surface, but `pyproject.toml` does not currently declare it (`solstone/apps/link/routes.py:33-35`, `pyproject.toml:32-93`).

## 2. Module layout

| Path | Role |
|---|---|
| `pyproject.toml` | Add direct dependency `cryptography>=42` because pairing will import `cryptography` directly, and the repo already does so in `solstone/apps/link/routes.py` (`solstone/apps/link/routes.py:33-35`, `pyproject.toml:55-61`). |
| `solstone/think/journal_default.json` | Add a flat `pairing` config block beside `voice` and `push`. `identity.name` and `identity.preferred` already exist here and are the source for `owner_identity` resolution (`solstone/think/journal_default.json:2-15`, `35-53`). |
| `solstone/think/pairing/__init__.py` | Narrow re-export surface so callers do not reach into module-private helpers, matching the small package-surface pattern used by `solstone/think/push` and `solstone/think/voice` (`docs/design/push.md:13-24`, `docs/design/voice-server.md:11-26`). |
| `solstone/think/pairing/config.py` | Config readers for `pairing.host_url`, `pairing.token_ttl_seconds`, and owner identity fallback. Mirrors the small-reader style in `solstone/think/push/config.py` and must supply defaults in code because fixture journals already have a `journal.json` (`solstone/think/push/config.py:17-81`, `solstone/think/utils.py:557-588`). |
| `solstone/think/pairing/tokens.py` | In-memory token store with a module-level singleton and `threading.Lock`, modeled after `solstone/think/link/nonces.py`’s single-use TTL store but intentionally kept process-local instead of journal-backed (`solstone/think/link/nonces.py:25-103`). |
| `solstone/think/pairing/keys.py` | Public-key validation, bearer-session-key generation, SHA-256 hashing, and log masking. This is the only module that knows the `ptk_...` / `dsk_...` wire formats and the ssh-ed25519-only rule. |
| `solstone/think/pairing/devices.py` | Sole writer for `journal/config/paired_devices.json`, mirroring the whole-file atomic rewrite pattern from `solstone/think/push/devices.py` (`solstone/think/push/devices.py:21-153`). |
| `solstone/convey/auth.py` | Shared bearer/owner auth helpers. Factors the Bearer extraction pattern out of `solstone/apps/observer/routes.py` / `solstone/apps/import/journal_sources.py` and adds paired-device resolution and owner-auth inspection without redirects (`solstone/apps/observer/routes.py:63-70`, `503-538`, `solstone/apps/import/journal_sources.py:108-128`, `solstone/convey/root.py:49-57`, `81-139`). |
| `solstone/convey/pairing.py` | Defines `pairing_bp = Blueprint("pairing", ..., url_prefix="/api/pairing")` and `pairing_ui_bp = Blueprint("pairing_ui", ..., url_prefix="/app/pairing")`, using the same local `_error`, `_required_json_object`, and `_optional_json_object` request-validation pattern as `solstone/convey/voice.py` / `solstone/convey/push.py` (`solstone/convey/voice.py:27-53`, `solstone/convey/push.py:24-50`). |
| `solstone/convey/__init__.py` | Import and register both pairing blueprints in the same root-blueprint block as `voice_bp` / `push_bp`, before app discovery (`solstone/convey/__init__.py:112-161`). |
| `solstone/convey/root.py` | Extend the exact-name allowlist in `require_login()` with `pairing.confirm_pairing`, `pairing.heartbeat`, `pairing.list_devices`, and `pairing.unpair_device`. Leave `pairing.create_token` and `pairing_ui.index` owner-authed (`solstone/convey/root.py:81-139`). |
| `solstone/convey/templates/pairing.html` | Flat owner-facing desktop page. Use a heading of “Pair a phone” to distinguish this flow from tunnel pairing in `solstone/apps/link/routes.py` (`solstone/apps/link/routes.py:4-24`). |
| `solstone/convey/static/pairing-qr.js` | Vendored `qrcode-generator` browser build, loaded directly because package data currently only includes flat `static/*` assets (`pyproject.toml:110-118`, `solstone/convey/__init__.py:118-133`). |
| `solstone/convey/static/pairing.js` | Page logic: mint token, render QR, countdown, 5-second polling against `GET /api/pairing/devices`, copy-paste fallback, success/error state. |
| `solstone/convey/static/pairing.css` | Minimal page styling only if `app.css` reuse is insufficient. |

### Public API surface

The public function surface is intentionally small and explicit.

#### `solstone/think/pairing/config.py`

- `def get_host_url() -> str:`
 Return the configured pairing host URL, or synthesize `http://localhost:<convey-port>` when `pairing.host_url` is null by reading the recorded Convey port and falling back to the installed default port `5015` (`solstone/think/utils.py:922-935`, `solstone/think/service.py:32-34`).
- `def get_token_ttl_seconds() -> int:`
 Return the configured token TTL, clamped to `60..3600`, defaulting to `600`.
- `def get_owner_identity() -> str:`
 Return `config.identity.preferred`, else `config.identity.name`, else `""` (`solstone/think/journal_default.json:2-15`).

#### `solstone/think/pairing/tokens.py`

- `def create_token(*, ttl_seconds: int | None = None, now: int | None = None) -> PairingToken:`
 Mint a `ptk_...` token, insert it into the in-memory store, and return its issued/expires metadata.
- `def consume_token(token: str, *, now: int | None = None) -> PairingToken | None:`
 Atomically validate and consume a token; return `None` for missing, expired, or already-consumed tokens.
- `def peek_token(token: str, *, now: int | None = None) -> PairingToken | None:`
 Return current token metadata without consuming it while still pruning expired entries.
- `def purge_expired_tokens(*, now: int | None = None) -> int:`
 Remove expired tokens and return the number purged.

#### `solstone/think/pairing/keys.py`

- `def validate_public_key(public_key: str) -> str:`
 Parse and validate an ssh-ed25519 public key, reject any non-Ed25519 algorithm, and return the normalized string.
- `def generate_session_key() -> str:`
 Mint a one-time `dsk_...` bearer session key for the paired device.
- `def hash_session_key(session_key: str) -> str:`
 Return `sha256:<hex>` for storage and lookup.
- `def mask_session_key(session_key: str) -> str:`
 Return a log-safe mask showing only the last four characters and the total length.

#### `solstone/think/pairing/devices.py`

- `def load_devices() -> list[Device]:`
 Load and validate `paired_devices.json`, returning `[]` on missing or malformed stores with a warning.
- `def find_device_by_id(device_id: str) -> Device | None:`
 Return one paired device by `id`, or `None`.
- `def find_device_by_session_key_hash(session_key_hash: str) -> Device | None:`
 Return one paired device matching a stored `session_key_hash`, or `None`.
- `def register_device(*, name: str, platform: str, public_key: str, session_key_hash: str, bundle_id: str, app_version: str, paired_at: str | None = None) -> Device:`
 Create or update a device row keyed by `public_key`, generating a stable `dev_...` id on first registration and rotating the stored `session_key_hash` on re-pair.
- `def touch_last_seen(device_id: str, *, last_seen_at: str | None = None) -> bool:`
 Update `last_seen_at` for an existing paired device.
- `def remove_device(device_id: str) -> bool:`
 Remove one paired device by `id`.
- `def status_view(device: Device) -> dict[str, Any]:`
 Return the non-secret JSON view exposed by `GET /api/pairing/devices`.

#### `solstone/convey/auth.py`

- `def extract_bearer_token() -> str | None:`
 Return the trimmed `Authorization: Bearer ...` token if present, else `None`.
- `def resolve_paired_device() -> Device | None:`
 Hash the presented bearer token, load the matching paired device, and return it when valid.
- `def is_owner_authed() -> bool:`
 Return `True` when the current request already satisfies the owner checks used by `require_login()` without triggering redirects: session cookie, Basic Auth, or the completed-setup localhost bypass (`solstone/convey/root.py:49-57`, `81-139`).
- `def require_paired_device(f):`
 Decorator that resolves a paired-device bearer, stores it on `g.paired_device`, and returns `401` JSON when no valid paired-device bearer is present.

## 3. Flow diagrams

### 3.1 Token mint (`POST /api/pairing/create`)

```text
owner browser on /app/pairing/
 -> POST /api/pairing/create (owner-auth via require_login)
 -> pairing config resolves host URL + TTL
 -> think.pairing.tokens.create_token(...)
 -> response includes token, expires_at, pairing_url, qr_data
 -> pairing.js renders QR client-side and starts countdown
```

This mirrors the local-request-validation shape of `solstone/convey/voice.py` and `solstone/convey/push.py`, but the write target is an in-memory store rather than journal state (`solstone/convey/voice.py:30-53`, `solstone/convey/push.py:27-50`, `solstone/think/link/nonces.py:45-103`).

### 3.2 Confirm (`POST /api/pairing/confirm`)

```text
iOS client
 -> POST /api/pairing/confirm
 -> allowlist bypasses require_login
 -> route validates JSON object + token/public_key/device metadata
 -> think.pairing.tokens.consume_token(token)
 -> think.pairing.keys.validate_public_key(public_key)
 -> think.pairing.keys.generate_session_key() + hash_session_key(...)
 -> think.pairing.devices.register_device(...)
 -> response returns session_key once, plus device_id/journal_root/owner_identity/server_version
```

The confirm flow deliberately follows the existing “token in header/body, then validate against a feature-owned store” pattern from observer ingest and journal-source ingest, but pairing consumes its own in-memory token and persists only the device ledger (`solstone/apps/observer/routes.py:524-538`, `solstone/apps/import/journal_sources.py:108-128`).

### 3.3 List / heartbeat (`GET /api/pairing/devices`, `POST /api/pairing/heartbeat`)

```text
paired device
 -> Authorization: Bearer dsk_...
 -> list: resolve paired device or owner path, return non-secret device rows
 -> heartbeat: @require_paired_device resolves bearer and stores g.paired_device
 -> think.pairing.devices.touch_last_seen(g.paired_device["id"])
```

Heartbeat is bearer-only. List is mixed-auth: allowlist bypass at `require_login()`, then explicit handler-level acceptance of either a paired-device bearer or an already-owner-authenticated request.

### 3.4 Unpair (`DELETE /api/pairing/devices/<device_id>`)

```text
paired device OR owner browser
 -> allowlist bypasses require_login
 -> route resolves either bearer device or owner auth
 -> think.pairing.devices.remove_device(device_id)
 -> 200 {"unpaired": true} on success
```

The storage rule is simple: unpair removes the row from `paired_devices.json` rather than soft-deleting it. That keeps the store authoritative for current pairings only, matching the existing push-device store style (`solstone/think/push/devices.py:76-124`).

### 3.5 Restart semantics

```text
process restart
 -> in-memory token store is empty
 -> pending QR codes immediately stop working
 -> paired_devices.json persists
 -> existing dsk_... bearer tokens continue to resolve
```

This split is deliberate. The token store is ephemeral by scope; the device ledger is durable config.

## 4. Endpoint specs

### 4.1 Endpoint table

| Route | Auth gate | Request shape | Success response | Error cases |
|---|---|---|---|---|
| `POST /api/pairing/create` | `require_login` only | Optional JSON object; no required fields in Wave 5 | `{"token","expires_at","pairing_url","qr_data"}` | `400` invalid JSON / non-object, `500` token mint failure |
| `POST /api/pairing/confirm` | allowlist only | Required JSON object with `token`, `public_key`, `device_name`, `platform`, `bundle_id`, `app_version` | `{"session_key","device_id","journal_root","owner_identity","server_version"}` | `400` bad JSON / missing fields / bad key / unsupported platform, `410` token expired or used |
| `POST /api/pairing/heartbeat` | allowlist + `@require_paired_device` | Optional empty body or JSON object ignored in Wave 5 | `{"ok": true}` | `400` invalid JSON / non-object, `401` missing or invalid bearer |
| `GET /api/pairing/devices` | allowlist + chained | No body | `{"devices":[...]}` | `401` no valid paired-device bearer and not owner-authed |
| `DELETE /api/pairing/devices/<device_id>` | allowlist + chained | No body | `{"unpaired": true}` | `401` no valid paired-device bearer and not owner-authed, `404` unknown `device_id` |
| `GET /app/pairing/` | `require_login` only | No body | `200` HTML page | No route-specific auth bypass; unauthenticated requests redirect via `require_login()` |

### 4.2 Allowlist additions

Add these exact endpoint names to `solstone/convey/root.py`’s `require_login()` allowlist and no others:

- `pairing.confirm_pairing`
- `pairing.heartbeat`
- `pairing.list_devices`
- `pairing.unpair_device`

Do **not** add `pairing.create_token`. Do **not** add `pairing_ui.index` (`solstone/convey/root.py:81-139`).

### 4.3 Mixed-auth chain for `list_devices` and `unpair_device`

The mixed-auth routes intentionally do **not** use `@require_paired_device`, because that decorator is bearer-only and would reject a valid owner request before the owner path could run. Their route-level chain is:

1. `require_login()` is bypassed by endpoint-name allowlist.
2. The handler calls `resolve_paired_device()`.
3. If a device is found, the handler sets `g.paired_device = device` and proceeds.
4. If no device is found, the handler calls `is_owner_authed()`.
5. If neither path succeeds, the handler returns `401 {"error": "...", "reason": "auth_required"}`.

This keeps the bypass explicit, preserves owner access via cookie / Basic Auth / localhost semantics, and avoids redirect responses on API routes (`solstone/convey/root.py:49-57`, `81-139`).

## 5. Config keys

Add this block to `solstone/think/journal_default.json` after `push` and before `retention`:

```json
"pairing": {
 "host_url": null,
 "token_ttl_seconds": 600
}
```

Key rules:

- `pairing.host_url`
 - Default in `journal_default.json`: `null`.
 - Runtime behavior: when null, synthesize `http://localhost:<convey-port>` using `read_service_port("convey")` and fall back to `DEFAULT_SERVICE_PORT = 5015` (`solstone/think/utils.py:922-935`, `solstone/think/service.py:32-34`).
 - Operator behavior: set this explicitly when Convey is exposed through a tunnel, reverse proxy, or non-localhost origin.
- `pairing.token_ttl_seconds`
 - Default: `600`.
 - Runtime clamp: `60..3600`.

`owner_identity` does not need its own config key; it resolves from existing identity fields: `identity.preferred`, then `identity.name`, then `""` (`solstone/think/journal_default.json:2-15`).

## 6. Feature-specific detail

### 6.1 Token store

Decision: use a module-level singleton store backed by `threading.Lock` and a plain dict. Rationale: the scope explicitly accepts restart-invalidated tokens, and the existing link nonce store shows the TTL + single-use semantics we need without implying journal persistence (`solstone/think/link/nonces.py:45-103`).

Token rules:

- Format: `ptk_<urlsafe-base64-32-bytes>`.
- TTL default: `600` seconds.
- TTL clamp: `60..3600`.
- Single-use: `consume_token()` is the only mutating read path.
- Restart invariant: store is empty after process restart; paired devices remain valid.

### 6.2 Session-key crypto

Decision: bearer session keys are `dsk_<urlsafe-base64-32-bytes>` and are stored only as `sha256:<hex>`. Rationale: this matches the scope’s one-time-return contract and keeps the journal ledger non-secret at rest.

### 6.3 Public-key validation

Decision: accept ssh-ed25519 only, with a `2048`-character cap on the incoming public-key string and a `128`-character cap on `device_name`. Rationale: the client contract already generates Ed25519 keys, and tight bounds keep the parser and logs safe.

### 6.4 `paired_devices.json` schema

The on-disk store is a single JSON object:

```json
{
 "devices": [
 {
 "id": "dev_...",
 "name": "jer's iPhone 15 Pro",
 "platform": "ios",
 "public_key": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI...",
 "session_key_hash": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
 "bundle_id": "org.solpbc.solstone-swift",
 "app_version": "0.1.0",
 "paired_at": "2026-04-20T15:31:02Z",
 "last_seen_at": null
 }
 ]
}
```

Store rules:

- Top-level object only.
- Device identity is unique by `public_key`; re-pairing the same phone updates the existing row in place and rotates `session_key_hash`.
- `last_seen_at` starts `null` and is updated only by `POST /api/pairing/heartbeat`.
- Unpair removes the row entirely.

### 6.5 Auth chain

Decision: ship a pairing-focused `solstone/convey/auth.py` now, but do not thread it into existing voice/push/observer routes. Rationale: Wave 5 needs the helper surface immediately, but the scope explicitly defers cross-route enforcement to Wave 5.1.

Helper contract:

- `extract_bearer_token()` mirrors the existing header parsing in observer and import.
- `resolve_paired_device()` hashes the bearer and resolves `paired_devices.json`.
- `require_paired_device` is used only on bearer-only pairing routes in this lode.
- `is_owner_authed()` mirrors `require_login()`’s owner checks without redirects.

### 6.6 Error model

Decision: every API error is JSON shaped as `{"error": "<human message>", "reason": "<stable_reason>"}`. Rationale: this preserves the concise error-helper style from `solstone/convey/voice.py` / `solstone/convey/push.py` while giving the client a stable machine-readable string that does not echo input back (`solstone/convey/voice.py:30-53`, `solstone/convey/push.py:27-50`).

### 6.7 QR generation

Decision: vendor `qrcode-generator` version `1.4.4` under `solstone/convey/static/pairing-qr.js` and generate the SVG client-side from the `qr_data` field returned by `POST /api/pairing/create`. Rationale: the repo packaging only includes flat `static/*` assets, and this keeps the QR dependency browser-only, MIT-licensed, and separate from the page logic in `solstone/convey/static/pairing.js` (`pyproject.toml:110-118`, `solstone/convey/__init__.py:118-133`).

### 6.8 Logging

Decision: use `logging` only; never log raw `session_key`; log masked session keys as last four chars + total length; log public keys only at `DEBUG` with truncation. Rationale: the scope forbids raw secret leakage and pairing is explicitly bearer-token based.

## 7. Domain write-ownership (L1-L9 declarations)

This lode stays within the repo’s layer-hygiene invariants (`scripts/check_layer_hygiene.py:38-110`, `183-240`).

- **L1 Layer boundaries**: `solstone/think/pairing/devices.py` owns the device ledger. `solstone/convey/pairing.py` only validates requests, coordinates feature calls, and returns HTTP responses, matching the root-blueprint split in voice/push (`solstone/convey/voice.py:30-197`, `solstone/convey/push.py:27-127`).
- **L2 Domain write ownership**: `journal/config/paired_devices.json` is owned exclusively by `solstone/think/pairing/devices.py`. No other module writes it. The token store is in memory only, so no extra journal domain is created.
- **L3 Naming contract**: read helpers use read verbs (`load_devices`, `find_device_by_id`, `find_device_by_session_key_hash`); write helpers use write verbs (`register_device`, `touch_last_seen`, `remove_device`).
- **L4 CLI read verbs are read-only**: no new CLI surface is introduced in this lode.
- **L5 Write-verb defaults**: not applicable to CLI; the only mutating surfaces are explicit HTTP write routes.
- **L6 Indexers never mutate source data**: not applicable; no indexer changes.
- **L7 Importers only write to imports/**: not applicable; no importer changes.
- **L8 Hooks have declared outputs**: not applicable; no hook changes.
- **L9 Event handlers are idempotent**: `heartbeat` is idempotent by row update; `confirm` is single-use because the token store enforces consume-once; `create` is side-effectful by construction but scoped to ephemeral memory.

## 8. Tests

### 8.1 `tests/test_pairing_config.py`

Verify config defaults, null-host synthesis, TTL clamp behavior, and owner-identity fallback against fixture journals that already contain `config/journal.json` (`solstone/think/utils.py:557-588`, `tests/conftest.py:77-84`).

### 8.2 `tests/test_pairing_tokens.py`

Verify token shape, TTL metadata, single-use semantics, expiry purge, and restart-local assumptions of the module singleton (`solstone/think/link/nonces.py:45-103`).

### 8.3 `tests/test_pairing_keys.py`

Verify ssh-ed25519 acceptance, rejection of ssh-rsa / ecdsa / malformed keys, session-key shape, SHA-256 hash format, and masking helpers.

### 8.4 `tests/test_pairing_devices.py`

Verify malformed-store recovery, atomic whole-file rewrites, public-key-keyed upsert behavior, `last_seen_at` updates, removal by id, and `status_view()` redaction pattern matching the push-device precedent (`solstone/think/push/devices.py:64-153`).

### 8.5 `tests/test_pairing_auth.py`

Verify `extract_bearer_token()`, `resolve_paired_device()`, `require_paired_device`, and `is_owner_authed()` against cookie, Basic Auth, and `trust_localhost` cases from `solstone/convey/root.py` (`solstone/convey/root.py:49-57`, `81-139`).

### 8.6 `tests/test_pairing_routes.py`

Verify the 5 JSON routes plus the UI route: JSON validation, allowlist expectations, mixed-auth acceptance on list/unpair, confirm success/error shapes, and polling-facing device list output.

### 8.7 `tests/test_pairing_integration.py`

Exercise the full owner-create -> confirm -> bearer-list -> heartbeat -> unpair round trip using `journal_copy`, with no fixture `pairing` stanza added (`tests/conftest.py:77-84`).

## 9. Security considerations

### Token reuse and replay

`consume_token()` is the only mutating read path and it enforces single-use + TTL. Tokens are not written to disk, so process restart invalidates all outstanding QR codes, which the scope explicitly accepts (`solstone/think/link/nonces.py:66-85`).

### Key validation and bounded input

Only ssh-ed25519 public keys are accepted. Rejecting all other SSH algorithms aligns the server with the iOS client contract and keeps the parser surface narrow.

### Log masking

No raw `session_key` appears in logs, error messages, or `paired_devices.json`. Public keys are truncated when logged at `DEBUG`. This is stricter than the existing link pair route, which still echoes CSR parse failures in the JSON error body; pairing should not repeat that pattern (`solstone/apps/link/routes.py:230-235`).

### Restart semantics

Ephemeral pairing tokens disappear on restart; paired-device bearers remain valid because only their hashes are persisted. This split is acceptable at MVP scale and keeps the durable store strictly journal-config scoped.

### Wave 5.1 enforcement gap

This lode ships `solstone/convey/auth.py` and the pairing-only route protections, but it does not yet enforce paired-device auth on existing voice, push, or observer routes. That follow-up is explicit and documented below.

### Naming and UI risk

There is already a separate “pair” concept in the tunnel subsystem (`solstone/apps/link/routes.py:4-24`, `161-256`, `solstone/think/link/README.md:1-20`). To reduce operator confusion, the desktop page heading should read **“Pair a phone”** rather than the more ambiguous **“Pair a device.”** This is a UX pitfall to revisit if operators continue to confuse iOS app pairing with tunnel pairing.

## 10. Live validation

Sandbox smoke-test commands:

```sh
BASE_URL=${BASE_URL:-http://127.0.0.1:5015}
AUTH=${AUTH:-":$SOL_PASSWORD"}
TMPDIR=$(mktemp -d)
KEY_PREFIX="$TMPDIR/pairing-smoke"

ssh-keygen -q -t ed25519 -N '' -f "$KEY_PREFIX" >/dev/null
PUBLIC_KEY=$(tr -d '\n' < "$KEY_PREFIX.pub")

CREATE_JSON=$(curl -u "$AUTH" \
 -H 'Content-Type: application/json' \
 -X POST "$BASE_URL/api/pairing/create" \
 -d '{}')
printf '%s\n' "$CREATE_JSON"

TOKEN=$(CREATE_JSON="$CREATE_JSON" python - <<'PY'
import json, os
print(json.loads(os.environ["CREATE_JSON"])["token"])
PY
)

CONFIRM_JSON=$(curl \
 -H 'Content-Type: application/json' \
 -X POST "$BASE_URL/api/pairing/confirm" \
 -d '{
 "token": "'"$TOKEN"'",
 "public_key": "'"$PUBLIC_KEY"'",
 "device_name": "Pairing Smoke iPhone",
 "platform": "ios",
 "bundle_id": "org.solpbc.solstone-swift",
 "app_version": "0.1.0"
 }' \
 "$BASE_URL/api/pairing/confirm")
printf '%s\n' "$CONFIRM_JSON"

SESSION_KEY=$(CONFIRM_JSON="$CONFIRM_JSON" python - <<'PY'
import json, os
print(json.loads(os.environ["CONFIRM_JSON"])["session_key"])
PY
)
DEVICE_ID=$(CONFIRM_JSON="$CONFIRM_JSON" python - <<'PY'
import json, os
print(json.loads(os.environ["CONFIRM_JSON"])["device_id"])
PY
)

curl -H "Authorization: Bearer $SESSION_KEY" \
 "$BASE_URL/api/pairing/devices"

curl -H "Authorization: Bearer $SESSION_KEY" \
 -H 'Content-Type: application/json' \
 -X POST "$BASE_URL/api/pairing/heartbeat" \
 -d '{}'

curl -u "$AUTH" \
 "$BASE_URL/api/pairing/devices"

curl -H "Authorization: Bearer $SESSION_KEY" \
 -X DELETE "$BASE_URL/api/pairing/devices/$DEVICE_ID"
```

Basic Auth uses only the password component, so `-u ":$SOL_PASSWORD"` is the portable form for owner-auth pairing routes (`solstone/convey/root.py:49-57`, `docs/design/push.md:609-646`).

## 11. Wave 5.1 follow-up

Wave 5.1 will apply `@require_paired_device` to existing iOS-facing routes, but **nothing in those files changes in this lode**.

Routes queued for Wave 5.1:

- `POST /api/voice/session`, `POST /api/voice/connect`, `POST /api/voice/refresh-brain`, `GET /api/voice/nav-hints`, `GET /api/voice/observer-actions`, `GET /api/voice/status` in `solstone/convey/voice.py` (`solstone/convey/voice.py:65-194`).
- `POST /api/push/register`, `DELETE /api/push/register`, `GET /api/push/status`, `POST /api/push/test` in `solstone/convey/push.py` (`solstone/convey/push.py:53-124`).
- `GET /api/voice/observer-actions` remains the immediate observer-actions surface requiring paired-device auth in addition to the pairing routes themselves (`solstone/convey/voice.py:170-176`).

The follow-up will be a targeted auth-enforcement sweep only; the helper surface is shipped in Wave 5 so that sweep can stay mechanical.

## 12. Open questions

- None — ready to implement.

## 13. Sources

- Root blueprint pattern: `solstone/convey/voice.py:27-197`, `solstone/convey/push.py:24-127`, `solstone/convey/__init__.py:110-169`
- Current auth gate and owner auth semantics: `solstone/convey/root.py:30-57`, `81-139`
- Existing Bearer-auth prior art: `solstone/apps/observer/routes.py:63-70`, `503-538`, `solstone/apps/import/journal_sources.py:108-128`
- Durable config/default behavior: `solstone/think/journal_default.json:2-53`, `solstone/think/utils.py:557-588`, `tests/conftest.py:77-84`
- Push-device storage prior art: `solstone/think/push/devices.py:21-153`, `solstone/think/push/config.py:17-81`
- Convey packaging/static constraints: `pyproject.toml:110-118`, `solstone/convey/__init__.py:118-133`
- Link pairing naming collision and prior-art nonce store: `solstone/apps/link/routes.py:4-24`, `74-89`, `161-256`, `solstone/think/link/nonces.py:25-103`, `solstone/think/link/README.md:1-20`, `solstone/think/link/paths.py:74-95`
- Design-doc structure precedent: `docs/design/push.md:1-654`, `docs/design/voice-server.md:1-465`
- Layer-hygiene scope: `scripts/check_layer_hygiene.py:38-110`, `183-240`
