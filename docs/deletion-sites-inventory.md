# Removal Sites Inventory

Inventory of every non-test, non-scratch, non-atomic-tmp destructive removal (`shutil.rmtree`, `Path.unlink`, `os.remove`, `os.unlink`) in production code.

> Atomic-tmp exclusion heuristic:
> same-directory temp sibling created inside the same function for atomic replacement of one target file, promoted via `os.replace`/`rename`, with `unlink` only in the exception cleanup branch. Do not exclude directory deletes, named domain paths, or rollback deletes of non-temp targets.

## Methodology

- Scope: every non-test, non-scratch, non-atomic-tmp destructive removal (`shutil.rmtree`, `Path.unlink`, `os.remove`, `os.unlink`) in production code.
- Grep command: `rg -n 'shutil\.rmtree|\.unlink\(|os\.remove|os\.unlink' --type py`
- Exclusion filter: `tests/`, `scratch/`, `.venv/`, `tmp/`, `observers/`
- Atomic-tmp exclusion heuristic:

  > same-directory temp sibling created inside the same function for atomic replacement of one target file, promoted via `os.replace`/`rename`, with `unlink` only in the exception cleanup branch. Do not exclude directory deletes, named domain paths, or rollback deletes of non-temp targets.

- Reference model: `think/retention.py`
  - scope-narrow docstring at `:4-19`
  - completion check at `:73-115`
  - per-file stream-hashed SHA-256 at `:416-422`
  - dry-run support at `:349-369`, `:427-429`, `:450-451`
  - narrow exception handling at `:378-381`, `:416-429`
  - retention log at `:456-472`
- Write-owner table pointer: `CLAUDE.md` / `AGENTS.md` §7 L2
- Importer convention: importers audit destructive operations via `log_app_action(app='import', ...)` per repo convention (`think/importers/journal_source_cli.py:40, 75, 230, 250`).
- Raw grep noise removed manually: nested app test hit at `apps/observer/tests/test_routes.py:1008` and regex literals in `scripts/check_layer_hygiene.py:54-55`.

## Classification Legend

- `✅` matches the retention reference model closely enough to serve as the template.
- `⚠️` has partial safety coverage or is intentionally out of scope for this sweep.
- `❌` remains a destructive gap after applying the exclusion heuristic.

## think/retention (reference)

| file:line | target | trigger | path validation | audit log | dry-run | class | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `think/retention.py:428` | raw media files in completed segments | retention purge on eligible segments | `is_segment_complete()` plus retention-policy eligibility | `_write_retention_log()` to `health/retention.log` | yes | `✅` | reference template for this sweep |

## think/entities

| file:line | target | trigger | path validation | audit log | dry-run | class | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `think/entities/journal.py:369,375` | `facets/*/entities/<id>/` rel dirs and `entities/<id>/` | `delete_journal_entity()` | entity must exist, must not be principal, and each target must exist as a directory | yes (route: `apps/entities/routes.py:910-918`) | no | `⚠️` | helper itself is unaudited, but the production route is audited; deferred follow-up |
| `think/entities/merge.py:520,536,697,702` | target rel dir overwrite, source rel dir, discovery cache, source entity dir | `merge_entity(..., commit=True)` | source/target entities are loaded and validated up front; delete paths come from the merge plan plus `exists()` checks | yes (`think/entities/merge.py:587-611,706-714`) | yes (`commit=False`) | `⚠️` | audited commit flow, but the broader merge bundle is intentionally deferred |

## think/importers

| file:line | target | trigger | path validation | audit log | dry-run | class | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `think/importers/shared.py:361` | existing `imports/<timestamp>/` directory | `_setup_import(..., force=True)` | fixed `journal/imports/<timestamp>` path and `import_dir.exists()` gate | yes (`think/importers/shared.py:351-357`) | yes | `✅` | fixed in this sweep: per-file manifest is hashed and logged before `rmtree` |
| `think/importers/plaud.py:196` | temporary download file | Plaud download write failure | exact `NamedTemporaryFile` path created in the same function | no | no | `⚠️` | temp download cleanup, not a journal-domain delete |

## think/facets

| file:line | target | trigger | path validation | audit log | dry-run | class | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `think/facets.py:907` | `facets/<name>/` directory | `delete_facet()` | facet path resolves under `journal/facets`, with existing-facet checks before delete | yes (`think/facets.py:899-906`) | no | `⚠️` | audited write-owner delete path; deferred rather than expanded in this lode |

## think/indexer

| file:line | target | trigger | path validation | audit log | dry-run | class | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `think/indexer/journal.py:876` | SQLite index database file | `reset_journal_index()` | fixed `journal/indexer/<db>` path | no | no | `⚠️` | index artifact reset; infrastructure is out of scope for retention-style parity |

## think/identity

| file:line | target | trigger | path validation | audit log | dry-run | class | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `think/identity.py:350` | newly created identity file | rollback on history-append failure in `_write_identity_locked()` | target is scoped to the locked identity dir and only removed on exception after create | no | no | `⚠️` | rollback delete of a just-created file, not a steady-state delete path |

## think/tools

| file:line | target | trigger | path validation | audit log | dry-run | class | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `think/tools/call.py:402` | source facet entity dir | facet merge when destination already has the entity | source/dest facets are validated and source dir must exist | yes (`think/tools/call.py:441-450`) | no | `⚠️` | audited merge flow, but the larger facet-merge bundle is deferred |

## apps/entities

| file:line | target | trigger | path validation | audit log | dry-run | class | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `apps/entities/call.py:179` | source facet entity dir | `sol call entities move --merge` when destination already has the entity | source facet, destination facet, entity resolution, and source dir existence are all checked first | yes (`apps/entities/call.py:184-193`) | no | `⚠️` | audited write-owner CLI path; deferred rather than widened in this sweep |

## apps/speakers

| file:line | target | trigger | path validation | audit log | dry-run | class | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `apps/speakers/routes.py:260` | `entities/<id>/voiceprints.npz` | `api_correct_attribution()` when the removed entry was the NPZ's last entry | entity memory path must resolve, NPZ must exist, and metadata tuple must match before unlink | yes (`apps/speakers/routes.py:981-994`) | no | `✅` | fixed in this sweep: audit payload now records `voiceprints_removed` only on actual unlink |
| `apps/speakers/discovery.py:87` | `awareness/discovery_clusters.json` | discovery starts without an owner centroid | fixed awareness cache path | no | no | `⚠️` | awareness cache invalidation, not journal-domain deletion |
| `apps/speakers/discovery.py:494` | `awareness/discovery_clusters.json` | `identify_unknown_speaker()` completes | fixed awareness cache path | no | no | `⚠️` | cache cleanup after identification; out of scope for this sweep |
| `apps/speakers/owner.py:419,446` | owner-candidate NPZ | owner candidate confirm/reject flows | fixed candidate path under awareness state | no (state update only at `apps/speakers/owner.py:421-428,447-452`) | no | `⚠️` | awareness candidate lifecycle cleanup, not a journal-domain delete |

## apps/transcripts

Out of scope for this sweep; keep visible because it is a destructive journal-domain route owned by a separate transcript bundle.

| file:line | target | trigger | path validation | audit log | dry-run | class | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `apps/transcripts/routes.py:521` | segment directory under `chronicle/<day>/<stream>/` | `DELETE /api/segment/...` | day regex, segment-key validation, existence check, and `commonpath` containment check | yes (`apps/transcripts/routes.py:524-529`) | no | `⚠️` | destructive transcript route owned by a separate bundle; tracked but out of scope here |

## apps/import

| file:line | target | trigger | path validation | audit log | dry-run | class | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `apps/import/routes.py:224,250` | request temp files used for timestamp detection and staged upload copy | import upload request handling | both paths come from `NamedTemporaryFile` in the same request | no | no | `⚠️` | request-scoped temp cleanup, not persisted journal deletion |
| `apps/import/call.py:278,279` | staged config diff files | final config-review resolution | fixed paths under the resolved import-review `state_dir` | yes (`apps/import/call.py:281-290`) | no | `⚠️` | review-state cleanup after explicit operator resolution |
| `apps/import/call.py:401,437,452` | staged entity review file | merge/create/skip entity review resolution | `staged_path` must exist under `state_dir/entities/staged` | yes (`apps/import/call.py:402-463`) | no | `⚠️` | review-state cleanup after explicit operator resolution |
| `apps/import/call.py:507,583,605` | staged facet review file | skip/apply facet review resolution | `staged_path` must exist under `state_dir/facets/staged` | yes (`apps/import/call.py:508-615`) | no | `⚠️` | review-state cleanup after explicit operator resolution |

## apps/settings

| file:line | target | trigger | path validation | audit log | dry-run | class | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `apps/settings/routes.py:878` | canonical `journal/.config/vertex-credentials.json` | provider update clears Vertex credentials | stored path must resolve to the canonical credential path before unlink | yes (`apps/settings/routes.py:892-899`) | no | `⚠️` | config artifact cleanup with a canonical-path guard |
| `apps/settings/call.py:511` | canonical `journal/.config/vertex-credentials.json` | `sol call settings vertex clear` | stored path must resolve to the canonical credential path before unlink | no | no | `⚠️` | CLI config cleanup outside the journal-domain sweep |

## apps/support

| file:line | target | trigger | path validation | audit log | dry-run | class | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `apps/support/routes.py:171` | uploaded attachment temp file | support attachment upload completes or fails | exact temp path created for the request | no | no | `⚠️` | request temp cleanup, not journal-domain deletion |

## observe

| file:line | target | trigger | path validation | audit log | dry-run | class | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `observe/observer_client.py:39` | files inside a draft capture directory | `cleanup_draft()` | iterates only files already inside the draft directory | no | no | `⚠️` | draft temp cleanup on the observe side |
| `observe/sense.py:982` | derived output files for a segment | `delete_outputs()` during reprocess cleanup | delete only when the file matches the requested reprocess type and a corresponding source exists | no (logger only) | yes | `⚠️` | observe-side cleanup has dry-run support but not retention-style logging |
| `observe/transcribe/main.py:546,682` | raw/audio capture files that fail VAD thresholds | transcription filtering | delete is gated by VAD outcome on the source file | no (callosum event only) | no | `⚠️` | observe-side source filtering, not part of this journal-domain sweep |
| `observe/transcribe/revai.py:396` | temporary audio upload file | Rev.ai transcription request teardown | exact temp path plus `exists()` check | no | no | `⚠️` | request temp cleanup |
| `observe/transcribe/whisper.py:231` | temporary audio upload file | Whisper transcription request teardown | exact temp path plus `exists()` check | no | no | `⚠️` | request temp cleanup |

## IPC/health

| file:line | target | trigger | path validation | audit log | dry-run | class | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `think/callosum.py:59,91` | `health/callosum.sock` | callosum server start/stop | fixed socket path and `exists()` checks | no | no | `⚠️` | IPC socket cleanup is out of scope for journal-domain parity |
| `think/supervisor.py:870` | `health/callosum.sock` | supervisor pre-start stale-socket cleanup | fixed `server.socket_path` plus `exists()` check | no | no | `⚠️` | IPC socket race prevention, out of scope |
| `think/heartbeat.py:91,101,138` | heartbeat PID file | stale/corrupt PID cleanup and final teardown | fixed PID path with stale/corrupt guards | no (logger only) | no | `⚠️` | service lifecycle cleanup, not journal-domain deletion |
| `think/service.py:199,215` | installed service plist/unit file | service uninstall | fixed platform-specific install path and `exists()` check | no | no | `⚠️` | installed-service artifact cleanup, out of scope |
| `think/install_guard.py:147,168` | owned `sol` alias symlink | install/uninstall guard | alias ownership is checked before unlink | no | no | `⚠️` | user-bin alias management, not journal-domain deletion |

## maint

| file:line | target | trigger | path validation | audit log | dry-run | class | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `apps/observer/maint/000_migrate_remote_to_observer.py:38` | legacy observer source file | one-shot remote-to-observer migration | migration resolves the legacy source path before delete | no | no | `⚠️` | shipped maint migration; one-shot historical cleanup |
| `apps/settings/maint/002_restructure_stream_dirs.py:122` | legacy segment directory | one-shot stream-dir restructuring migration | delete happens only after migration work on that segment dir | no | no | `⚠️` | shipped maint migration; one-shot historical cleanup |
| `apps/sol/maint/000_migrate_agent_layout.py:46` | legacy agent layout file | one-shot agent-layout migration | migration resolves the legacy source before unlink | no | no | `⚠️` | shipped maint migration; one-shot historical cleanup |
| `apps/sol/maint/001_migrate_agent_run_logs.py:92` | legacy agent run-log file | one-shot run-log migration | delete follows successful migration of that log file | no | no | `⚠️` | shipped maint migration; one-shot historical cleanup |
| `apps/sol/maint/002_migrate_chronicle.py:77,91` | legacy chronicle day dir and legacy SQLite db | one-shot chronicle migration | delete follows successful day/db migration | no | no | `⚠️` | shipped maint migration; one-shot historical cleanup |

## Deferred Follow-ups

- `apps/entities/call.py:179` — audited write-owner move path; defer to a broader entities deletion parity pass.
- `think/facets.py:907` — audited write-owner delete path; not a named gap for this sweep.
- `think/entities/journal.py:369,375` — production route coverage exists, but helper-local parity remains deferred.
- `think/entities/merge.py:520,536,697,702` — audited, commit-gated merge workflow; too broad for this lode.
- `think/tools/call.py:402` — audited facet-merge flow; broader merge semantics make it a defer.
- No `❌` rows remain after B1 and B2 in this sweep.
