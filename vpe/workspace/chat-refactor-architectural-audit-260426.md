# Chat Refactor Architectural Audit 260426

## Structural patterns that produced these bugs

- Parallel terminal-talent branches with asymmetric prompt contracts. The old shape had separate `talent_finished` and `talent_errored` prose in `talent/chat_context.py`; 16f89ba3 tightened the errored path only, leaving finished with weaker dispatch suppression. Current code closes that shape through `STOP_AND_REPORT_CONTRACT` at `talent/chat_context.py:21`, `_terminal_result_details()` at `talent/chat_context.py:216`, `_render_terminal_followup()` at `talent/chat_context.py:227`, and `_append_terminal_trigger_context()` at `talent/chat_context.py:304`.
- Runtime terminal handling was duplicated. Current `_on_cortex_finish()` routes active talent completion through `_handle_talent_terminal_locked()` at `convey/chat.py:360`, and `_on_cortex_error()` routes active talent errors through the same helper at `convey/chat.py:411`. The helper owns event append, trigger construction, raw-use rotation, and retry reset at `convey/chat.py:439`.
- Trigger reconstruction had parallel terminal cases. `_trigger_from_stream_event()` now delegates both terminal events to `_talent_terminal_trigger()` at `convey/chat.py:997` and `convey/chat.py:1005`; the shared trigger builder is `convey/chat.py:1016`.
- SSR and live append had two markdown renderers for the same data. 1242c4e8 introduced markdown rendering; a7c8a560 added hidden SSR `data-markdown` cards but left bootstrap timing sensitive to script order. Current SSR source remains in `apps/chat/_chat_event.html:27` and `apps/chat/_chat_event.html:35`; current renderer consolidation is `apps/chat/workspace.html:174` and `apps/chat/workspace.html:185`.
- Lazy bootstrap failed silently relative to vendor script ordering. `convey/templates/app.html:130` includes the app workspace before `marked` and `DOMPurify` at `convey/templates/app.html:133`; current chat bootstrap defers to `DOMContentLoaded` at `apps/chat/workspace.html:77`, after those parser-blocking vendor scripts load.
- Compatibility funnels accumulated stale behavior. `_normalize_trigger()` now accepts the production `context["trigger"]["type"]` shape only at `talent/chat_context.py:180`; the legacy `target = "exec"` fallback was removed from `_parse_chat_result()` and missing targets now fail at `convey/chat.py:889`.

## What was cleaned up

- Removed `report_back_only` prompt prose and replaced it with one terminal report-back rule in `talent/chat.md:103`.
- Collapsed finished/errored synthetic followups into `_render_terminal_followup()` in `talent/chat_context.py:227`.
- Collapsed trigger-context terminal instructions into `_append_terminal_trigger_context()` in `talent/chat_context.py:304`.
- Consolidated active talent finish/error state transitions through `_handle_talent_terminal_locked()` in `convey/chat.py:439`.
- Consolidated terminal stream-event trigger reconstruction through `_talent_terminal_trigger()` in `convey/chat.py:1016`.
- Removed `/api/chat/stream/<day>`, `/api/chat/result/<use_id>`, `_read_result_state()`, stream-limit constants, and their API baselines in 1d3da2c4.
- Hardened chat result parsing so `talent_request.target` is required by runtime and schema (`convey/chat.py:889`, `talent/chat.schema.json:15`).
- Deferred SSR markdown bootstrap and made initial transcript plus live append call `renderTalentMarkdownInto()` (`apps/chat/workspace.html:77`, `apps/chat/workspace.html:177`, `apps/chat/workspace.html:313`).
- Added browser coverage for SSR markdown rendering in `apps/chat/tests/test_markdown_bootstrap_browser.py:103`.

## What was considered and explicitly deferred

- Moving `marked` and `DOMPurify` into `<head>` was rejected: broader script-order blast radius for a chat-local seam.
- Python-side markdown rendering was rejected: it would create a second renderer in a different language.
- Removing the SSR seam was rejected: `data-markdown="1"` plus hidden-until-render remains load-bearing for no-flash.
- Pinchtab was rejected for Test B because Playwright is already declared in `pyproject.toml:95` and gives a direct DOM assertion.
- External consumers of removed `/api/chat/stream` and `/api/chat/result` were not found by `git grep -E "/api/chat/(stream|result)" -- ':!tests' ':!cpo'`; if downstream non-repo clients exist, that is a product/API decision outside this audit.

## New test patterns introduced

- Test A is the cortex-boundary mocked state-machine cycle in `tests/test_chat_runtime.py:431`: owner message enters `/api/chat`, first chat generate dispatches `exec`, `talent_finished` or `talent_errored` returns through the shared terminal path, the report-back chat context carries the same stop-and-report contract, and the final `talent_request: null` reply produces a `sol_message` with no redispatch. Future chat state-machine regressions should use this shape instead of isolated helper tests.
- Test B is the Playwright + Werkzeug page-level check in `apps/chat/tests/test_markdown_bootstrap_browser.py:103`: copy a journal, append a prior `talent_finished` event on disk, serve `create_app()`, navigate to `/app/chat/<day>`, and assert rendered DOM plus no remaining `data-markdown="1"`. Chromium is intentionally not added to `make ci`; the test is covered by `make test-app APP=chat` when Playwright Chromium is installed.
- Failing-on-main snippet captured against pre-render commit 1d3da2c4:

```text
E                       AssertionError: assert 0 > 0
E                        +  where 0 = count()
E                        +    where count = <Locator frame=<Frame name= url='http://127.0.0.1:38511/app/chat/20990102'> selector='.chat-talent-card-detail--markdown strong'>.count
apps/chat/tests/test_markdown_bootstrap_browser.py:126: AssertionError
```

## Tests changed and why

- `tests/test_chat_context.py:354` used to assert observable asymmetry, including strong dispatch suppression only on errored. It now asserts same suppression contract for finished and errored, with kind-distinct result labels at `tests/test_chat_context.py:411`.
- `apps/chat/tests/test_routes.py:175` used to assert the bootstrap seam as if that proved rendering. It now asserts only SSR raw markdown source and `data-markdown="1"` presence, and explicitly asserts server output is not already rendered HTML at `apps/chat/tests/test_routes.py:215`.
- `tests/test_chat_runtime.py:1391` now asserts missing `talent_request.target` is invalid, matching the schema instead of preserving the removed fallback.

## Findings outside scope (for future work)

- Minor stale test fixture: `tests/test_exec_context.py:189` and `tests/test_exec_context.py:195` still pass `trigger_kind` / `trigger_payload` to `exec_context.pre_process()`, which ignores them. This is not a runtime compatibility path, but Test/Prompt owner should clean it when touching shared context tests.
- The ONNX runtime issue was environment-only in this lode. `git show --name-only 3ac49432..HEAD -- pyproject.toml uv.lock` is empty; `pyproject.toml:106` already defines `parakeet-onnx-cuda`, introduced earlier by 0a930163. No repo cleanup is needed here, but Build/Install owner should review why plain `uv sync` can remove the optional runtime before Makefile import checks.
- No chat docs under `docs/` reference `report_back_only`, terminal trigger asymmetry, or `data-markdown`; doc hygiene grep found no in-scope updates.
