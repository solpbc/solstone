## Decision Log

Request: `req_h5iyql3s`

- D1 — Config key name: `messages`
  The chat pre-hook returns a top-level `messages` key because `_run_talent()` copies non-`template_vars` keys directly onto config, `_apply_template_vars()` does not touch them, and no existing pre-hook key collides with `messages`.

- D2 — Shape: plain dict
  Structured history uses a plain `list[dict[str, str]]` with `role` and `content` because every provider adapter already accepts dict-based message lists or can map them locally without adding shared abstractions.

- D3 — Owner-turn handling
  `owner_message` triggers rely on the current owner turn already being present in the chat tail, while `talent_finished` and `talent_errored` triggers synthesize a final user turn like `[talent <name> finished: <summary>]` or `[talent <name> errored: <reason>]` to keep the model input user-final and explicit.

- D4 — `chat.md` reconciliation
  The flattened `$chat_stream_tail` prompt variable is removed from `talent/chat.md` so structured history lives only in `messages`, and the preview baseline is expected to change because it reflects the raw prompt template.

- D5 — Talent-event markers in structured history
  Mid-tail `talent_spawned`, `talent_finished`, and `talent_errored` events are dropped from the structured message list because they are side-channel metadata that disrupt conversational role alternation; only the current finished/errored trigger is preserved via the synthesized final user turn.

- D6 — Google mapper
  Google generate paths convert structured `{role, content}` dicts into Gemini-native `types.Content` objects, mapping `assistant` to `model`, while leaving `system_instruction` on `GenerateContentConfig` and keeping the legacy string/list-of-strings path unchanged.
