# Installer strings

Edit terminal messages in [the English catalog](catalog/source/en-US.ftl). The installer builder inserts them into the shell template, the shared install hint and the embedded parser and runtime diagnostics. `make check-installer-catalog` builds the production script and checks its references and shell syntax.

Keep the message ID when editing its wording. Each `argN` is bound by the message reference in the template. Arguments can carry shell expansions, paths, format specifiers or literal command text. Use each argument once and keep its binding in the template. The catalog accepts single-line Fluent patterns with text, variable placeables and quoted string placeables for significant whitespace. The help heredoc carries the option and example layout.

For a local preview of expanded text, build a separate script with `PYTHONPATH=src:tools python3 tools/build_installer.py --pseudo-locale -o /var/tmp/install-en-XA.sh`. The production builder embeds en-US.

The [translation notes](catalog/metadata.json) describe the operational confirmation prompts and the prerequisite message with a fixed prefix. Keep command examples, answer tokens and protocol fields verbatim. Any legal or license text added to this surface stays verbatim English. The separate journal bootstrap and the programs it launches supply their own output.

## Draft locales

Draft translations live under `catalog/translations/<locale>.ftl` (such as `catalog/translations/es-419.ftl`) with a lockfile at `catalog/source-lock.json` recording the SHA-256 of each message's English source text. Production builds (`--production`) embed only `en-US` and refuse non-English draft locales. For a local preview, pass `--locale es-419` to `tools/build_installer.py`; omitted draft entries fall back to `en-US.ftl`. When editing English prose in `catalog/source/en-US.ftl`, refresh `source-lock.json` without acknowledging `es-419` (`PYTHONPATH=src:tools python3 -m solstone_platform.installer_strings lock`) so edited draft entries stay marked stale until updated.
