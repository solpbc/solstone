# Importing Granola Transcripts

Granola doesn't offer a native export — [muesli](https://github.com/harperreed/muesli) extracts your meeting transcripts to local markdown files.

## Install muesli

```bash
cargo install --git https://github.com/harperreed/muesli.git --all-features
```

Requires [Rust/Cargo](https://rustup.rs/). If you already have Rust installed, this takes about a minute.

## Extract your transcripts

```bash
muesli sync
```

This pulls your Granola meeting transcripts to `~/.local/share/muesli/transcripts/`. First run downloads everything; subsequent runs are incremental (~1 second).

## Point solstone at the output

In the next step, confirm the path to your muesli transcripts folder (auto-detected if found at the default location).

After import, you can enable hourly sync in **Settings > Sync** so new Granola meetings appear in your journal automatically.
