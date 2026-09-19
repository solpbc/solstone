# install solstone on linux

use this guide with the person whose device this is. the platform installer installs the journal, the linux desktop app, or the tmux app from signed releases. `cli` installs the journal tools without setting up a local journal.

## check what is already there

read the existing receipts before choosing a new installation:

- tree installs: `${XDG_DATA_HOME:-$HOME/.local/share}/solstone/install.conf`
- package installs: `/etc/solstone/install.conf`
- journal-only installs: `$HOME/.local/solstone-journal/install-receipt`

keep the existing prefix and route. use `--upgrade` for a platform-owned installation; it selects installed components and does not add new ones. a journal-only installation continues to use [the journal installer](https://solstone.app/install.sh). do not delete receipts or create a second journal to get past an ownership refusal.

## choose together

choose `journal` for a journal on this device, or `cli` for the tools alone. they are alternative roles for the same download. add `desktop` or `tmux` as needed. `all` selects the journal and available apps; `capture` selects the cli and available apps. the signed catalogue determines availability on this architecture.

the default is a user-owned tree under `$HOME/.local`. `--route package` uses apt on debian/ubuntu or dnf on fedora, with sudo for package operations. run as the person who will use solstone so service setup belongs to their account. agents should use `--non-interactive --json`; use `--no-start` when service setup must wait for the person’s session.

## install and verify

save the installer before running it so a download failure cannot look like a successful installation:

```sh
curl -fsSL https://solstone.app/platform-install.sh -o platform-install.sh
sh platform-install.sh --help
```

check that the download succeeded. the help lists options and prerequisites, including minisign. install missing prerequisites with the distribution’s package manager. the journal’s local models also need the OpenMP runtime (`libgomp1` on debian/ubuntu, `libgomp` on fedora). the tmux app needs tmux; package installation resolves its declared dependencies.

preview the agreed selection, then install it:

```sh
sh platform-install.sh --components cli,tmux --dry-run --non-interactive --json
sh platform-install.sh --components cli,tmux --non-interactive --json
```

a preview verifies release metadata and reads ownership. it does not install payloads, acquire installation locks, or change services. inspect refusals before proceeding. signature and digest verification run by default; do not add `--skip-signature` to work around a verification failure.

stdout with `--json` is one JSON document; diagnostics go to stderr. check both the exit status and each component’s status. `target_version` identifies the selected release, including on a failed attempt; it is null for removal. a later failure can leave earlier components successfully installed and recorded.

follow the printed PATH instructions. for a tree installation, the journal tools live at `PREFIX/current/bin/journal`; app launchers live under `PREFIX/bin`. when app PATH configuration is enabled, source `${XDG_CONFIG_HOME:-$HOME/.config}/solstone/env` in the current shell and add that source command to the appropriate shell startup file if needed. `--no-path` leaves this to you.

verify the tools you selected with `journal --version`, `solstone-linux --version`, or `solstone-tmux --version`. for a local journal, run `journal service status`. on systemd hosts, check `systemctl --user status solstone`, `solstone-linux`, or `solstone-tmux` as applicable. installation completion means the selected setup commands succeeded; it is not a claim that intake, pairing, models, or every background service is healthy. the tmux app needs a running tmux session; the desktop app needs a graphical session.

## update or remove

```sh
sh platform-install.sh --upgrade --dry-run --non-interactive --json
sh platform-install.sh --upgrade --non-interactive --json
sh platform-install.sh --components tmux --uninstall --dry-run --json
sh platform-install.sh --components tmux --uninstall --json
```

include the same `--prefix` for a custom tree. repeat `--no-start` and `--no-path` if those policies should continue. removal keeps journal data and does not remove unrelated packages or their dependencies.

## when an operation stops

read stderr and the JSON result before retrying. package receipts retain pending installation or setup state so a matching retry can continue. a failed app tree update attempts to restore the previous active app. if recovery is incomplete, preserve the payload and receipts for inspection. successful earlier components stay installed.

journal setup is delegated to its versioned installer. if native setup fails, or finishes before the platform receipt can be saved, keep the tree and journal data. the error includes a recovery command using the exact versioned bootstrap, its authenticated SHA-256 digest, and the original service/PATH flags. run that command from a writable scratch directory and require its digest check to pass. native recovery does not establish platform ownership; do not fabricate a platform receipt afterward. the public journal-only bootstrap is a separate entry point and may not accept the platform’s role flags.

a killed process can leave files without a complete ownership record. preserve those files and receipts for inspection. an ownership refusal is a reason to investigate, not permission to remove a directory.
