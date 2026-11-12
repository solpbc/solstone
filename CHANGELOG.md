# changelog

## [Unreleased]

### changed

- on linux, removing the journal or the command-line tools with `--uninstall` now also takes out the PATH lines the journal's installer added to `~/.profile`, `~/.bashrc` and `~/.zshrc`. everything else in those files stays as it was.
- on linux, the installer now takes over a journal installed by the journal's own installer and updates it where it is from then on. it says so in one line. the journal stays where it is, and its service and PATH lines are kept. if the takeover can't finish, the journal keeps working with its own installer, where it is.
- on linux, a journal or the solstone app you installed from a `.deb` or `.rpm` by hand is now recorded as a package install once every file it installed matches the signed release, and `solstone update` updates it after that. if a file doesn't match, the installer stops and changes nothing.
- on linux, a computer with the 1.0 journal now installs with this installer on either route. setup replaces the old journal's commands and service, and keeps your journal.
- when the solstone app can't start because libraries it needs aren't installed, the installer now names them and how to install them, and once they're installed, running the installer again works.
- in installer revision 20, a new catalogue does not carry solstone for tmux, terminal is part of the solstone app, and upgrade asks before moving an installed solstone for tmux into it.
- on linux, installing the desktop app with this installer now sets it up to open `solstone:` links, so `open in the solstone app` on a pairing link's page opens the window with the link filled in. if you installed it before this, run the installer again with `--components desktop --upgrade`. if the button still goes nowhere, paste the pairing link in the solstone app's journal tab.
- on linux, when it finishes, the installer now says the solstone app's journal tab finds a journal on this computer and connects with one press.
- on linux, the installer now finishes setup with `solstone journal open /init`, which opens your journal with a one-time link, and it prints one setup step instead of two. without a desktop, it says to forward the ports over SSH, then run that command with `--print-url` and open the link on your other computer. an older journal still opens at its address.
- on an arm64 linux computer, the installer can now add the desktop app when the release includes it.
- when the installer asks what to install, it lists the current component names.

- on a linux computer with a desktop, the installer opens setup in your browser and shows where to add the solstone app. without a desktop, it prints how to finish setup and an SSH tunnel command for another computer.
- without a desktop, the SSH command the installer prints now forwards the agent port too, so an agent on the journal's computer can finish connecting from your other computer's browser.
- when the installer stops to ask what to install, its example is now the full `curl` command to run.
- installing the desktop app now adds solstone to your app menu, and removing it takes it out again.
- when it finishes, the installer now says to open solstone from your app menu and paste your pairing link in its journal tab. if the desktop app isn't installed yet, it shows the command that adds it.
- choosing everything on a computer without tmux no longer stops the install. the installer leaves out solstone for tmux and says how to add it later.
