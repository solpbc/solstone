# changelog

## [Unreleased]

### changed

- on a linux computer with a desktop, the installer opens setup in your browser and shows where to add the solstone app. without a desktop, it prints the setup link and an SSH tunnel command for another computer.
- without a desktop, the SSH command the installer prints now forwards the agent port too, so an agent on the journal's computer can finish connecting from your other computer's browser.
- when the installer stops to ask what to install, its example is now the full `curl` command to run.
- installing the desktop app now adds solstone to your app menu, and removing it takes it out again.
- when it finishes, the installer now says to open solstone from your app menu and paste your pairing link in its journal tab. if the desktop app isn't installed yet, it shows the command that adds it.
- choosing everything on a computer without tmux no longer stops the install. the installer leaves out the tmux app and says how to add it later.
