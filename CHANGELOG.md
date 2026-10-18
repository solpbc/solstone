# changelog

## [Unreleased]

### changed

- on linux, the installer opens setup in your browser and shows where to add the solstone app. without a desktop, it prints the setup link and an SSH tunnel command for another computer.
- without a desktop, the SSH command the installer prints now forwards the agent port too, so an agent on the journal's computer can finish connecting from your other computer's browser.
- when the installer stops to ask what to install, its example is now the full `curl` command to run.
