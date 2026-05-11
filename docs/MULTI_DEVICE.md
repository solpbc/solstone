# Multi-Device Journals

## One journal, one writer

A solstone journal is a working dataset. Run one solstone instance against a
journal at a time.

This matters because the journal includes append logs, SQLite indexes, per-day
JSON files, raw media, and observations. Those files are not designed for two
running solstone instances to write at once.

Synced storage is fine as a way to move a journal between devices. The safe
rule is simple: let one device finish, then use the other.

## What the conflict detector does

When solstone starts, the supervisor writes a heartbeat file under:

`<journal>/health/sync/<host>.check`

The heartbeat says which device is active, when it last wrote its heartbeat, and
which journal path it sees locally. Because different devices can mount the same
journal at different paths, the detector uses machine identity instead of path
prefixes.

Startup includes a 20-second probe. If another active heartbeat appears, solstone
refuses to start. While running, the supervisor refreshes its heartbeat and
checks for other active heartbeats every 15 seconds. If another active writer
appears mid-run, solstone stops so the journal does not get corrupted.

## What to do if you see a conflict

Stop solstone on one device, then start it on the device you want to use.

If the other device is truly gone, wait about 60 seconds and try again. That
gives old heartbeat files time to become stale.

Do not manually delete heartbeat files unless you are certain the other device
is off or no longer using this journal. Deleting a live heartbeat only hides the
warning; it does not make concurrent writes safe.

## Living with synced storage today

Use synced storage as a handoff mechanism, not as active multi-device
coordination.

Practical habits:

- Use one device at a time for a given journal.
- Let sync finish before opening the journal elsewhere.
- Keep raw media, the originals, and observations together with the journal.
- If you need to move devices often, stop solstone first, wait for storage sync
  to settle, then start solstone on the next device.

Genuine multi-device coordination is not built yet. This detector only prevents
known unsafe writes.

A first-class multi-device mode is on the long-term plan.
