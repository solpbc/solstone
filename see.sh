#!/bin/bash

# Simple wrapper to repeatedly run screen_watch
if [ $# -lt 1 ]; then
  echo "Usage: $0 <wait_seconds> [screen_watch args...]" >&2
  exit 1
fi

WAIT=$1
shift

while true; do
  python3 "$(dirname "$0")/see/screen_watch.py" "$@"
  sleep "$WAIT"
done

