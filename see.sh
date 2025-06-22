#!/bin/bash

# Run screen_watch and describe.py in parallel with restart loops

if [ $# -lt 2 ]; then
  echo "Usage: $0 <wait_seconds> <output_dir> [screen_watch args...]" >&2
  exit 1
fi

WAIT=$1
shift
OUT_DIR=$1
shift

run_watch() {
  while true; do
    start_ts=$(date +%Y%m%d_%H%M%S)
    echo "Starting screen_watch.py at $start_ts"
    python3 "$(dirname "$0")/see/screen_watch.py" "$OUT_DIR" "$@"
    echo "screen_watch.py exited, restarting in $WAIT seconds..."
    sleep "$WAIT"
  done
}

run_describe() {
  while true; do
    start_ts=$(date +%Y%m%d_%H%M%S)
    echo "Starting describe.py at $start_ts"
    python3 "$(dirname "$0")/see/describe.py" "$OUT_DIR"
    echo "describe.py exited, restarting in 1 second..."
    sleep 1
  done
}

run_watch "$@" &
WATCH_PID=$!

run_describe &
DESCRIBE_PID=$!

cleanup() {
  echo "Stopping processes..."
  kill $WATCH_PID $DESCRIBE_PID 2>/dev/null
  wait
  exit 0
}

trap cleanup SIGINT SIGTERM

echo "Started screen_watch.py (PID: $WATCH_PID) and describe.py (PID: $DESCRIBE_PID)"
echo "Press Ctrl+C to stop both processes"

wait

