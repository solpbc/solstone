#!/bin/bash

# Run screen_watch and describe.py in parallel with restart loops

if [ $# -lt 2 ]; then
  echo "Usage: $0 <interval_seconds> <journal_dir> [args...]" >&2
  exit 1
fi

INTERVAL=$1
shift
JOURNAL_DIR=$1
shift

# designed to run in a loop based on given interval
run_scan() {
  while true; do
    start_ts=$(date +%Y%m%d_%H%M%S)
    python3 "$(dirname "$0")/see/scan.py" "$JOURNAL_DIR" --min 250 "$@"
    sleep "$INTERVAL"
  done
}

run_describe() {
  while true; do
    start_ts=$(date +%Y%m%d_%H%M%S)
    echo "Starting describe.py at $start_ts"
    python3 "$(dirname "$0")/see/describe.py" "$JOURNAL_DIR" "$@"
    echo "describe.py exited, restarting in 1 second..."
    sleep 1
  done
}

run_scan "$@" &
SCAN_PID=$!

run_describe "$@" &
DESCRIBE_PID=$!

cleanup() {
  echo "Stopping processes..."
  kill $SCAN_PID $DESCRIBE_PID 2>/dev/null
  wait
  exit 0
}

trap cleanup SIGINT SIGTERM

echo "Started scan.py (PID: $SCAN_PID) and describe.py (PID: $DESCRIBE_PID)"
echo "Press Ctrl+C to stop both processes"

wait

