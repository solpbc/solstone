#!/bin/bash

# Run capture.py and transcribe.py in parallel with restart loops

mkdir -p logs

# Function to run capture.py in a restart loop
run_capture() {
  while true; do
    start_ts=$(date +%Y%m%d_%H%M%S)
    echo "Starting capture.py at $start_ts"
    python3 "$(dirname "$0")/hear/capture.py" "$@"
    echo "capture.py exited, restarting in 1 second..."
    sleep 1
  done
}

# Function to run transcribe.py in a restart loop
run_transcribe() {
  # Extract the journal directory from arguments (first positional arg)
  journal_dir="${1:-$(pwd)}"
  while true; do
    start_ts=$(date +%Y%m%d_%H%M%S)
    echo "Starting transcribe.py at $start_ts"
    python3 "$(dirname "$0")/hear/transcribe.py" "$journal_dir"
    echo "transcribe.py exited, restarting in 1 second..."
    sleep 1
  done
}

# Start both processes in background
run_capture "$@" &
CAPTURE_PID=$!

run_transcribe "$@" &
TRANSCRIBE_PID=$!

# Function to cleanup background processes
cleanup() {
  echo "Stopping processes..."
  kill $CAPTURE_PID $TRANSCRIBE_PID 2>/dev/null
  wait
  exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "Started capture.py (PID: $CAPTURE_PID) and transcribe.py (PID: $TRANSCRIBE_PID)"
echo "Press Ctrl+C to stop both processes"

# Wait for both processes
wait

