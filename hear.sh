#!/bin/bash

# Run gemini_mic.py in a restart loop, logging output

mkdir -p logs

while true; do
  start_ts=$(date +%Y%m%d_%H%M%S)
  logfile="logs/hear_${start_ts}.log"
  python3 "$(dirname "$0")/hear/gemini_mic.py" "$@" > "$logfile" 2>&1
  sleep 1
done

