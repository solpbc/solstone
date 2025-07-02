#!/bin/bash
# Thin wrapper to launch the Python runner for capture and transcribe
exec python3 -m hear.runner "$@"
