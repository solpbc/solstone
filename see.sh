#!/bin/bash
# Thin wrapper to launch the Python runner for scanning and describing
exec python3 -m see.runner "$@"
