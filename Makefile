# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

.PHONY: all hopper-install build test lint format ci

all: ci

hopper-install:
	@true

build:
	python3 -m compileall -q src tools tests

test:
	PYTHONPATH=src:tools python3 -m unittest discover -s tests -p "test_*.py" -v

lint:
	python3 -m py_compile src/solstone_platform/*.py tools/*.py tests/*.py

format:
	python3 tools/formatcheck.py

ci: build lint format test
	python3 tools/secretscan.py
	@echo "All CI gates passed successfully."
