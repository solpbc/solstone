# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

.PHONY: all agent-setup build build-installer build-installer-dev check-installer-catalog test qualify lint format ci

LOCALE ?= es-419

all: ci

agent-setup:
	@true

build:
	python3 -m compileall -q src tools tests

build-installer:
	PYTHONPATH=src:tools python3 tools/build_installer.py --production -o dist/install.sh

build-installer-dev:
	PYTHONPATH=src:tools python3 tools/build_installer.py --locale $(LOCALE) -o dist/install-$(LOCALE).sh
	sh -n dist/install-$(LOCALE).sh

check-installer-catalog: build-installer
	sh -n dist/install.sh

test:
	PYTHONPATH=src:tools python3 -m unittest discover -s tests -p "test_*.py" -v

qualify:
	PYTHONPATH=src:. python3 qualify/catalogue_revision.py -v

lint:
	python3 -m py_compile src/solstone_platform/*.py tools/*.py tests/*.py qualify/*.py

format:
	python3 tools/formatcheck.py

ci: build lint format test qualify check-installer-catalog
	python3 tools/secretscan.py
	@echo "All CI gates passed successfully."
