# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI main execution module."""

import sys
from solstone_platform.cli import main

if __name__ == "__main__":
    main(sys.argv[1:])
