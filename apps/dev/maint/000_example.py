# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Example maintenance task demonstrating the maint interface.

This no-op task shows the recommended patterns for maintenance scripts:
- Proper docstring (first line used as description in `sol maint --list`)
- Using setup_cli for consistent argument parsing and logging
- Progress output to stdout (captured in state file)
- Clean exit with appropriate exit code
"""

import argparse
import logging

from think.utils import setup_cli

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    args = setup_cli(parser)

    logger.info("Example maintenance task starting")
    print("This is an example maintenance task.")
    print("Real tasks would process journal data here.")
    print("Use numeric prefixes (000_, 001_) to control execution order.")
    logger.info("Example maintenance task complete")

    # Exit code 0 = success, non-zero = failure
    # No explicit sys.exit(0) needed - implicit on clean return


if __name__ == "__main__":
    main()
