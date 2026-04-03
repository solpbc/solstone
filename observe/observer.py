# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unified observer entry point with platform detection.

Detects the current platform and delegates to the appropriate
platform-specific observer implementation. Currently supports Linux only;
macOS capture is handled by the solstone-macos native companion app.
"""

import sys


def main() -> None:
    """Platform-aware observer entry point.

    Detects the current platform and calls the appropriate observer:
    - Linux: observe.linux.observer
    - macOS: handled by solstone-macos native companion app (not this command)
    """
    platform = sys.platform

    if platform == "linux":
        from observe.linux.observer import main as platform_main
    else:
        print(
            f"Error: Observer not available for platform '{platform}'", file=sys.stderr
        )
        print(
            "Supported platform: Linux. macOS capture is handled by the"
            " solstone-macos native companion app.",
            file=sys.stderr,
        )
        sys.exit(1)

    platform_main()


if __name__ == "__main__":
    main()
