# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import argparse
import sys

from think.callosum import callosum_send
from think.utils import setup_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a notification via callosum")
    parser.add_argument("message", nargs="+", help="notification message text")
    parser.add_argument("--title", help="notification title")
    parser.add_argument("--icon", help="emoji icon")
    parser.add_argument("--event", default="show", help="event name (default: show)")
    parser.add_argument("--action", help="URL path to open on click")
    parser.add_argument("--facet", help="facet context")
    parser.add_argument("--app", help="source app name")
    parser.add_argument("--badge", help="badge text or number")
    parser.add_argument(
        "--auto-dismiss",
        type=int,
        dest="auto_dismiss",
        help="auto-dismiss after N milliseconds",
    )
    parser.add_argument(
        "--no-dismiss",
        action="store_true",
        dest="no_dismiss",
        help="make notification non-dismissible",
    )

    args = setup_cli(parser)

    message = " ".join(args.message)
    kwargs = {"message": message}

    if args.title is not None:
        kwargs["title"] = args.title
    if args.icon is not None:
        kwargs["icon"] = args.icon
    if args.action is not None:
        kwargs["action"] = args.action
    if args.facet is not None:
        kwargs["facet"] = args.facet
    if args.app is not None:
        kwargs["app"] = args.app
    if args.badge is not None:
        kwargs["badge"] = args.badge
    if args.auto_dismiss is not None:
        kwargs["autoDismiss"] = args.auto_dismiss
    if args.no_dismiss:
        kwargs["dismissible"] = False

    ok = callosum_send("notification", args.event, **kwargs)
    if ok:
        print("Notification sent", file=sys.stderr)
        return

    print("Failed to send notification (is callosum running?)", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
