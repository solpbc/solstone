#!/usr/bin/env python3
"""
screencast_viewer.py — minimal web server to view screencast frames

Serves PNG frames from a webm screencast on port 9999:
  /           - First full frame
  /?id=XX     - First frame cropped to monitor XX
  /1          - Full frame at 1 second
  /1?id=XX    - Frame at 1 second cropped to monitor XX

Usage:
  python screencast_viewer.py screencast.webm
  gnome-screencast-viewer screencast.webm
"""

import argparse
import io
import sys
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import av
from PIL import Image


class ScreencastViewer:
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Parse monitor geometries from title metadata
        self.monitors = self._parse_title()

    def _parse_title(self) -> dict:
        """Extract and parse monitor geometries from video title metadata."""
        try:
            with av.open(str(self.video_path)) as container:
                title = container.metadata.get("title", "")
        except Exception as e:
            print(f"WARNING: Failed to read video metadata: {e}", file=sys.stderr)
            title = ""

        # Parse title format: "connector-id:position,x1,y1,x2,y2 ..."
        monitors = {}
        for part in title.split():
            if ":" not in part or "," not in part:
                continue
            connector_id, rest = part.split(":", 1)
            coord_parts = rest.split(",")
            if len(coord_parts) >= 5:
                # Format: position,x1,y1,x2,y2
                position, x1, y1, x2, y2 = coord_parts[:5]
                monitors[connector_id] = {
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "position": position,
                }

        return monitors

    def get_frame(self, timestamp: float = 0.0, monitor_id: str = None) -> bytes:
        """Extract frame at timestamp, optionally cropped to monitor."""
        try:
            with av.open(str(self.video_path)) as container:
                stream = container.streams.video[0]
                tb: Fraction = stream.time_base  # seconds per tick
                target_pts = int(timestamp / tb)  # convert seconds -> PTS units

                # Seek near target (previous keyframe), then decode forward
                container.seek(
                    target_pts, stream=stream, any_frame=False, backward=True
                )

                img = None
                for packet in container.demux(stream):
                    for frame in packet.decode():
                        if frame.pts is None:
                            continue
                        # Use frame.time (float seconds) or calculate from pts
                        frame_time = (
                            frame.time if frame.time is not None else (frame.pts * tb)
                        )
                        if frame_time + 1e-9 >= timestamp:
                            # Convert to PIL Image
                            img = frame.to_image()
                            break
                    if img:
                        break

                if img is None:
                    print(
                        f"WARNING: No frame found at/after timestamp {timestamp}",
                        file=sys.stderr,
                    )
                    raise RuntimeError("No frame at/after the requested timestamp")

        except FileNotFoundError:
            print(f"ERROR: Video file not found: {self.video_path}", file=sys.stderr)
            img = Image.new("RGB", (1, 1), color="red")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:
            print(
                f"ERROR: Failed to extract frame at timestamp {timestamp}: {e}",
                file=sys.stderr,
            )
            import traceback

            traceback.print_exc(file=sys.stderr)
            img = Image.new("RGB", (1, 1), color="red")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        # Crop if monitor_id specified
        if monitor_id:
            if monitor_id not in self.monitors:
                print(
                    f"WARNING: Monitor ID '{monitor_id}' not found. Available: {list(self.monitors.keys())}",
                    file=sys.stderr,
                )
            else:
                try:
                    box = self.monitors[monitor_id]["box"]
                    print(
                        f"Cropping to monitor {monitor_id} box: {box}", file=sys.stderr
                    )
                    img = img.crop(tuple(box))
                except Exception as e:
                    print(
                        f"ERROR: Failed to crop frame for monitor {monitor_id}: {e}",
                        file=sys.stderr,
                    )
                    print(f"  Box: {self.monitors[monitor_id]['box']}", file=sys.stderr)
                    print(f"  Image size: {img.size}", file=sys.stderr)

        # Convert to PNG bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def make_handler(viewer: ScreencastViewer):
    """Create request handler with access to viewer instance."""

    class RequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            """Suppress default request logging."""
            sys.stderr.write(f"{self.address_string()} - {format % args}\n")

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path.strip("/")
            query = parse_qs(parsed.query)

            # Parse timestamp from path (e.g., /1.5 -> 1.5 seconds)
            timestamp = 0.0
            if path:
                try:
                    timestamp = float(path)
                except ValueError:
                    timestamp = 0.0

            # Parse monitor ID from query (e.g., /?id=HDMI-1)
            monitor_id = query.get("id", [None])[0]

            # Get frame
            frame_data = viewer.get_frame(timestamp, monitor_id)

            # Send response
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(frame_data)))
            self.end_headers()
            self.wfile.write(frame_data)

    return RequestHandler


def main():
    parser = argparse.ArgumentParser(
        description="Serve screencast frames as PNG images"
    )
    parser.add_argument("video", help="Path to screencast webm file")
    parser.add_argument(
        "--port", type=int, default=9999, help="Server port (default: 9999)"
    )
    args = parser.parse_args()

    viewer = ScreencastViewer(args.video)

    print(f"Serving {args.video}")
    print(f"Monitors: {list(viewer.monitors.keys())}")
    print(f"\nServer running at http://0.0.0.0:{args.port}/")
    print("Examples:")
    print(f"  http://0.0.0.0:{args.port}/           - First full frame")
    print(f"  http://0.0.0.0:{args.port}/1          - Frame at 1 second")
    if viewer.monitors:
        first_id = list(viewer.monitors.keys())[0]
        print(
            f"  http://0.0.0.0:{args.port}/?id={first_id}  - First frame, monitor {first_id}"
        )
    print("\nPress Ctrl+C to stop")

    server = HTTPServer(("0.0.0.0", args.port), make_handler(viewer))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
