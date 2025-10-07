#!/usr/bin/env python3
"""
screencast_diff.py — find the most visually different frames in a screencast

Two methods available:

1. packet-size (default, fast): Reads compressed packet sizes without decoding.
   Larger packets indicate more complex/detailed frames with more visual change.

2. visual-diff (slower, more accurate): Computes the minimum distance from each
   frame to all earlier frames using intensity, gradient, and histogram-based
   scoring. Frames with high min-distance are novel (never seen before), while
   repeated frames (e.g., A→B→A toggles) score low.

Both methods identify the most divergent frames and extract them as WebP (default: top 10).

Usage:
  gnome-screencast-diff screencast.webm
  gnome-screencast-diff screencast.webm --method visual-diff
  gnome-screencast-diff screencast.webm --interval 0.5  # sample every 0.5s
  gnome-screencast-diff screencast.webm --count 20  # extract top 20 frames
"""

import argparse
import io
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.distance import jensenshannon

from observe.utils import compare_frames, get_frames

# Frame comparison helper functions


def to_luma(img: Image.Image) -> np.ndarray:
    """Convert PIL image to luma (grayscale) as float32 normalized to [0, 1]."""
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def avg_pool(Y: np.ndarray, pool_h: int, pool_w: int) -> np.ndarray:
    """Average pooling via reshape and mean."""
    h, w = Y.shape
    new_h = h // pool_h
    new_w = w // pool_w

    # Trim to make dimensions divisible
    Y_trimmed = Y[: new_h * pool_h, : new_w * pool_w]

    # Reshape and compute mean
    reshaped = Y_trimmed.reshape(new_h, pool_h, new_w, pool_w)
    return reshaped.mean(axis=(1, 3))


def gradient_mag(Y: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using np.diff in x and y directions."""
    dx = np.diff(Y, axis=1, prepend=Y[:, :1])
    dy = np.diff(Y, axis=0, prepend=Y[:1, :])
    return np.sqrt(dx**2 + dy**2)


def hist32(Y: np.ndarray) -> np.ndarray:
    """Compute 32-bin normalized histogram of image."""
    hist, _ = np.histogram(Y, bins=32, range=(0, 1))
    hist = hist.astype(np.float32)
    # Normalize to probability distribution
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum
    return hist


def js_div(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability distributions."""
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(jensenshannon(p, q) ** 2)


def compute_frame_score(
    Y_prev: np.ndarray,
    Y_curr: np.ndarray,
    G_prev: np.ndarray,
    G_curr: np.ndarray,
    hist_prev: np.ndarray,
    hist_curr: np.ndarray,
) -> float:
    """
    Compute comprehensive frame difference score.

    S = 0.5*S_int + 0.3*S_grad + 0.2*JSD
    where:
        S_int  = mean((Y_curr - Y_prev)**2) / (var(Y_prev) + eps)
        S_grad = mean((G_curr - G_prev)**2) / (mean(G_prev**2) + eps)
        JSD    = jensen_shannon_divergence(hist_prev, hist_curr)
    """
    eps = 1e-10

    # Intensity difference score
    var_prev = np.var(Y_prev)
    S_int = np.mean((Y_curr - Y_prev) ** 2) / (var_prev + eps)

    # Gradient difference score
    mean_G_prev_sq = np.mean(G_prev**2)
    S_grad = np.mean((G_curr - G_prev) ** 2) / (mean_G_prev_sq + eps)

    # Histogram JSD
    JSD = js_div(hist_prev, hist_curr)

    # Weighted combination
    S = 0.5 * S_int + 0.3 * S_grad + 0.2 * JSD

    return float(S)


class ScreencastDiffer:
    def __init__(
        self,
        video_path: str,
        sample_interval: float = 1.0,
        method: str = "packet-size",
        count: int = 10,
    ):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.sample_interval = sample_interval
        self.method = method
        self.count = count
        self.frame_scores = []  # List of (timestamp, score) - computed during scan
        self.divergence_scores = []  # List of (timestamp, score) - sorted by score
        self.top_frames = (
            {}
        )  # Dict of {idx: (timestamp, score, webp_bytes, box_stats)} for top N

        # Performance timing
        self.timings = {
            "video_scan": 0.0,
            "frame_to_image": 0.0,
            "luma_compute": 0.0,
            "pooling": 0.0,
            "gradient": 0.0,
            "histogram": 0.0,
            "score_compute": 0.0,
            "top_frames_extract": 0.0,
            "top_frames_decode": 0.0,
            "top_frames_compare": 0.0,
            "top_frames_draw": 0.0,
            "top_frames_to_pil": 0.0,
            "webp_encode": 0.0,
            "packet_scan": 0.0,
        }

        if method == "packet-size":
            print(
                f"Scanning {video_path} with packet-size method...",
                file=sys.stderr,
            )
            self._process_packets()
        elif method == "visual-diff":
            print(
                f"Scanning {video_path} with min-distance-to-history comparison...",
                file=sys.stderr,
            )
            self._process_video()
        else:
            raise ValueError(f"Unknown method: {method}")

        print("Sorting divergence scores...", file=sys.stderr)
        self._compute_divergence()
        print(f"Extracting top {self.count} frames as WebP...", file=sys.stderr)
        self._extract_top_frames()
        print(
            f"Found {len(self.divergence_scores)} scored frames, ready to serve top {self.count}",
            file=sys.stderr,
        )
        self._print_timings()

    def _process_packets(self):
        """Scan video packets and record compressed frame sizes using utility."""
        try:
            t_scan_start = time.perf_counter()

            # Use utility function to get all frames sorted by packet size
            with av.open(str(self.video_path)) as container:
                self.frame_scores = get_frames(container)

            self.timings["packet_scan"] = time.perf_counter() - t_scan_start

            print(
                f"  Scanned {len(self.frame_scores)} frames using packet-size method",
                file=sys.stderr,
            )

        except Exception as e:
            print(f"ERROR: Failed to process packets: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
            raise

    def _process_video(self):
        """Scan video at intervals and compute min distance to any earlier frame."""
        try:
            t_scan_start = time.perf_counter()
            with av.open(str(self.video_path)) as container:
                stream = container.streams.video[0]
                duration = (
                    float(stream.duration * stream.time_base)
                    if stream.duration
                    else None
                )

                if duration:
                    print(f"Video duration: {duration:.2f}s", file=sys.stderr)

                last_sampled = -self.sample_interval
                frame_count = 0

                # Store all previous frame features for comparison
                previous_frames = []  # List of (Yd, G, hist) tuples

                for frame in container.decode(video=0):
                    if frame.pts is None:
                        continue

                    timestamp = frame.time if frame.time is not None else 0.0

                    # Sample at intervals
                    if timestamp - last_sampled >= self.sample_interval:
                        # Convert frame to image
                        t_img_start = time.perf_counter()
                        img = frame.to_image()
                        self.timings["frame_to_image"] += (
                            time.perf_counter() - t_img_start
                        )

                        # Convert to luma
                        t_luma_start = time.perf_counter()
                        Y = to_luma(img)
                        self.timings["luma_compute"] += (
                            time.perf_counter() - t_luma_start
                        )

                        # Average pool to reduce resolution
                        t_pool_start = time.perf_counter()
                        Yd = avg_pool(Y, 128, 128)
                        self.timings["pooling"] += time.perf_counter() - t_pool_start

                        # Compute gradient magnitude
                        t_grad_start = time.perf_counter()
                        G = gradient_mag(Yd)
                        self.timings["gradient"] += time.perf_counter() - t_grad_start

                        # Compute histogram
                        t_hist_start = time.perf_counter()
                        hist = hist32(Yd)
                        self.timings["histogram"] += time.perf_counter() - t_hist_start

                        # Compute min score against all previous frames
                        if previous_frames:
                            t_score_start = time.perf_counter()
                            min_score = float("inf")
                            for Y_prev, G_prev, hist_prev in previous_frames:
                                score = compute_frame_score(
                                    Y_prev, Yd, G_prev, G, hist_prev, hist
                                )
                                min_score = min(min_score, score)
                            self.timings["score_compute"] += (
                                time.perf_counter() - t_score_start
                            )
                            self.frame_scores.append((timestamp, min_score))

                        # Store current frame features for future comparisons
                        previous_frames.append((Yd, G, hist))

                        last_sampled = timestamp
                        frame_count += 1

                        if frame_count % 10 == 0:
                            print(
                                f"  Scanned {frame_count} frames at {timestamp:.1f}s",
                                file=sys.stderr,
                            )

            self.timings["video_scan"] = time.perf_counter() - t_scan_start

        except Exception as e:
            print(f"ERROR: Failed to process video: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
            raise

    def _compute_divergence(self):
        """Sort frame scores by divergence (highest first)."""
        # Scores are already computed during scan, just need to sort
        self.divergence_scores = sorted(
            self.frame_scores, key=lambda x: x[1], reverse=True
        )

    def _extract_top_frames(self):
        """Extract and encode top N most divergent frames as WebP with bounding boxes."""
        if not self.divergence_scores:
            return

        t_extract_start = time.perf_counter()

        # Get top N frames by divergence score
        top_n_by_score = self.divergence_scores[: self.count]

        # Re-order by timestamp (chronological order)
        top_n_chronological = sorted(top_n_by_score, key=lambda x: x[0])

        # Create mapping from timestamp to (original_rank, score)
        timestamp_to_data = {
            ts: (idx, score) for idx, (ts, score) in enumerate(top_n_by_score, 1)
        }

        # Collect set of timestamps we need
        target_timestamps = {ts for ts, _ in top_n_chronological}

        try:
            # First pass: decode all target frames in order
            t_decode_start = time.perf_counter()
            frames_dict = {}  # timestamp -> av.VideoFrame

            with av.open(str(self.video_path)) as container:
                last_sampled = -self.sample_interval

                for frame in container.decode(video=0):
                    if frame.pts is None:
                        continue

                    timestamp = frame.time if frame.time is not None else 0.0

                    # Sample at intervals and check if this is a top frame
                    if timestamp - last_sampled >= self.sample_interval:
                        if timestamp in target_timestamps:
                            # Store the decoded frame
                            frames_dict[timestamp] = frame

                            # Exit early if we've found all top frames
                            if len(frames_dict) >= self.count:
                                break

                        last_sampled = timestamp

            self.timings["top_frames_decode"] = time.perf_counter() - t_decode_start

            # Second pass: compare consecutive frames and draw bounding boxes
            previous_frame = None

            for i, (timestamp, _) in enumerate(top_n_chronological):
                if timestamp not in frames_dict:
                    continue

                current_frame = frames_dict[timestamp]
                original_rank, score = timestamp_to_data[timestamp]

                # Convert to PIL image
                t_pil_start = time.perf_counter()
                img = current_frame.to_image()
                self.timings["top_frames_to_pil"] += time.perf_counter() - t_pil_start

                # Initialize box statistics
                box_stats = None

                # If not the first frame, compute bounding boxes and draw them
                if previous_frame is not None:
                    # Compare with previous frame to get change regions
                    t_compare_start = time.perf_counter()
                    boxes = compare_frames(previous_frame, current_frame)
                    self.timings["top_frames_compare"] += (
                        time.perf_counter() - t_compare_start
                    )

                    # Calculate box statistics
                    if boxes:
                        img_width, img_height = img.size
                        total_pixels = img_width * img_height
                        total_area = 0
                        largest_area = 0
                        largest_box = None

                        for box_data in boxes:
                            y_min, x_min, y_max, x_max = box_data["box_2d"]
                            area = (x_max - x_min) * (y_max - y_min)
                            total_area += area
                            if area > largest_area:
                                largest_area = area
                                largest_box = box_data

                        # Store statistics
                        box_stats = {
                            "num_boxes": len(boxes),
                            "total_area": total_area,
                            "percent_changed": (total_area / total_pixels) * 100,
                            "largest_box": (
                                largest_box["box_2d"] if largest_box else None
                            ),
                            "largest_area": largest_area,
                        }

                        # Draw red 5px rectangles around changed regions
                        t_draw_start = time.perf_counter()
                        draw = ImageDraw.Draw(img)
                        for box_data in boxes:
                            y_min, x_min, y_max, x_max = box_data["box_2d"]
                            # Draw rectangle with 5px red border
                            for offset in range(5):
                                draw.rectangle(
                                    [
                                        x_min + offset,
                                        y_min + offset,
                                        x_max - offset,
                                        y_max - offset,
                                    ],
                                    outline="red",
                                    width=1,
                                )
                        self.timings["top_frames_draw"] += (
                            time.perf_counter() - t_draw_start
                        )

                # Encode as WebP with quality setting
                t_webp_start = time.perf_counter()
                buf = io.BytesIO()
                img.save(buf, format="WEBP", quality=85)
                webp_bytes = buf.getvalue()
                self.timings["webp_encode"] += time.perf_counter() - t_webp_start

                # Store with original divergence rank as key
                self.top_frames[original_rank] = (
                    timestamp,
                    score,
                    webp_bytes,
                    box_stats,
                )

                # Update previous frame for next iteration
                previous_frame = current_frame

        except Exception as e:
            print(f"ERROR: Failed to extract top frames: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
            raise

        self.timings["top_frames_extract"] = time.perf_counter() - t_extract_start

    def get_top_divergent(self, n: int = None):
        """Get the top N most divergent frames."""
        if n is None:
            n = self.count
        return self.divergence_scores[:n]

    def get_top_chronological(self, n: int = None):
        """Get the top N frames in chronological order with their divergence rank."""
        if n is None:
            n = self.count
        top_by_score = self.divergence_scores[:n]
        # Create rank mapping (1-indexed)
        rank_map = {ts: idx for idx, (ts, _) in enumerate(top_by_score, 1)}
        # Sort by timestamp and add rank
        chronological = sorted(top_by_score, key=lambda x: x[0])
        return [(ts, score, rank_map[ts]) for ts, score in chronological]

    def _print_timings(self):
        """Print performance timing breakdown."""
        total = sum(self.timings.values())

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Performance Breakdown (total: {total:.2f}s)", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        # Sort by time descending
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)

        for name, duration in sorted_timings:
            pct = (duration / total * 100) if total > 0 else 0
            bar_width = int(pct / 2)  # Scale to 50 chars max
            bar = "█" * bar_width
            print(f"{name:20s} {duration:7.2f}s {pct:5.1f}% {bar}", file=sys.stderr)

        print(f"{'='*60}", file=sys.stderr)

        # Calculate derived metrics
        if len(self.frame_scores) > 0:
            per_frame = total / (
                len(self.frame_scores) + 1
            )  # +1 for first frame with no score
            print(f"Frames scanned:   {len(self.frame_scores) + 1}", file=sys.stderr)
            print(f"Frames scored:    {len(self.frame_scores)}", file=sys.stderr)
            print(f"Time per frame:   {per_frame:.3f}s", file=sys.stderr)

            if len(self.frame_scores) > 0:
                per_score = self.timings["score_compute"] / len(self.frame_scores)
                print(f"Time per score:   {per_score*1000:.3f}ms", file=sys.stderr)

        print(f"{'='*60}\n", file=sys.stderr)


def make_handler(differ: ScreencastDiffer):
    """Create request handler with access to differ instance."""

    class RequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            """Suppress default request logging."""
            sys.stderr.write(f"{self.address_string()} - {format % args}\n")

        def do_GET(self):
            if self.path == "/":
                # Serve HTML page with top 10 frames
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()

                html = ["<!DOCTYPE html>"]
                html.append("<html><head>")
                html.append("<title>Screencast Divergence</title>")
                html.append("<style>")
                html.append(
                    "body { font-family: monospace; margin: 20px; background: #1e1e1e; color: #ccc; }"
                )
                html.append("h1 { color: #fff; }")
                html.append(
                    ".frame { margin: 20px 0; padding: 10px; border: 1px solid #444; background: #2e2e2e; }"
                )
                html.append(
                    ".frame img { width: 100%; display: block; margin: 10px 0; }"
                )
                html.append(".info { color: #8cf; }")
                html.append(".rank { color: #f90; font-weight: bold; }")
                html.append("</style>")
                html.append("</head><body>")
                html.append(
                    f"<h1>Top {differ.count} Most Divergent Frames (Chronological Order)</h1>"
                )
                html.append(f"<p>Video: {differ.video_path.name}</p>")

                if differ.method == "packet-size":
                    html.append(
                        "<p>Scoring method: Compressed packet size (larger = more complex frame)</p>"
                    )
                else:
                    html.append(
                        "<p>Scoring method: Min distance to any earlier frame (intensity + gradient + histogram JSD)</p>"
                    )

                html.append(
                    f"<p>Total frames analyzed: {len(differ.frame_scores)} (sampled every {differ.sample_interval}s)</p>"
                )
                html.append(
                    "<p>Frames shown in chronological order with red boxes highlighting changes from previous frame</p>"
                )

                for timestamp, score, rank in differ.get_top_chronological():
                    html.append('<div class="frame">')

                    if differ.method == "packet-size":
                        score_label = f"Packet Size: {int(score):,} bytes"
                    else:
                        score_label = f"Divergence Score: {score:.4f}"

                    # Get box statistics for this frame
                    _, _, _, box_stats = differ.top_frames[rank]

                    html.append(
                        f'<div class="info"><span class="rank">Rank #{rank}</span> - Timestamp: {timestamp:.2f}s - {score_label}</div>'
                    )

                    # Display box statistics if available
                    if box_stats:
                        html.append('<div class="info">')
                        html.append(
                            f'  Changed regions: {box_stats["num_boxes"]} boxes, '
                        )
                        html.append(f'{box_stats["percent_changed"]:.2f}% of screen ')
                        html.append(f'({box_stats["total_area"]:,} pixels)')
                        if box_stats["largest_box"]:
                            y_min, x_min, y_max, x_max = box_stats["largest_box"]
                            w = x_max - x_min
                            h = y_max - y_min
                            html.append(f" | Largest box: {w}x{h}px")
                        html.append("</div>")
                    else:
                        html.append(
                            '<div class="info">First frame (no comparison)</div>'
                        )

                    html.append(
                        f'<img src="/frame/{rank}" alt="Frame at {timestamp:.2f}s">'
                    )
                    html.append("</div>")

                html.append("</body></html>")

                output = "\n".join(html).encode("utf-8")
                self.wfile.write(output)

            elif self.path.startswith("/frame/"):
                # Serve individual frame image
                try:
                    idx = int(self.path.split("/")[-1])
                    if idx in differ.top_frames:
                        _, _, webp_bytes, _ = differ.top_frames[idx]
                        self.send_response(200)
                        self.send_header("Content-Type", "image/webp")
                        self.send_header("Content-Length", str(len(webp_bytes)))
                        self.end_headers()
                        self.wfile.write(webp_bytes)
                    else:
                        self.send_error(404, "Frame not found")
                except (ValueError, IndexError):
                    self.send_error(400, "Invalid frame index")
            else:
                self.send_error(404, "Not found")

    return RequestHandler


def main():
    parser = argparse.ArgumentParser(
        description="Find and display the most visually different frames in a screencast"
    )
    parser.add_argument("video", help="Path to screencast webm file")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sample interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--method",
        choices=["packet-size", "visual-diff"],
        default="packet-size",
        help="Scoring method: packet-size (fast, default) or visual-diff (slower, more accurate)",
    )
    parser.add_argument(
        "--port", type=int, default=9999, help="Server port (default: 9999)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of top divergent frames to extract (default: 10)",
    )
    args = parser.parse_args()

    differ = ScreencastDiffer(
        args.video, sample_interval=args.interval, method=args.method, count=args.count
    )

    print(f"\nServer running at http://0.0.0.0:{args.port}/")
    print("Press Ctrl+C to stop")

    server = HTTPServer(("0.0.0.0", args.port), make_handler(differ))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
