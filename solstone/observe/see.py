# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from pathlib import Path

from PIL import Image


def draw_bounding_box(
    image: Image.Image,
    box_2d: list[int],
    color: str = "red",
    width: int = 3,
) -> None:
    """Draw bounding box on image (mutates in place).

    Args:
        image: PIL image to draw on (modified in place)
        box_2d: [y_min, x_min, y_max, x_max] coordinates
        color: Box color (default "red")
        width: Line width in pixels (default 3)
    """
    from PIL import ImageDraw

    y_min, x_min, y_max, x_max = box_2d
    draw = ImageDraw.Draw(image)
    for i in range(width):
        draw.rectangle(
            [x_min - i, y_min - i, x_max + i, y_max + i],
            outline=color,
        )


def image_to_jpeg_bytes(image: Image.Image, quality: int = 85) -> bytes:
    """Convert PIL image to JPEG bytes.

    Args:
        image: PIL image
        quality: JPEG quality 1-100 (default 85)

    Returns:
        JPEG encoded bytes
    """
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def decode_frames(
    video_path: str | Path,
    frames: list[dict],
    annotate_boxes: bool = True,
) -> list[Image.Image | None]:
    """
    Decode video frames with optional bounding box annotation.

    Takes frame metadata from a screen.jsonl file and returns corresponding
    PIL images with change regions annotated.

    Args:
        video_path: Path to the raw video file
        frames: List of frame dicts from screen.jsonl, each containing:
            - frame_id (int): 1-based sequential frame number from video
            - box_2d (list[int], optional): [y_min, x_min, y_max, x_max] change region
            Additional fields (timestamp, etc.) are preserved but not used
        annotate_boxes: Draw red borders around box_2d regions (default True)

    Returns:
        List of PIL Images in same order as input frames.
        None for frames that couldn't be matched/decoded.

    Raises:
        ValueError: If frames are missing frame_id field

    Example:
        >>> from solstone.observe.utils import load_analysis_frames
        >>> from solstone.observe.see import decode_frames
        >>> all_frames = load_analysis_frames("20250101/092152/center_DP-3_screen.jsonl")
        >>> # Filter to actual frames (skip header)
        >>> frames = [f for f in all_frames if "frame_id" in f]
        >>> # Get first 10 frames
        >>> images = decode_frames("20250101/092152/center_DP-3_screen.webm", frames[:10])
        >>> images[0].show()  # Display first frame
    """
    if not frames:
        return []

    # Validate frames have frame_id field
    for frame in frames:
        if frame.get("frame_id") is None:
            raise ValueError("All frames must have 'frame_id' field")

    import av

    # Build a map of zero-based frame index -> (index, frame_dict) for lookup
    # JSONL frame_id values are 1-based, so convert to zero-based for decoding.
    frame_map = {}
    for i, frame in enumerate(frames):
        frame_id = frame.get("frame_id")
        if frame_id is None:
            continue
        frame_index = frame_id - 1 if frame_id > 0 else frame_id
        frame_map[frame_index] = (i, frame)

    # Initialize result list with Nones
    results: list[Image.Image | None] = [None] * len(frames)

    # Open video and decode requested frames
    video_path = Path(video_path)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]

        frame_count = 0
        for av_frame in container.decode(stream):
            if av_frame.pts is None:
                continue

            # Check if this frame_id is requested
            if frame_count in frame_map:
                idx, frame_dict = frame_map[frame_count]

                # Convert to PIL Image
                arr = av_frame.to_ndarray(format="rgb24")
                img = Image.fromarray(arr)

                # Draw bounding box if requested and present
                if annotate_boxes and "box_2d" in frame_dict:
                    box_2d = frame_dict["box_2d"]
                    draw_bounding_box(img, box_2d)

                results[idx] = img

            frame_count += 1

            # Early exit if we've processed all requested frames
            if all(r is not None for r in results):
                break

    return results


__all__ = [
    "draw_bounding_box",
    "image_to_jpeg_bytes",
    "decode_frames",
]
