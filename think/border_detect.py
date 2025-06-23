import numpy as np
from PIL import Image


def detect_border(
    im: Image.Image,
    color: tuple[int, int, int],
    *,
    min_length: int = 100,
    border: int = 3,
    tolerance: int = 5,
) -> tuple[int, int, int, int]:
    """Detect a coloured border and return the bounding box coordinates.

    Parameters
    ----------
    im : Image.Image
        Image to analyse.
    color : tuple[int, int, int]
        RGB values of the border colour to detect.
    min_length : int, optional
        Minimum number of matching pixels per side, by default 100.
    border : int, optional
        Expected thickness of the border in pixels, by default 3.
    tolerance : int, optional
        Allowed deviation per channel for colour matching, by default 5.

    Returns
    -------
    tuple[int, int, int, int]
        Bounding box as ``(y_min, x_min, y_max, x_max)``.
    """
    arr = np.asarray(im)
    r, g, b = color
    mask = (
        (np.abs(arr[..., 0] - r) <= tolerance)
        & (np.abs(arr[..., 1] - g) <= tolerance)
        & (np.abs(arr[..., 2] - b) <= tolerance)
    )

    col_hits = mask.sum(0)
    row_hits = mask.sum(1)

    cols = np.where(col_hits >= min_length)[0]
    rows = np.where(row_hits >= min_length)[0]
    if cols.size == 0 or rows.size == 0:
        raise ValueError("No border detected")

    def first_last(groups):
        groups = np.split(groups, np.where(np.diff(groups) != 1)[0] + 1)
        groups = [g for g in groups if g.size >= border]
        if not groups:
            raise ValueError("Border not thick enough")
        return groups[0][0], groups[-1][-1]

    x_min, x_max = first_last(cols)
    y_min, y_max = first_last(rows)

    return int(y_min), int(x_min), int(y_max), int(x_max)


__all__ = ["detect_border"]
