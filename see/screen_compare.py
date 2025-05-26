import numpy as np
import cv2
from math import ceil
from skimage.metrics import structural_similarity as ssim

def compare_images(img1, img2, block_size=50, ssim_threshold=0.90, margin=5):
    # Convert PIL images to numpy arrays and then to LAB. Use L-channel.
    arr1 = np.array(img1.convert("RGB"))
    arr2 = np.array(img2.convert("RGB"))
    lab1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2LAB)[:,:,0]
    lab2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2LAB)[:,:,0]

    height, width = lab1.shape
    grid_rows = ceil(height / block_size)
    grid_cols = ceil(width / block_size)
    changed = [[False]*grid_cols for _ in range(grid_rows)]
    # Compute SSIM for each block.
    for i in range(grid_rows):
        for j in range(grid_cols):
            y0 = i * block_size
            x0 = j * block_size
            y1 = min(y0 + block_size, height)
            x1 = min(x0 + block_size, width)
            block1 = lab1[y0:y1, x0:x1]
            block2 = lab2[y0:y1, x0:x1]
            score, _ = ssim(block1, block2, full=True)
            if score < ssim_threshold:
                changed[i][j] = True

    # Group contiguous changed blocks using DFS.
    groups = []
    visited = [[False]*grid_cols for _ in range(grid_rows)]
    
    # Replace recursive DFS with an iterative version.
    def dfs(i, j, group):
        stack = [(i, j)]
        while stack:
            ci, cj = stack.pop()
            if ci < 0 or ci >= grid_rows or cj < 0 or cj >= grid_cols:
                continue
            if visited[ci][cj] or not changed[ci][cj]:
                continue
            visited[ci][cj] = True
            group.append((ci, cj))
            for ni, nj in [(ci-1, cj), (ci+1, cj), (ci, cj-1), (ci, cj+1)]:
                stack.append((ni, nj))

    for i in range(grid_rows):
        for j in range(grid_cols):
            if changed[i][j] and not visited[i][j]:
                group = []
                dfs(i, j, group)
                groups.append(group)

    boxes = []
    for group in groups:
        # Calculate bounding box in pixel coordinates.
        min_x = width
        min_y = height
        max_x = 0
        max_y = 0
        for (i, j) in group:
            x0 = j * block_size
            y0 = i * block_size
            x1 = min(x0 + block_size, width)
            y1 = min(y0 + block_size, height)
            min_x = min(min_x, x0)
            min_y = min(min_y, y0)
            max_x = max(max_x, x1)
            max_y = max(max_y, y1)
        # Add margin.
        min_x = max(0, min_x - margin)
        min_y = max(0, min_y - margin)
        max_x = min(width, max_x + margin)
        max_y = min(height, max_y + margin)
        # Format as [y_min, x_min, y_max, x_max]
        boxes.append({"box_2d": [min_y, min_x, max_y, max_x]})
    return boxes
