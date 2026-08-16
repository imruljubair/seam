"""Verbatim port of the ORIGINAL notebook's seam-carving code (numpy-based),
kept as a correctness baseline for seam_carving/pooling.py's tests. Do not
"fix" this file -- its bugs are intentionally preserved for the regression
test that documents them (see test_seam_carving.py).
"""
import numpy as np
from scipy.ndimage import sobel


def energy_function(image):
    gray = np.mean(image, axis=0)
    dx = sobel(gray, axis=1)
    dy = sobel(gray, axis=0)
    return np.hypot(dx, dy)


def find_vertical_seam(energy):
    rows, cols = energy.shape
    seam = np.zeros(rows, dtype=int)
    cost = energy.copy()
    for i in range(1, rows):
        for j in range(cols):
            min_cost = cost[i - 1, j]
            if j > 0:
                min_cost = min(min_cost, cost[i - 1, j - 1])
            if j < cols - 1:
                min_cost = min(min_cost, cost[i - 1, j + 1])
            cost[i, j] += min_cost
    seam[-1] = np.argmin(cost[-1])
    for i in range(rows - 2, -1, -1):
        prev_x = seam[i + 1]
        offset = np.argmin(cost[i, max(0, prev_x - 1):min(cols, prev_x + 2)])
        seam[i] = max(0, prev_x + offset - 1)  # NOTE: buggy at prev_x == 0, kept intentionally
    return seam


def find_horizontal_seam(energy):
    rows, cols = energy.shape
    seam = np.zeros(cols, dtype=int)
    cost = energy.copy()
    for j in range(1, cols):
        for i in range(rows):
            min_cost = cost[i, j - 1]
            if i > 0:
                min_cost = min(min_cost, cost[i - 1, j - 1])
            if i < rows - 1:
                min_cost = min(min_cost, cost[i + 1, j - 1])
            cost[i, j] += min_cost
    seam[-1] = np.argmin(cost[:, -1])
    for j in range(cols - 2, -1, -1):
        prev_y = seam[j + 1]
        offset = np.argmin(cost[max(0, prev_y - 1):min(rows, prev_y + 2), j])
        seam[j] = max(0, prev_y + offset - 1)  # NOTE: buggy at prev_y == 0, kept intentionally
    return seam


def remove_vertical_seam(image, seam):
    rows, cols = image.shape[1:]
    output = np.zeros((image.shape[0], rows, cols - 1), dtype=image.dtype)
    for i in range(rows):
        j = seam[i]
        output[:, i, :] = np.delete(image[:, i, :], j, axis=1)
    return output


def remove_horizontal_seam(image, seam):
    rows, cols = image.shape[1:]
    output = np.zeros((image.shape[0], rows - 1, cols), dtype=image.dtype)
    for j in range(cols):
        i = seam[j]
        output[:, :, j] = np.delete(image[:, :, j], i, axis=1)
    return output


def seam_carve_pooling(image, target_height, target_width):
    while image.shape[2] > target_width:
        energy = energy_function(image)
        if energy.shape[1] == 0:
            break
        seam = find_vertical_seam(energy)
        image = remove_vertical_seam(image, seam)
    while image.shape[1] > target_height:
        energy = energy_function(image)
        if energy.shape[0] == 0:
            break
        seam = find_horizontal_seam(energy)
        image = remove_horizontal_seam(image, seam)
    return image


def reference_batch(x_np, target_height, target_width):
    """x_np: (B,C,H,W) numpy -> (B,C,target_height,target_width), per-image loop."""
    return np.stack(
        [seam_carve_pooling(x_np[b], target_height, target_width) for b in range(x_np.shape[0])],
        axis=0,
    )


def corrected_find_vertical_seam(energy):
    """Same DP, but with the boundary bug in the backtrack fixed."""
    rows, cols = energy.shape
    seam = np.zeros(rows, dtype=int)
    cost = energy.copy()
    for i in range(1, rows):
        for j in range(cols):
            min_cost = cost[i - 1, j]
            if j > 0:
                min_cost = min(min_cost, cost[i - 1, j - 1])
            if j < cols - 1:
                min_cost = min(min_cost, cost[i - 1, j + 1])
            cost[i, j] += min_cost
    seam[-1] = np.argmin(cost[-1])
    for i in range(rows - 2, -1, -1):
        prev_x = seam[i + 1]
        lo, hi = max(0, prev_x - 1), min(cols, prev_x + 2)
        offset = np.argmin(cost[i, lo:hi])
        seam[i] = lo + offset  # fixed: absolute column, not prev_x + offset - 1
    return seam


def corrected_find_horizontal_seam(energy):
    rows, cols = energy.shape
    seam = np.zeros(cols, dtype=int)
    cost = energy.copy()
    for j in range(1, cols):
        for i in range(rows):
            min_cost = cost[i, j - 1]
            if i > 0:
                min_cost = min(min_cost, cost[i - 1, j - 1])
            if i < rows - 1:
                min_cost = min(min_cost, cost[i + 1, j - 1])
            cost[i, j] += min_cost
    seam[-1] = np.argmin(cost[:, -1])
    for j in range(cols - 2, -1, -1):
        prev_y = seam[j + 1]
        lo, hi = max(0, prev_y - 1), min(rows, prev_y + 2)
        offset = np.argmin(cost[lo:hi, j])
        seam[j] = lo + offset
    return seam


def corrected_seam_carve_pooling(image, target_height, target_width):
    while image.shape[2] > target_width:
        energy = energy_function(image)
        seam = corrected_find_vertical_seam(energy)
        image = remove_vertical_seam(image, seam)
    while image.shape[1] > target_height:
        energy = energy_function(image)
        seam = corrected_find_horizontal_seam(energy)
        image = remove_horizontal_seam(image, seam)
    return image


def corrected_reference_batch(x_np, target_height, target_width):
    return np.stack(
        [corrected_seam_carve_pooling(x_np[b], target_height, target_width)
         for b in range(x_np.shape[0])],
        axis=0,
    )
