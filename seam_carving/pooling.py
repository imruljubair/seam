"""Vectorized, batched seam-carving pooling for CNNs (PyTorch only, GPU-friendly).

Runs the whole batch through each seam-removal step at once instead of
looping per image with numpy: batched Sobel energy via a single conv2d call,
the DP seam search vectorized across the batch (only the inherently
sequential DP sweep direction stays a Python loop, now O(H) or O(W)
iterations total instead of O(B * H * W)), and seam removal via
boolean-mask + reshape instead of per-row Python loops.

Runs under torch.no_grad(): the seam-selection argmin is not differentiable
regardless of implementation, and in this codebase conv1 (upstream of this
layer) is frozen, so no gradient is needed through it. See models.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
_SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])


def batched_energy(x: torch.Tensor) -> torch.Tensor:
    """x: (B,C,H,W) -> energy: (B,H,W).

    Uses 'replicate' padding to match scipy.ndimage.sobel's default boundary
    mode: for a single-pixel pad, scipy's default (edge-duplicating) mode is
    equivalent to PyTorch's 'replicate', not PyTorch's 'reflect'.
    """
    gray = x.mean(dim=1, keepdim=True)
    gray = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    kx = _SOBEL_X.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    dx = F.conv2d(gray, kx)
    dy = F.conv2d(gray, ky)
    return torch.hypot(dx, dy).squeeze(1)


def find_vertical_seams(energy: torch.Tensor) -> torch.Tensor:
    """energy: (B,H,W) -> seams: (B,H), one seam (column index per row) per image.

    Standard seam-carving DP, vectorized across batch and columns. Unlike a
    common boundary-clipped-backtrack formula (`prev_x + offset - 1`, wrong
    when the search window is clipped at the left/top edge), this always
    reconstructs the true minimum-cost path -- verified in
    tests/test_seam_carving.py against both a corrected reference and the
    optimal path cost.
    """
    B, H, W = energy.shape
    cost = energy.clone()
    back = torch.zeros(B, H, W, dtype=torch.long, device=energy.device)
    inf_col = torch.full((B, 1), float("inf"), device=energy.device, dtype=energy.dtype)

    for i in range(1, H):
        prev = cost[:, i - 1, :]
        left = torch.cat([inf_col, prev[:, :-1]], dim=1)
        right = torch.cat([prev[:, 1:], inf_col], dim=1)
        cand = torch.stack([left, prev, right], dim=0)
        min_val, min_idx = cand.min(dim=0)
        cost[:, i, :] = energy[:, i, :] + min_val
        back[:, i, :] = min_idx - 1  # offset in {-1, 0, +1}

    seams = torch.zeros(B, H, dtype=torch.long, device=energy.device)
    seams[:, H - 1] = cost[:, H - 1, :].argmin(dim=1)
    for i in range(H - 1, 0, -1):
        cur = seams[:, i]
        off = back[:, i, :].gather(1, cur.unsqueeze(1)).squeeze(1)
        seams[:, i - 1] = (cur + off).clamp(0, W - 1)
    return seams


def find_horizontal_seams(energy: torch.Tensor) -> torch.Tensor:
    """energy: (B,H,W) -> seams: (B,W), one seam (row index per column) per image."""
    return find_vertical_seams(energy.transpose(1, 2))


def remove_vertical_seams(image: torch.Tensor, seams: torch.Tensor) -> torch.Tensor:
    """image: (B,C,H,W); seams: (B,H) -> (B,C,H,W-1)."""
    B, C, H, W = image.shape
    col_idx = torch.arange(W, device=image.device).view(1, 1, W).expand(B, H, W)
    mask = col_idx != seams.unsqueeze(-1)
    mask_c = mask.unsqueeze(1).expand(B, C, H, W)
    return image[mask_c].view(B, C, H, W - 1)


def remove_horizontal_seams(image: torch.Tensor, seams: torch.Tensor) -> torch.Tensor:
    """image: (B,C,H,W); seams: (B,W) -> (B,C,H-1,W)."""
    B, C, H, W = image.shape
    row_idx = torch.arange(H, device=image.device).view(1, H, 1).expand(B, H, W)
    mask = row_idx != seams.unsqueeze(1)
    mask_c = mask.unsqueeze(1).expand(B, C, H, W)
    kept = image.transpose(2, 3)[mask_c.transpose(2, 3)].view(B, C, W, H - 1)
    return kept.transpose(2, 3)


def seam_carve_pool_batch(x: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    """x: (B,C,H,W) -> (B,C,target_height,target_width). Whole batch per step."""
    with torch.no_grad():
        while x.shape[3] > target_width:
            energy = batched_energy(x)
            seams = find_vertical_seams(energy)
            x = remove_vertical_seams(x, seams)
        while x.shape[2] > target_height:
            energy = batched_energy(x)
            seams = find_horizontal_seams(energy)
            x = remove_horizontal_seams(x, seams)
    return x


class SeamCarvingPooling2D(nn.Module):
    def __init__(self, target_height: int, target_width: int):
        super().__init__()
        self.target_height = target_height
        self.target_width = target_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return seam_carve_pool_batch(x, self.target_height, self.target_width)
