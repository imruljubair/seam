"""Correctness tests for the vectorized seam carving against the original
notebook's implementation, plus the freeze-invariant check the models rely on.

Run with: python -m pytest tests/ -v
"""
import time

import numpy as np
import pytest
import torch

from tests import reference_original as ref
from seam_carving.pooling import (
    batched_energy,
    find_vertical_seams,
    seam_carve_pool_batch,
)
from seam_carving.models import ExampleCNN, ExampleCNNmaxpool
from seam_carving.verify import snapshot


def test_energy_matches_scipy_sobel():
    np.random.seed(0)
    img_np = np.random.rand(3, 16, 16).astype(np.float32)
    e_ref = ref.energy_function(img_np)
    e_vec = batched_energy(torch.from_numpy(img_np).unsqueeze(0))[0].numpy()
    assert np.abs(e_ref - e_vec).max() < 1e-5


@pytest.mark.parametrize("trial", range(10))
def test_seam_is_optimal_and_matches_corrected_reference(trial):
    np.random.seed(trial)
    H, W = np.random.randint(4, 16), np.random.randint(4, 16)
    energy = np.random.rand(H, W).astype(np.float32)

    seam_corrected = ref.corrected_find_vertical_seam(energy)
    seam_vec = find_vertical_seams(torch.from_numpy(energy).unsqueeze(0))[0].numpy()
    assert np.array_equal(seam_corrected, seam_vec)

    # also check true path optimality independent of the reference
    rows, cols = energy.shape
    cost = energy.copy()
    for i in range(1, rows):
        for j in range(cols):
            m = cost[i - 1, j]
            if j > 0:
                m = min(m, cost[i - 1, j - 1])
            if j < cols - 1:
                m = min(m, cost[i - 1, j + 1])
            cost[i, j] += m
    optimal_cost = cost[-1].min()
    path_cost = sum(energy[i, seam_vec[i]] for i in range(rows))
    assert abs(path_cost - optimal_cost) < 1e-4


def test_full_pipeline_matches_corrected_reference():
    np.random.seed(0)
    B, C, H, W = 6, 16, 32, 32
    x_np = np.random.rand(B, C, H, W).astype(np.float32)
    fixed_out = ref.corrected_reference_batch(x_np, 16, 16)
    vec_out = seam_carve_pool_batch(torch.from_numpy(x_np.copy()), 16, 16).numpy()
    assert np.abs(fixed_out - vec_out).max() < 1e-4


def test_original_has_a_boundary_backtrack_bug():
    """Documents the original's bug: it disagrees with the true-optimal result
    on real feature-map-shaped data. Not a defect in seam_carving/pooling.py."""
    np.random.seed(0)
    B, C, H, W = 6, 16, 32, 32
    x_np = np.random.rand(B, C, H, W).astype(np.float32)
    buggy_out = ref.reference_batch(x_np, 16, 16)
    fixed_out = ref.corrected_reference_batch(x_np, 16, 16)
    mismatch_fraction = (np.abs(buggy_out - fixed_out) > 1e-4).mean()
    assert mismatch_fraction > 0.1  # the bug reliably manifests on real-sized inputs


def test_vectorized_is_faster_at_training_batch_size():
    np.random.seed(0)
    B = 16  # matches the original notebook's training batch_size
    x_np = np.random.rand(B, 16, 32, 32).astype(np.float32)

    t0 = time.time()
    ref.reference_batch(x_np, 16, 16)
    orig_t = time.time() - t0

    t0 = time.time()
    seam_carve_pool_batch(torch.from_numpy(x_np.copy()), 16, 16)
    vec_t = time.time() - t0

    assert vec_t < orig_t


def test_conv1_frozen_and_identical_across_both_models():
    torch.manual_seed(12)
    model_seam = ExampleCNN(freeze_conv1=True)
    torch.manual_seed(12)
    model_max = ExampleCNNmaxpool(freeze_conv1=True)
    assert torch.equal(model_seam.conv1.weight, model_max.conv1.weight)
    assert all(not p.requires_grad for p in model_seam.conv1.parameters())
    assert all(not p.requires_grad for p in model_max.conv1.parameters())


def test_gradients_flow_to_conv2_and_fc1_but_not_conv1():
    torch.manual_seed(12)
    model = ExampleCNN(freeze_conv1=True)
    before = snapshot(model)

    x = torch.rand(8, 3, 32, 32)
    y = torch.randint(0, 2, (8,))
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(3):
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

    after = snapshot(model)
    assert torch.equal(before["conv1.weight"], after["conv1.weight"])
    assert not torch.equal(before["conv2.weight"], after["conv2.weight"])
    assert not torch.equal(before["fc1.weight"], after["fc1.weight"])
