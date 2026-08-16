"""Runtime checks that the conv1-freeze invariant actually holds."""
import torch
import torch.nn as nn


def snapshot(model: nn.Module) -> dict:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def verify_freeze(model: nn.Module, before: dict, name: str) -> None:
    after = snapshot(model)
    conv1_changed = not torch.equal(before["conv1.weight"], after["conv1.weight"])
    conv2_changed = not torch.equal(before["conv2.weight"], after["conv2.weight"])
    print(f"[{name}] conv1 unchanged (frozen correctly): {not conv1_changed}")
    print(f"[{name}] conv2 changed (gradients flowing):  {conv2_changed}")
    assert not conv1_changed, f"{name}: conv1 changed despite being frozen!"
    assert conv2_changed, f"{name}: conv2 did not update -- gradients aren't flowing!"
