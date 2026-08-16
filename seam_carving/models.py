"""CNN variants for the seam-carving-vs-max-pooling comparison.

conv1 is frozen in BOTH models. The seam-carving pooling layer runs under
torch.no_grad() (see pooling.py) and can't pass gradients back to conv1;
freezing conv1 in the max-pooling model too removes that asymmetry, and with
matching seeds both models get bit-identical frozen conv1 weights -- so the
comparison is isolated to the pooling operator itself. See
verify.snapshot/verify_freeze for a runtime check of this invariant.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pooling import SeamCarvingPooling2D


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


class ExampleCNN(nn.Module):
    """Seam-carving pooling variant."""

    def __init__(self, num_classes: int = 2, freeze_conv1: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.seam_pool = SeamCarvingPooling2D(target_height=16, target_width=16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, num_classes)
        if freeze_conv1:
            _freeze(self.conv1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.seam_pool(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc1(x)  # raw logits -- CrossEntropyLoss applies log_softmax internally


class ExampleCNNmaxpool(nn.Module):
    """Max-pooling baseline, conv1 frozen the same way for a fair comparison."""

    def __init__(self, num_classes: int = 2, freeze_conv1: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, num_classes)
        if freeze_conv1:
            _freeze(self.conv1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc1(x)
