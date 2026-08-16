"""Dataset loaders. Fashion-MNIST is grayscale/28x28/10-class; it's resized to
32x32 and replicated to 3 channels so it drops into the exact same conv1
(3-channel input) architecture used for the CUB bird experiments -- keeping
architecture and hyperparameters fixed while only the dataset changes.
"""
import random

import torch
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import FashionMNIST

FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_fashion_mnist_dataset(root: str = "data/fashion_mnist", image_size: int = 32,
                               samples_per_class: int | None = 200, seed: int = 12,
                               download: bool = True):
    """Returns a torch Dataset of (3,image_size,image_size) tensors, 10 classes.

    samples_per_class=None uses the full 60,000-image training split (slow on
    CPU with seam-carving pooling in the loop); the default subsamples evenly
    per class to keep a full train/val run to a few minutes on a laptop,
    similar in spirit to the small subset used for the CUB bird experiments.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    dataset = FashionMNIST(root=root, train=True, download=download, transform=transform)

    if samples_per_class is None:
        return dataset

    rng = random.Random(seed)
    by_class = {c: [] for c in range(10)}
    for idx, label in enumerate(dataset.targets.tolist()):
        by_class[label].append(idx)
    indices = []
    for c, idxs in by_class.items():
        rng.shuffle(idxs)
        indices.extend(idxs[:samples_per_class])
    rng.shuffle(indices)
    return Subset(dataset, indices)
