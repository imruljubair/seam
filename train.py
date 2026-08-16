"""Train and compare the seam-carving and max-pooling CNNs.

Usage:
    # CUB bird dataset (ImageFolder layout), matches the paper's setup:
    python train.py --dataset imagefolder --data-dir /path/to/dataset3_normal --out-dir runs/birds

    # Fashion-MNIST, auto-downloads on first run:
    python train.py --dataset fashion_mnist --out-dir runs/fashion_mnist
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from seam_carving import ExampleCNN, ExampleCNNmaxpool
from seam_carving.data import FASHION_MNIST_CLASSES, get_fashion_mnist_dataset
from seam_carving.evaluate import evaluate_and_report
from seam_carving.verify import snapshot, verify_freeze


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, dataloader, val_loader, device, num_epochs, patience,
                 learning_rate, weight_path, desc, name):
    model.to(device)
    before = snapshot(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=learning_rate)

    loss_list, val_loss_list = [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for _ in tqdm(range(num_epochs), desc=desc):
        model.train()
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for vinputs, vtargets in val_loader:
                vinputs, vtargets = vinputs.to(device), vtargets.to(device)
                running_val_loss += criterion(model(vinputs), vtargets).item()
        epoch_val_loss = running_val_loss / len(val_loader)
        val_loss_list.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), weight_path)
        else:
            patience_counter += 1

        loss_list.append(loss.item())
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    print(f"{desc} complete. Best eval loss: {best_val_loss:.4f}")
    verify_freeze(model, before, name)
    return loss_list, val_loss_list, best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["imagefolder", "fashion_mnist"], default="imagefolder")
    parser.add_argument("--data-dir", default=None,
                         help="ImageFolder-style dataset dir (required for --dataset imagefolder; "
                              "download/cache dir for --dataset fashion_mnist, default data/fashion_mnist)")
    parser.add_argument("--out-dir", default="runs/default")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None, help="default: 300 (imagefolder) / 60 (fashion_mnist)")
    parser.add_argument("--patience", type=int, default=None, help="default: 25 (imagefolder) / 10 (fashion_mnist)")
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--samples-per-class", type=int, default=200,
                         help="fashion_mnist only; None (pass -1) uses the full 60k training set")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.dataset == "imagefolder":
        if args.data_dir is None:
            parser.error("--data-dir is required for --dataset imagefolder")
        args.epochs = args.epochs if args.epochs is not None else 300
        args.patience = args.patience if args.patience is not None else 25
        num_classes = None  # inferred from ImageFolder below

        set_seed(args.seed)
        # raw CUB images are variable-size; the paper resizes to 32x32
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        dataset = ImageFolder(root=args.data_dir, transform=transform)
        num_classes = len(dataset.classes)
        class_names = dataset.classes
        print(f"Dataset size: {len(dataset)}, classes: {dataset.classes}")
    else:
        args.epochs = args.epochs if args.epochs is not None else 60
        args.patience = args.patience if args.patience is not None else 10
        samples_per_class = None if args.samples_per_class == -1 else args.samples_per_class
        root = args.data_dir or "data/fashion_mnist"

        set_seed(args.seed)
        dataset = get_fashion_mnist_dataset(root=root, samples_per_class=samples_per_class,
                                             seed=args.seed)
        num_classes = 10
        class_names = FASHION_MNIST_CLASSES
        print(f"Dataset size: {len(dataset)} (samples_per_class={samples_per_class}), "
              f"classes: 10 (Fashion-MNIST)")

    train_size = int(args.split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)

    set_seed(args.seed)
    model_seam = ExampleCNN(num_classes=num_classes, freeze_conv1=True)
    loss_seam, val_loss_seam, best_seam = train_model(
        model_seam, dataloader, val_loader, device, args.epochs, args.patience,
        args.lr, f"{args.out_dir}/seam_carving.pth", "Seam Carving - Epochs", "SeamCarving")

    set_seed(args.seed)
    model_max = ExampleCNNmaxpool(num_classes=num_classes, freeze_conv1=True)
    loss_max, val_loss_max, best_max = train_model(
        model_max, dataloader, val_loader, device, args.epochs, args.patience,
        args.lr, f"{args.out_dir}/maxpool.pth", "Maxpool - Epochs", "Maxpool")

    same_conv1 = torch.equal(model_seam.conv1.weight, model_max.conv1.weight)
    print(f"\nconv1 identical between both models (same seed, both frozen): {same_conv1}")

    np.save(f"{args.out_dir}/loss_seam.npy", loss_seam)
    np.save(f"{args.out_dir}/val_loss_seam.npy", val_loss_seam)
    np.save(f"{args.out_dir}/loss_max.npy", loss_max)
    np.save(f"{args.out_dir}/val_loss_max.npy", val_loss_max)
    print(f"\nBest eval loss -- seam carving: {best_seam:.4f}, maxpool: {best_max:.4f}")

    # Reload each model's best (early-stopping) checkpoint -- not necessarily the
    # last epoch trained -- before computing final accuracy/F1/confusion matrix.
    model_seam.load_state_dict(torch.load(f"{args.out_dir}/seam_carving.pth", map_location=device))
    model_max.load_state_dict(torch.load(f"{args.out_dir}/maxpool.pth", map_location=device))
    evaluate_and_report(model_seam, model_max, val_loader, device, class_names, args.out_dir)
    print(f"\nSaved confusion_matrices.png and classification_report.txt to {args.out_dir}/")


if __name__ == "__main__":
    main()
