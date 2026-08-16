"""Accuracy / precision / recall / F1 / confusion-matrix reporting."""
import matplotlib
matplotlib.use("Agg")  # headless-safe: no GUI backend required
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(targets.numpy())
    return np.array(all_labels), np.array(all_preds)


_ANNOTATE_MAX_CLASSES = 15  # above this, per-cell numbers/tick labels stop being legible/fast


def report(y_true, y_pred, class_names, name):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    text = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                  target_names=class_names, zero_division=0)
    accuracy = (y_true == y_pred).mean()
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy:.4f}")
    if len(class_names) <= _ANNOTATE_MAX_CLASSES:
        print(text)
    else:
        # full per-class report still goes to the saved file; console gets the summary
        summary = "\n".join(text.splitlines()[-4:])
        print(f"({len(class_names)} classes -- full per-class report saved to file)")
        print(summary)
    return cm, text, accuracy


def plot_confusion_matrices(cm_seam, cm_max, class_names, out_path):
    small = len(class_names) <= _ANNOTATE_MAX_CLASSES
    figsize = (5 + len(class_names), 2.5 + len(class_names) * 0.4) if small else (11, 5)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, cm, title in zip(axes, [cm_seam, cm_max], ["Seam Carving", "Maxpool"]):
        ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        if small:
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(class_names, fontsize=7)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                             color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        else:
            # too many classes for readable ticks/annotations -- raw heatmap only
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_and_report(model_seam, model_max, loader, device, class_names, out_dir):
    y_true_seam, y_pred_seam = collect_predictions(model_seam, loader, device)
    y_true_max, y_pred_max = collect_predictions(model_max, loader, device)

    cm_seam, text_seam, acc_seam = report(y_true_seam, y_pred_seam, class_names, "Seam Carving")
    cm_max, text_max, acc_max = report(y_true_max, y_pred_max, class_names, "Maxpool")

    plot_confusion_matrices(cm_seam, cm_max, class_names, f"{out_dir}/confusion_matrices.png")
    with open(f"{out_dir}/classification_report.txt", "w") as f:
        f.write("=== Seam Carving ===\n")
        f.write(f"Accuracy: {acc_seam:.4f}\n")
        f.write(text_seam)
        f.write("\n=== Maxpool ===\n")
        f.write(f"Accuracy: {acc_max:.4f}\n")
        f.write(text_max)

    return {"seam_accuracy": acc_seam, "max_accuracy": acc_max}
