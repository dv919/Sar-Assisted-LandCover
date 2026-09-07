"""
"Which classes confuse each other" -- named explicitly in REQUIREMENTS.md's own
uncertainty-reasoning ask (§10) but not answered by per-class F1 alone (it shows *that* a class is
missed, not *what the model said instead*). No retraining needed: loads the broad experiment's
already-trained checkpoints (model_a_v2, model_c_fixed50_v2 -- the optical-only baseline and the
selected SAR-fusion configuration) and runs inference at 50% coverage on the test split.

Confusion definition used (multi-label, so a standard single-label confusion matrix doesn't
apply): for every sample and every true class i the model MISSES (false negative), count every
class j the model WRONGLY PREDICTS on that same sample (false positive) as "i confused for j".
This directly answers "when the model misses class i, what does it say instead?" -- restricted to
the 16 classes with valid support (see evaluate.compute_valid_class_mask) since the 3 excluded
classes have no meaningful signal to confuse.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import BENSubset
from evaluate import compute_valid_class_mask, predict
from labels import load_classes
from models import SimpleCNN

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
FIG_DIR = ROOT / "outputs" / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"
SEED = 42
COVERAGE = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def confusion_matrix_multilabel(y_true, y_pred, classes, valid_idx):
    n = len(valid_idx)
    M = np.zeros((n, n), dtype=int)
    for sample_true, sample_pred in zip(y_true, y_pred):
        fn = [k for k, vi in enumerate(valid_idx) if sample_true[vi] == 1 and sample_pred[vi] == 0]
        fp = [k for k, vi in enumerate(valid_idx) if sample_true[vi] == 0 and sample_pred[vi] == 1]
        for i in fn:
            for j in fp:
                M[i, j] += 1
    return M


def plot_confusion(M, class_names, title, out_name, top_n=None):
    fig, ax = plt.subplots(figsize=(9, 9.5), dpi=150)
    im = ax.imshow(M, cmap="Oranges")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=75, ha="right", fontsize=7.5)
    ax.set_yticklabels(class_names, fontsize=7.5)
    ax.set_xlabel("Wrongly predicted (false positive)")
    ax.set_ylabel("Missed true class (false negative)")
    fig.suptitle(title, fontsize=10.5, wrap=True, y=0.99)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="co-occurrence count")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / out_name)
    plt.close(fig)


def top_pairs(M, class_names, k=8):
    pairs = []
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if i != j and M[i, j] > 0:
                pairs.append((M[i, j], class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    return pairs[:k]


def main():
    df = pd.read_csv(DATA_DIR / "subset_patches_complete.csv")
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    classes = load_classes()
    valid_mask, excluded, _ = compute_valid_class_mask(train_df, val_df, test_df, classes)
    valid_idx = [i for i, m in enumerate(valid_mask) if m]
    valid_names = [classes[i] for i in valid_idx]
    print(f"Analyzing confusion over {len(valid_idx)} valid classes (excluded: {excluded})")

    model_a = SimpleCNN(in_channels=12, num_classes=len(classes))
    model_a.load_state_dict(torch.load(CKPT_DIR / "model_a_v2.pt", map_location="cpu"))
    model_a = model_a.to(DEVICE)
    model_c = SimpleCNN(in_channels=14, num_classes=len(classes))
    model_c.load_state_dict(torch.load(CKPT_DIR / "model_c_fixed50_v2.pt", map_location="cpu"))
    model_c = model_c.to(DEVICE)

    ds_b = BENSubset(test_df, mode="optical_masked", coverage=COVERAGE, seed_eval=SEED)
    loader_b = torch.utils.data.DataLoader(ds_b, batch_size=32, shuffle=False)
    y_true_b, y_prob_b, ids_b = predict(model_a, loader_b, DEVICE)
    y_pred_b = (y_prob_b >= 0.5).astype(int)

    ds_c = BENSubset(test_df, mode="fusion", coverage=COVERAGE, seed_eval=SEED)
    loader_c = torch.utils.data.DataLoader(ds_c, batch_size=32, shuffle=False)
    y_true_c, y_prob_c, ids_c = predict(model_c, loader_c, DEVICE)
    y_pred_c = (y_prob_c >= 0.5).astype(int)
    assert ids_b == ids_c

    M_b = confusion_matrix_multilabel(y_true_b, y_pred_b, classes, valid_idx)
    M_c = confusion_matrix_multilabel(y_true_c, y_pred_c, classes, valid_idx)

    plot_confusion(M_b, valid_names, "Optical-only model (B) @ 50% coverage: "
                                      "missed class (row) vs. wrongly-predicted class (col)",
                   "confusion_B.png")
    plot_confusion(M_c, valid_names, "SAR-fusion model @ 50% coverage: "
                                      "missed class (row) vs. wrongly-predicted class (col)",
                   "confusion_fusion.png")

    top_b = top_pairs(M_b, valid_names)
    top_c = top_pairs(M_c, valid_names)
    print("\nTop confused pairs, model B:")
    for count, i, j in top_b:
        print(f"  missed '{i}' -> wrongly said '{j}'  (n={count})")
    print("\nTop confused pairs, model C:")
    for count, i, j in top_c:
        print(f"  missed '{i}' -> wrongly said '{j}'  (n={count})")

    summary = {
        "valid_classes": valid_names,
        "top_pairs_B": [{"count": int(c), "missed": i, "predicted_instead": j} for c, i, j in top_b],
        "top_pairs_C": [{"count": int(c), "missed": i, "predicted_instead": j} for c, i, j in top_c],
    }
    with open(METRICS_DIR / "confusion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved outputs/figures/confusion_B.png, confusion_fusion.png, "
          "outputs/metrics/confusion_summary.json")


if __name__ == "__main__":
    main()
