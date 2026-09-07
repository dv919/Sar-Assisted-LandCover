"""
One-off, inference-only regeneration of the final consolidated figure set with neutral naming
(no 'round N' language) -- reuses already-trained checkpoints and already-saved metrics, no
retraining. Run once after the codebase's iterative-round framing was consolidated into a single
final narrative for README.md / the reference report.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from calibration import apply_temperature, expected_calibration_error, fit_temperature
from dataset import BENSubset, load_norm_stats
from evaluate import predict_logits
from labels import load_classes
from models import SimpleCNN
from visualize import plot_reliability_diagram, FIG_DIR

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
METRICS_DIR = ROOT / "outputs" / "metrics"
SEED = 42
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_splits():
    df = pd.read_csv(DATA_DIR / "subset_patches_complete.csv")
    return (df[df["split"] == "train"].reset_index(drop=True),
            df[df["split"] == "validation"].reset_index(drop=True),
            df[df["split"] == "test"].reset_index(drop=True))


def regenerate_reliability_diagrams():
    print("=== Regenerating reliability diagrams (neutral naming) ===")
    _, val_df, test_df = get_splits()
    classes = load_classes()

    model_b = SimpleCNN(in_channels=12, num_classes=len(classes))
    model_b.load_state_dict(torch.load(CKPT_DIR / "model_a_v2.pt", map_location="cpu"))
    model_b = model_b.to(DEVICE)
    model_c = SimpleCNN(in_channels=14, num_classes=len(classes))
    model_c.load_state_dict(torch.load(CKPT_DIR / "model_c_fixed50_v2.pt", map_location="cpu"))
    model_c = model_c.to(DEVICE)

    val_ds_b = BENSubset(val_df, mode="optical_masked", coverage=0.5, seed_eval=SEED + 1)
    y_val_b, logits_val_b, _ = predict_logits(model_b, torch.utils.data.DataLoader(val_ds_b, batch_size=BATCH_SIZE), DEVICE)
    T_b = fit_temperature(logits_val_b, y_val_b)

    val_ds_c = BENSubset(val_df, mode="fusion", coverage=0.5, seed_eval=SEED + 1)
    y_val_c, logits_val_c, _ = predict_logits(model_c, torch.utils.data.DataLoader(val_ds_c, batch_size=BATCH_SIZE), DEVICE)
    T_c = fit_temperature(logits_val_c, y_val_c)

    test_ds_b = BENSubset(test_df, mode="optical_masked", coverage=0.5, seed_eval=SEED)
    y_test_b, logits_test_b, _ = predict_logits(model_b, torch.utils.data.DataLoader(test_ds_b, batch_size=BATCH_SIZE), DEVICE)
    probs_before_b = 1.0 / (1.0 + np.exp(-logits_test_b))
    probs_after_b = apply_temperature(logits_test_b, T_b)
    ece_before_b = expected_calibration_error(probs_before_b, y_test_b)
    ece_after_b = expected_calibration_error(probs_after_b, y_test_b)

    test_ds_c = BENSubset(test_df, mode="fusion", coverage=0.5, seed_eval=SEED)
    y_test_c, logits_test_c, _ = predict_logits(model_c, torch.utils.data.DataLoader(test_ds_c, batch_size=BATCH_SIZE), DEVICE)
    probs_before_c = 1.0 / (1.0 + np.exp(-logits_test_c))
    probs_after_c = apply_temperature(logits_test_c, T_c)
    ece_before_c = expected_calibration_error(probs_before_c, y_test_c)
    ece_after_c = expected_calibration_error(probs_after_c, y_test_c)

    plot_reliability_diagram(ece_before_b["bins"], ece_after_b["bins"], ece_before_b["ece"], ece_after_b["ece"],
                              title="Optical-only model (B) at 50% simulated cloud coverage",
                              out_name="reliability_B.png")
    plot_reliability_diagram(ece_before_c["bins"], ece_after_c["bins"], ece_before_c["ece"], ece_after_c["ece"],
                              title="SAR-fusion model at 50% simulated cloud coverage",
                              out_name="reliability_fusion.png")
    print(f"T_B={T_b:.3f} ECE {ece_before_b['ece']:.4f}->{ece_after_b['ece']:.4f}")
    print(f"T_C={T_c:.3f} ECE {ece_before_c['ece']:.4f}->{ece_after_c['ece']:.4f}")


def regenerate_entropy_chart():
    print("=== Regenerating mean-entropy-vs-coverage chart (final/weighted results) ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(METRICS_DIR / "results_v2.csv")
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    b = df[df["condition"] == "B"].sort_values("coverage")
    c = df[df["condition"] == "C-fixed50"].sort_values("coverage")
    a_ref = df[df["condition"] == "A"]["mean_entropy"].iloc[0]
    ax.plot(b["coverage"] * 100, b["mean_entropy"], marker="o", color="#DD8452", label="B: degraded optical only", linewidth=2)
    ax.plot(c["coverage"] * 100, c["mean_entropy"], marker="o", color="#55A868", label="C: degraded optical + SAR", linewidth=2)
    ax.axhline(a_ref, color="#4C72B0", linestyle="--", linewidth=1.5, label="A: optical only (clean)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Simulated cloud coverage (%)")
    ax.set_ylabel("mean entropy")
    ax.set_title("Mean predictive entropy vs. cloud coverage")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mean_entropy_vs_coverage.png")
    plt.close(fig)
    print("saved mean_entropy_vs_coverage.png")


def rename_remaining_figures():
    print("=== Renaming remaining figures to neutral filenames ===")
    import shutil
    mapping = {
        "v2_per_class_f1_B_vs_bestC.png": "per_class_f1_broad_experiment_50pct.png",
        "v2_qualitative_cases_50pct.png": "qualitative_cases_broad_experiment.png",
        "v3_macro_f1_vs_coverage.png": "macro_f1_deep_experiment.png",
    }
    for old, new in mapping.items():
        old_p, new_p = FIG_DIR / old, FIG_DIR / new
        if old_p.exists():
            shutil.copy(old_p, new_p)
            print(f"  {old} -> {new}")
        else:
            print(f"  MISSING: {old}")


if __name__ == "__main__":
    regenerate_reliability_diagrams()
    regenerate_entropy_chart()
    rename_remaining_figures()
    print("\nDone.")
