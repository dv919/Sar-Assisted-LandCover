"""
Round 3: fewer classes (8, down from 19), much deeper per-class support (~800-1700+ train
examples per class vs. a few hundred at best in rounds 1-2). See select_subset_v3.py for how
this subset was built from the already-downloaded candidate pool (same 5 tiles, no new network
cost). This script:

1. Trains a weighted-loss optical-only model (B) and one SAR-fusion model (C) on the deeper
   8-class data, using round 2's winning training regime (fixed 50% training-coverage -- the
   ablation winner from run_experiment_v2.py's C-fixed50).
2. Evaluates both across the same four coverage levels as rounds 1-2.
3. Directly answers "did more depth per class actually help?" by ALSO evaluating round 2's
   already-trained 19-class models (model_a_v2, model_c_fixed50_v2) on round 3's test set,
   sliced down to the same 8 classes -- an apples-to-apples comparison on the same evaluation
   data, old (shallow) vs. new (deep) training data for the same classes.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from class_weights import compute_pos_weights
from dataset import BENSubset, load_norm_stats
from evaluate import compute_metrics, compute_valid_class_mask, predict
from labels import load_classes
from models import SimpleCNN
from train import train_model
from visualize import plot_metric_vs_coverage, plot_per_class_f1

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
METRICS_DIR = ROOT / "outputs" / "metrics"
FIG_DIR = ROOT / "outputs" / "figures"

COVERAGE_LEVELS = [0.0, 0.25, 0.5, 0.75]
EPOCHS = 10
BATCH_SIZE = 32
SEED = 42
DEVICE = "cpu"


def get_v3_splits():
    df = pd.read_csv(DATA_DIR / "subset_patches_v3_complete.csv")
    return (df[df["split"] == "train"].reset_index(drop=True),
            df[df["split"] == "validation"].reset_index(drop=True),
            df[df["split"] == "test"].reset_index(drop=True))


def main():
    import os
    torch.set_num_threads(os.cpu_count() or 4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_df, val_df, test_df = get_v3_splits()
    classes8 = json.load(open(DATA_DIR / "classes_v3.json"))
    stats_v3 = load_norm_stats(DATA_DIR / "norm_stats_v3.json")
    print(f"[v3] train={len(train_df)} val={len(val_df)} test={len(test_df)} classes={len(classes8)}")

    valid_mask, excluded, support = compute_valid_class_mask(train_df, val_df, test_df, classes8)
    print(f"[v3] Classes with 0 support in some split: {excluded} (expect none, given 97.8% coverage)")

    pos_weight = compute_pos_weights(train_df, classes8)
    print(f"[v3] pos_weight range: min={pos_weight.min():.2f} max={pos_weight.max():.2f}")

    common_kwargs = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
                          pos_weight=pos_weight, valid_class_mask=valid_mask, classes=classes8)

    # ---------------- Model B: optical-only, clean, weighted loss, 8-class ----------------
    print("\n=== [Round 3] Training Model B (optical only, clean, 8-class, weighted) ===")
    train_b = BENSubset(train_df, mode="optical_only", coverage=0.0, classes=classes8, stats=stats_v3)
    val_b = BENSubset(val_df, mode="optical_only", coverage=0.0, classes=classes8, stats=stats_v3)
    model_b = SimpleCNN(in_channels=12, num_classes=len(classes8))
    model_b, hist_b = train_model(model_b, train_b, val_b, log_prefix="[v3-B] ", **common_kwargs)
    torch.save(model_b.state_dict(), CKPT_DIR / "model_b_v3.pt")
    json.dump(hist_b, open(METRICS_DIR / "v3_history_b.json", "w"), indent=2)

    # ---------------- Model C: SAR fusion, fixed 50% coverage (round-2 winning regime) -------
    print("\n=== [Round 3] Training Model C (concat fusion, coverage FIXED @ 50%, 8-class) ===")
    train_c = BENSubset(train_df, mode="fusion", coverage=0.5, classes=classes8, stats=stats_v3)
    val_c = BENSubset(val_df, mode="fusion", coverage=0.5, classes=classes8, stats=stats_v3)
    model_c = SimpleCNN(in_channels=14, num_classes=len(classes8))
    model_c, hist_c = train_model(model_c, train_c, val_c, log_prefix="[v3-C] ", **common_kwargs)
    torch.save(model_c.state_dict(), CKPT_DIR / "model_c_v3.pt")
    json.dump(hist_c, open(METRICS_DIR / "v3_history_c.json", "w"), indent=2)

    # ---------------- Evaluate v3 B/C across coverage levels on v3 test split ----------------
    print("\n=== [Round 3] Evaluating v3 B/C across coverage levels ===")
    rows = []
    per_class_by_cov = {}
    for cov in COVERAGE_LEVELS:
        ds_b = BENSubset(test_df, mode="optical_masked", coverage=cov, seed_eval=SEED,
                          classes=classes8, stats=stats_v3)
        loader_b = torch.utils.data.DataLoader(ds_b, batch_size=BATCH_SIZE, shuffle=False)
        y_true_b, y_prob_b, ids_b = predict(model_b, loader_b, DEVICE)
        metrics_b = compute_metrics(y_true_b, y_prob_b, valid_class_mask=valid_mask, classes=classes8)
        cond_label = "A" if cov == 0.0 else "B"
        rows.append({"condition": cond_label, "coverage": cov,
                     **{k: v for k, v in metrics_b.items() if k != "per_class_f1"}})
        per_class_by_cov.setdefault(cov, {})["B"] = metrics_b["per_class_f1"]

        ds_c = BENSubset(test_df, mode="fusion", coverage=cov, seed_eval=SEED,
                          classes=classes8, stats=stats_v3)
        loader_c = torch.utils.data.DataLoader(ds_c, batch_size=BATCH_SIZE, shuffle=False)
        y_true_c, y_prob_c, ids_c = predict(model_c, loader_c, DEVICE)
        metrics_c = compute_metrics(y_true_c, y_prob_c, valid_class_mask=valid_mask, classes=classes8)
        rows.append({"condition": "C", "coverage": cov,
                     **{k: v for k, v in metrics_c.items() if k != "per_class_f1"}})
        per_class_by_cov[cov]["C"] = metrics_c["per_class_f1"]

        print(f"[v3] coverage={cov:.2f}  B: macroF1={metrics_b['macro_f1_all']:.4f}  |  "
              f"C: macroF1={metrics_c['macro_f1_all']:.4f}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(METRICS_DIR / "results_v3.csv", index=False)
    json.dump(per_class_by_cov, open(METRICS_DIR / "v3_per_class_f1_by_coverage.json", "w"), indent=2)

    plot_metric_vs_coverage(results_df, metric="macro_f1_all", out_name="v3_macro_f1_vs_coverage.png")
    plot_per_class_f1(per_class_by_cov[0.5]["B"], per_class_by_cov[0.5]["C"], "50%",
                       out_name="v3_per_class_f1_50pct.png",
                       title="Round 3 (8 deep classes) per-class F1 @ 50% coverage: B vs C")

    # ---------------- Cross-round comparison: does more depth actually help? ----------------
    print("\n=== [Round 3] Cross-round comparison: round-2 (shallow) vs round-3 (deep) models, "
          "both evaluated on round-3's test set, same 8 classes ===")
    cross_compare(test_df, classes8, valid_mask, model_b, model_c)

    print("\n[Round 3] Done. See outputs/figures/v3_*.png, outputs/metrics/*_v3*, "
          "outputs/checkpoints/*_v3.pt.")


def cross_compare(test_df_v3, classes8, valid_mask8, model_b_v3, model_c_v3):
    """Evaluate round 2's 19-class models on round 3's (larger, differently-sampled) test set,
    then slice their predictions down to the 8 shared classes -- a same-test-set, same-classes
    comparison of shallow-training-data (round 2) vs. deep-training-data (round 3) models."""
    classes19 = load_classes()  # round 1/2's 19-class vocabulary
    stats_v1v2 = load_norm_stats()  # round 1/2's norm_stats.json (what those checkpoints expect)
    idx8_in_19 = [classes19.index(c) for c in classes8]

    model_a_v2 = SimpleCNN(in_channels=12, num_classes=len(classes19))
    model_a_v2.load_state_dict(torch.load(CKPT_DIR / "model_a_v2.pt", map_location="cpu"))
    model_c_v2 = SimpleCNN(in_channels=14, num_classes=len(classes19))
    model_c_v2.load_state_dict(torch.load(CKPT_DIR / "model_c_fixed50_v2.pt", map_location="cpu"))

    rows = []
    for cov in COVERAGE_LEVELS:
        # round-2 models (19-class), evaluated on v3 test set, normalized with round-1/2 stats
        ds_b19 = BENSubset(test_df_v3, mode="optical_masked", coverage=cov, seed_eval=SEED,
                            classes=classes19, stats=stats_v1v2)
        loader_b19 = torch.utils.data.DataLoader(ds_b19, batch_size=BATCH_SIZE, shuffle=False)
        y_true_19, y_prob_19, _ = predict(model_a_v2, loader_b19, DEVICE)
        y_true_8 = y_true_19[:, idx8_in_19]
        y_prob_8 = y_prob_19[:, idx8_in_19]
        m = compute_metrics(y_true_8, y_prob_8, classes=classes8)
        rows.append({"model": "round2_shallow_B", "coverage": cov, "macro_f1": m["macro_f1_all"],
                     "micro_f1": m["micro_f1"]})

        ds_c19 = BENSubset(test_df_v3, mode="fusion", coverage=cov, seed_eval=SEED,
                            classes=classes19, stats=stats_v1v2)
        loader_c19 = torch.utils.data.DataLoader(ds_c19, batch_size=BATCH_SIZE, shuffle=False)
        y_true_19c, y_prob_19c, _ = predict(model_c_v2, loader_c19, DEVICE)
        y_true_8c = y_true_19c[:, idx8_in_19]
        y_prob_8c = y_prob_19c[:, idx8_in_19]
        mc = compute_metrics(y_true_8c, y_prob_8c, classes=classes8)
        rows.append({"model": "round2_shallow_C", "coverage": cov, "macro_f1": mc["macro_f1_all"],
                     "micro_f1": mc["micro_f1"]})

        # round-3 models (8-class, deep), same test set
        ds_b3 = BENSubset(test_df_v3, mode="optical_masked", coverage=cov, seed_eval=SEED,
                           classes=classes8, stats=load_norm_stats(DATA_DIR / "norm_stats_v3.json"))
        loader_b3 = torch.utils.data.DataLoader(ds_b3, batch_size=BATCH_SIZE, shuffle=False)
        y_true_b3, y_prob_b3, _ = predict(model_b_v3, loader_b3, DEVICE)
        m3b = compute_metrics(y_true_b3, y_prob_b3, classes=classes8)
        rows.append({"model": "round3_deep_B", "coverage": cov, "macro_f1": m3b["macro_f1_all"],
                     "micro_f1": m3b["micro_f1"]})

        ds_c3 = BENSubset(test_df_v3, mode="fusion", coverage=cov, seed_eval=SEED,
                           classes=classes8, stats=load_norm_stats(DATA_DIR / "norm_stats_v3.json"))
        loader_c3 = torch.utils.data.DataLoader(ds_c3, batch_size=BATCH_SIZE, shuffle=False)
        y_true_c3, y_prob_c3, _ = predict(model_c_v3, loader_c3, DEVICE)
        m3c = compute_metrics(y_true_c3, y_prob_c3, classes=classes8)
        rows.append({"model": "round3_deep_C", "coverage": cov, "macro_f1": m3c["macro_f1_all"],
                     "micro_f1": m3c["micro_f1"]})

        print(f"coverage={cov:.2f}  shallow_B={m['macro_f1_all']:.4f} deep_B={m3b['macro_f1_all']:.4f}  |  "
              f"shallow_C={mc['macro_f1_all']:.4f} deep_C={m3c['macro_f1_all']:.4f}")

    cross_df = pd.DataFrame(rows)
    cross_df.to_csv(METRICS_DIR / "v3_cross_round_comparison.csv", index=False)
    plot_cross_comparison(cross_df)


def plot_cross_comparison(cross_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)
    styles = {
        "round2_shallow_B": ("#DD8452", "--", "Round 2 (shallow data) B"),
        "round3_deep_B": ("#DD8452", "-", "Round 3 (deep data) B"),
        "round2_shallow_C": ("#55A868", "--", "Round 2 (shallow data) C"),
        "round3_deep_C": ("#55A868", "-", "Round 3 (deep data) C"),
    }
    for model_name, (color, ls, label) in styles.items():
        sub = cross_df[cross_df["model"] == model_name].sort_values("coverage")
        ax.plot(sub["coverage"] * 100, sub["macro_f1"], marker="o", color=color,
                linestyle=ls, label=label, linewidth=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Simulated cloud coverage (%)")
    ax.set_ylabel("Macro F1 (8 shared classes)")
    ax.set_title("Same 8 classes/test set: shallow (round 2) vs.\ndeep (round 3) training data", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v3_cross_round_comparison.png")
    plt.close(fig)
    print("Saved outputs/figures/v3_cross_round_comparison.png")


if __name__ == "__main__":
    main()
