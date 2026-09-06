"""
End-to-end experiment runner for the three conditions:
  A. Optical only, clean.
  B. Same clean-trained model A, evaluated on masked optical at several coverage levels
     (per REQUIREMENTS.md #5, "minimum baseline": train once, evaluate on degraded input).
  C. One SAR-assisted model (early channel concatenation), trained with randomly-sampled
     mask coverage so it learns to lean on SAR when optical is degraded, evaluated at the
     same fixed coverage levels as B for a fair, matched-mask comparison.

Produces:
  outputs/checkpoints/model_a.pt, model_c.pt
  outputs/metrics/results.json         -- full metrics table across conditions x coverage levels
  outputs/metrics/history_a.json, history_c.json
  outputs/figures/*.png                -- curves, per-class bars, qualitative cases
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dataset import BENSubset
from evaluate import compute_metrics, predict
from labels import load_classes
from models import build_model
from train import train_model
from visualize import plot_metric_vs_coverage, plot_per_class_f1, plot_case_grid

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
METRICS_DIR = ROOT / "outputs" / "metrics"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

COVERAGE_LEVELS = [0.0, 0.25, 0.5, 0.75]
EPOCHS = 8
BATCH_SIZE = 32
SEED = 42
DEVICE = "cpu"


def get_splits():
    df = pd.read_csv(DATA_DIR / "subset_patches_complete.csv")
    return (df[df["split"] == "train"].reset_index(drop=True),
            df[df["split"] == "validation"].reset_index(drop=True),
            df[df["split"] == "test"].reset_index(drop=True))


def main():
    import os
    torch.set_num_threads(os.cpu_count() or 4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_df, val_df, test_df = get_splits()
    classes = load_classes()
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)} classes={len(classes)}")

    # ---------------- Condition A: train clean optical-only model ----------------
    print("\n=== Training Model A (optical only, clean) ===")
    train_a = BENSubset(train_df, mode="optical_only", coverage=0.0)
    val_a = BENSubset(val_df, mode="optical_only", coverage=0.0)
    model_a = build_model(in_channels=12, num_classes=len(classes))
    model_a, hist_a = train_model(model_a, train_a, val_a, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                   device=DEVICE, log_prefix="[A] ")
    torch.save(model_a.state_dict(), CKPT_DIR / "model_a.pt")
    with open(METRICS_DIR / "history_a.json", "w") as f:
        json.dump(hist_a, f, indent=2)

    # ---------------- Condition C: train SAR-fusion model w/ random-coverage augmentation --------
    print("\n=== Training Model C (degraded optical + SAR fusion) ===")
    train_c = BENSubset(train_df, mode="fusion", coverage=(0.0, 0.75))
    val_c = BENSubset(val_df, mode="fusion", coverage=(0.0, 0.75))
    model_c = build_model(in_channels=14, num_classes=len(classes))
    model_c, hist_c = train_model(model_c, train_c, val_c, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                   device=DEVICE, log_prefix="[C] ")
    torch.save(model_c.state_dict(), CKPT_DIR / "model_c.pt")
    with open(METRICS_DIR / "history_c.json", "w") as f:
        json.dump(hist_c, f, indent=2)

    # ---------------- Evaluation sweep across coverage levels on TEST split ----------------
    print("\n=== Evaluating A / B / C across coverage levels on test split ===")
    rows = []
    per_class_by_cov = {}
    case_pool = []  # for qualitative figure: (patch_id, coverage, true, pred_b, pred_c, prob_b, prob_c)

    for cov in COVERAGE_LEVELS:
        # Condition A/B: model_a evaluated on optical at this coverage (cov=0 IS condition A)
        eval_ds_b = BENSubset(test_df, mode="optical_masked", coverage=cov, seed_eval=SEED)
        loader_b = torch.utils.data.DataLoader(eval_ds_b, batch_size=BATCH_SIZE, shuffle=False)
        y_true_b, y_prob_b, ids_b = predict(model_a, loader_b, DEVICE)
        metrics_b = compute_metrics(y_true_b, y_prob_b)
        cond_label = "A" if cov == 0.0 else "B"
        rows.append({"condition": cond_label, "coverage": cov, **{k: v for k, v in metrics_b.items() if k != "per_class_f1"}})
        per_class_by_cov.setdefault(cov, {})["B"] = metrics_b["per_class_f1"]

        # Condition C: model_c evaluated on (SAME masked optical realization) + SAR at this coverage
        eval_ds_c = BENSubset(test_df, mode="fusion", coverage=cov, seed_eval=SEED)
        loader_c = torch.utils.data.DataLoader(eval_ds_c, batch_size=BATCH_SIZE, shuffle=False)
        y_true_c, y_prob_c, ids_c = predict(model_c, loader_c, DEVICE)
        metrics_c = compute_metrics(y_true_c, y_prob_c)
        rows.append({"condition": "C", "coverage": cov, **{k: v for k, v in metrics_c.items() if k != "per_class_f1"}})
        per_class_by_cov[cov]["C"] = metrics_c["per_class_f1"]

        print(f"coverage={cov:.2f}  B: macroF1={metrics_b['macro_f1']:.4f} hamAcc={metrics_b['hamming_acc']:.4f}  "
              f"|  C: macroF1={metrics_c['macro_f1']:.4f} hamAcc={metrics_c['hamming_acc']:.4f}")

        assert ids_b == ids_c, "B/C must be evaluated on the same patch order for a fair per-sample comparison"
        for i, pid in enumerate(ids_b):
            case_pool.append({
                "patch_id": pid, "coverage": cov,
                "y_true": y_true_b[i], "prob_b": y_prob_b[i], "prob_c": y_prob_c[i],
            })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(METRICS_DIR / "results.csv", index=False)
    with open(METRICS_DIR / "per_class_f1_by_coverage.json", "w") as f:
        json.dump(per_class_by_cov, f, indent=2)
    print("\nSaved outputs/metrics/results.csv")

    # ---------------- Figures ----------------
    for metric in ["macro_f1", "micro_f1", "hamming_acc", "mean_entropy"]:
        plot_metric_vs_coverage(results_df, metric=metric)
    mid_cov = 0.5
    plot_per_class_f1(per_class_by_cov[mid_cov]["B"], per_class_by_cov[mid_cov]["C"], "50%",
                       out_name="per_class_f1_50pct.png")

    build_case_figures(case_pool, classes, test_df)

    print("\nDone. See outputs/figures/, outputs/metrics/, outputs/checkpoints/.")


def build_case_figures(case_pool, classes, test_df, threshold=0.5, n_examples=3):
    """Pick a few illustrative cases at coverage=0.5: one where SAR clearly helps (B wrong, C
    right), one where SAR clearly hurts (B right, C wrong), one where both fail."""
    from dataset import load_norm_stats
    import numpy as np

    df_idx = test_df.set_index("patch_id")
    cases_at_50 = [c for c in case_pool if c["coverage"] == 0.5]

    def to_labelset(vec):
        return [classes[i] for i in range(len(classes)) if vec[i] >= threshold]

    def exact_correct(y_true, y_pred_vec):
        pred_bin = (y_pred_vec >= threshold).astype(int)
        return bool((pred_bin == y_true).all())

    sar_helps, sar_hurts, both_fail = [], [], []
    for c in cases_at_50:
        b_ok = exact_correct(c["y_true"], c["prob_b"])
        c_ok = exact_correct(c["y_true"], c["prob_c"])
        if not b_ok and c_ok:
            sar_helps.append(c)
        elif b_ok and not c_ok:
            sar_hurts.append(c)
        elif not b_ok and not c_ok:
            both_fail.append(c)

    print(f"At 50% coverage: SAR helps (B wrong->C right) n={len(sar_helps)}; "
          f"SAR hurts (B right->C wrong) n={len(sar_hurts)}; both fail n={len(both_fail)}")

    import random
    random.seed(SEED)
    chosen = []
    for pool, tag in [(sar_helps, "sar_helps"), (sar_hurts, "sar_hurts"), (both_fail, "both_fail")]:
        if pool:
            chosen.append((tag, random.choice(pool)))

    if not chosen:
        print("No qualitative cases found to plot (unexpected with n>0 test set).")
        return

    from preprocess import CACHE_DIR as _  # noqa: ensure module importable
    cache_dir = ROOT / "outputs" / "cache" / "preprocessed"

    cases_for_fig = []
    for tag, c in chosen:
        pid = c["patch_id"]
        row = df_idx.loc[pid]
        s1_name = row["s1_name"]
        clean_optical = np.load(cache_dir / "S2" / f"{pid}.npy")
        sar = np.load(cache_dir / "S1" / f"{s1_name}.npy")
        # regenerate the same mask realization used in eval (seed_eval=SEED consistent w/ dataset.py)
        idx_in_test = test_df.index[test_df["patch_id"] == pid][0]
        from masking import make_mask, apply_mask
        from dataset import normalize, load_norm_stats
        stats = load_norm_stats()
        rng = np.random.default_rng(SEED * 1_000_003 + idx_in_test)
        mask = make_mask(0.5, rng)
        masked_optical = apply_mask(clean_optical.copy(), mask)

        true_labels = [classes[i] for i in range(len(classes)) if c["y_true"][i] == 1]
        cases_for_fig.append({
            "clean_optical": clean_optical,
            "masked_optical": masked_optical,
            "sar": sar,
            "true_labels": true_labels,
            "pred_b": to_labelset(c["prob_b"]),
            "pred_c": to_labelset(c["prob_c"]),
            "coverage": 0.5,
            "correct_b": exact_correct(c["y_true"], c["prob_b"]),
            "correct_c": exact_correct(c["y_true"], c["prob_c"]),
        })

    plot_case_grid(cases_for_fig, out_name="qualitative_cases_50pct.png")
    print("Saved outputs/figures/qualitative_cases_50pct.png")


if __name__ == "__main__":
    main()
