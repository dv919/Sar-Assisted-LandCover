"""
The broad experiment: 19 classes, 4,200 patches (see select_subset.py / README.md sec. 2.1).
Trains the class-balanced optical-only baseline and three SAR-fusion configurations, then
evaluates all four across cloud-coverage levels with a scientifically-valid metric.

1. **Scientifically-valid metric** (evaluate.compute_valid_class_mask): a naive 19-class macro-F1
   would be silently penalized by 3 classes that are structurally unlearnable/untestable in this
   5-tile subset ("Beaches, dunes, sands" has 0 TRAIN examples; "Marine waters" and "Coastal
   wetlands" have 0 VAL/TEST examples) -- not a modeling failure. macro_f1_valid (16 classes with
   support in all three splits) is reported as primary, alongside the naive macro_f1_all (19
   classes) for transparency.
2. **Class-balanced loss** (class_weights.py): inverse-frequency pos_weight in BCEWithLogitsLoss,
   computed from train-split support only, targeting the real (non-structural) rare-class gap.
   The comparison against unweighted loss (see train_baseline_unweighted.py) is reported as the
   per-class loss-ablation figure.
3. **A second SAR fusion architecture** (models.TwoBranchCNN): separate optical/SAR encoders with
   late feature fusion, compared against early-concatenation fusion -- an actual architecture
   ablation, since the case only *requires* one fusion approach.
4. **A training-coverage ablation**: does training condition C with mask coverage sampled
   uniformly over [0, 75%] generalize better across eval coverage levels than training at a single
   fixed 50%? Tested directly (C-fixed50 vs C-early, evaluated identically). Selection between the
   three configurations uses validation-set performance only (see README.md sec. 7.2).

Also adds **temperature-scaling calibration + Expected Calibration Error** (calibration.py) for
the two headline models (B and the selected SAR-fusion configuration), extending the raw-entropy
uncertainty proxy with a standard, quantifiable metric.

Outputs use a `_v2`/`v2_` filename prefix throughout (a legacy of this project's iterative
development; kept as-is since many other scripts and cached artifacts already reference these
exact names -- see README.md for what each output actually represents).
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from calibration import apply_temperature, expected_calibration_error, fit_temperature
from class_weights import compute_pos_weights
from dataset import BENSubset
from evaluate import compute_metrics, compute_valid_class_mask, predict, predict_logits
from labels import load_classes
from models import SimpleCNN, TwoBranchCNN
from train import train_model
from visualize import (plot_case_grid, plot_multi_condition_vs_coverage, plot_per_class_f1,
                        plot_reliability_diagram)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
METRICS_DIR = ROOT / "outputs" / "metrics"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

COVERAGE_LEVELS = [0.0, 0.25, 0.5, 0.75]
EPOCHS = 10
BATCH_SIZE = 32
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_splits():
    df = pd.read_csv(DATA_DIR / "subset_patches_complete.csv")
    return (df[df["split"] == "train"].reset_index(drop=True),
            df[df["split"] == "validation"].reset_index(drop=True),
            df[df["split"] == "test"].reset_index(drop=True))


def evaluate_condition(model, df, mode, coverage, classes, valid_mask, batch_size=BATCH_SIZE):
    ds = BENSubset(df, mode=mode, coverage=coverage, seed_eval=SEED)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    y_true, y_prob, ids = predict(model, loader, DEVICE)
    metrics = compute_metrics(y_true, y_prob, valid_class_mask=valid_mask)
    return y_true, y_prob, ids, metrics


def main():
    import os
    torch.set_num_threads(os.cpu_count() or 4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_df, val_df, test_df = get_splits()
    classes = load_classes()
    valid_mask, excluded_classes, support = compute_valid_class_mask(train_df, val_df, test_df, classes)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)} classes={len(classes)}")
    print(f"Excluded from macro_f1_valid ({len(excluded_classes)} classes, 0 support in some split): "
          f"{excluded_classes}")
    with open(METRICS_DIR / "v2_valid_class_mask.json", "w") as f:
        json.dump({"excluded_classes": excluded_classes, "support": support,
                   "valid_mask": valid_mask}, f, indent=2)

    pos_weight = compute_pos_weights(train_df, classes)
    print(f"Class pos_weight range: min={pos_weight.min():.2f} max={pos_weight.max():.2f}")
    np.save(METRICS_DIR / "v2_pos_weight.npy", pos_weight)

    common_kwargs = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
                          pos_weight=pos_weight, valid_class_mask=valid_mask)

    # ---------------- Model A2: optical-only, clean, class-weighted loss ----------------
    print("\n=== [Broad experiment] Training Model A2 (optical only, clean, weighted loss) ===")
    train_a = BENSubset(train_df, mode="optical_only", coverage=0.0)
    val_a = BENSubset(val_df, mode="optical_only", coverage=0.0)
    model_a2 = SimpleCNN(in_channels=12, num_classes=len(classes))
    model_a2, hist_a2 = train_model(model_a2, train_a, val_a, log_prefix="[A2] ", **common_kwargs)
    torch.save(model_a2.state_dict(), CKPT_DIR / "model_a_v2.pt")
    with open(METRICS_DIR / "v2_history_a.json", "w") as f:
        json.dump(hist_a2, f, indent=2)

    # ---------------- Model C-early2: early concat fusion, random coverage aug, weighted loss --
    print("\n=== [Broad experiment] Training Model C-early (concat fusion, coverage~U[0,0.75], weighted) ===")
    train_c_early = BENSubset(train_df, mode="fusion", coverage=(0.0, 0.75))
    val_c_early = BENSubset(val_df, mode="fusion", coverage=(0.0, 0.75))
    model_c_early = SimpleCNN(in_channels=14, num_classes=len(classes))
    model_c_early, hist_c_early = train_model(model_c_early, train_c_early, val_c_early,
                                               log_prefix="[C-early] ", **common_kwargs)
    torch.save(model_c_early.state_dict(), CKPT_DIR / "model_c_early_v2.pt")
    with open(METRICS_DIR / "v2_history_c_early.json", "w") as f:
        json.dump(hist_c_early, f, indent=2)

    # ---------------- Model C-late: two-branch late fusion, same regime -----------------------
    print("\n=== [Broad experiment] Training Model C-late (two-branch late fusion, coverage~U[0,0.75]) ===")
    train_c_late = BENSubset(train_df, mode="fusion", coverage=(0.0, 0.75))
    val_c_late = BENSubset(val_df, mode="fusion", coverage=(0.0, 0.75))
    model_c_late = TwoBranchCNN(optical_channels=12, sar_channels=2, num_classes=len(classes))
    model_c_late, hist_c_late = train_model(model_c_late, train_c_late, val_c_late,
                                             log_prefix="[C-late] ", **common_kwargs)
    torch.save(model_c_late.state_dict(), CKPT_DIR / "model_c_late_v2.pt")
    with open(METRICS_DIR / "v2_history_c_late.json", "w") as f:
        json.dump(hist_c_late, f, indent=2)

    # ---------------- Model C-fixed50: same arch as C-early, trained at FIXED 50% coverage ----
    print("\n=== [Broad experiment] Training Model C-fixed50 (concat fusion, coverage FIXED @ 50%) ===")
    train_c_fixed = BENSubset(train_df, mode="fusion", coverage=0.5)
    val_c_fixed = BENSubset(val_df, mode="fusion", coverage=0.5)
    model_c_fixed50 = SimpleCNN(in_channels=14, num_classes=len(classes))
    model_c_fixed50, hist_c_fixed = train_model(model_c_fixed50, train_c_fixed, val_c_fixed,
                                                 log_prefix="[C-fixed50] ", **common_kwargs)
    torch.save(model_c_fixed50.state_dict(), CKPT_DIR / "model_c_fixed50_v2.pt")
    with open(METRICS_DIR / "v2_history_c_fixed50.json", "w") as f:
        json.dump(hist_c_fixed, f, indent=2)

    # ---------------- Evaluation sweep across coverage levels on TEST split -------------------
    print("\n=== [Broad experiment] Evaluating A2/B/C-early/C-late/C-fixed50 across coverage levels ===")
    rows = []
    per_class_by_cov = {}
    case_pool = {}  # cov -> list of per-model dicts, for qualitative figure

    c_models = {"C-early": model_c_early, "C-late": model_c_late, "C-fixed50": model_c_fixed50}

    for cov in COVERAGE_LEVELS:
        y_true_b, y_prob_b, ids_b, metrics_b = evaluate_condition(
            model_a2, test_df, "optical_masked", cov, classes, valid_mask)
        cond_label = "A" if cov == 0.0 else "B"
        rows.append({"condition": cond_label, "coverage": cov,
                     **{k: v for k, v in metrics_b.items() if k != "per_class_f1"}})
        per_class_by_cov.setdefault(cov, {})["B"] = metrics_b["per_class_f1"]

        cov_probs = {"B": (y_true_b, y_prob_b, ids_b)}
        for name, model in c_models.items():
            y_true_c, y_prob_c, ids_c, metrics_c = evaluate_condition(
                model, test_df, "fusion", cov, classes, valid_mask)
            assert ids_c == ids_b, "all conditions must share the same per-sample mask realization"
            rows.append({"condition": name, "coverage": cov,
                         **{k: v for k, v in metrics_c.items() if k != "per_class_f1"}})
            per_class_by_cov[cov][name] = metrics_c["per_class_f1"]
            cov_probs[name] = (y_true_c, y_prob_c, ids_c)

        print(f"coverage={cov:.2f}  " + "  |  ".join(
            f"{k}: f1_valid={compute_metrics(v[0], v[1], valid_class_mask=valid_mask)['macro_f1_valid']:.4f}"
            for k, v in cov_probs.items()))
        case_pool[cov] = cov_probs

    results_df = pd.DataFrame(rows)
    results_df.to_csv(METRICS_DIR / "results_v2.csv", index=False)
    with open(METRICS_DIR / "v2_per_class_f1_by_coverage.json", "w") as f:
        json.dump(per_class_by_cov, f, indent=2)
    print("\nSaved outputs/metrics/results_v2.csv")

    # pick the best C variant by mean macro_f1_valid across coverage>0 (i.e. under degradation)
    c_scores = {}
    for name in c_models:
        sub = results_df[(results_df["condition"] == name) & (results_df["coverage"] > 0)]
        c_scores[name] = sub["macro_f1_valid"].mean()
    best_c_name = max(c_scores, key=c_scores.get)
    print(f"Best C variant under degradation (mean macro_f1_valid, coverage>0): "
          f"{c_scores} -> best={best_c_name}")

    # ---------------- Figures: multi-condition curves ----------------
    a_ref_valid = results_df[results_df["condition"] == "A"]["macro_f1_valid"].iloc[0]
    a_ref_all = results_df[results_df["condition"] == "A"]["macro_f1_all"].iloc[0]
    plot_multi_condition_vs_coverage(results_df, metric="macro_f1_valid",
                                      out_name="macro_f1_valid_fusion_comparison.png",
                                      conditions=["B", "C-early", "C-late", "C-fixed50"],
                                      a_reference=a_ref_valid)
    plot_multi_condition_vs_coverage(results_df, metric="macro_f1_all",
                                      out_name="macro_f1_all_fusion_comparison.png",
                                      conditions=["B", "C-early", "C-late", "C-fixed50"],
                                      a_reference=a_ref_all)

    # ---------------- Per-class comparison: unweighted-loss baseline vs class-balanced loss -----
    # requires train_baseline_unweighted.py to have been run first (produces this file); see
    # README.md sec. 12 for the full reproduction order.
    unweighted_pc_path = METRICS_DIR / "per_class_f1_by_coverage.json"
    if unweighted_pc_path.exists():
        with open(unweighted_pc_path) as f:
            unweighted_pc = json.load(f)
        standard_b = unweighted_pc.get("0.5", {}).get("B", {})
        weighted_b = per_class_by_cov[0.5]["B"]
        if standard_b:
            plot_per_class_f1(standard_b, weighted_b, "50%", out_name="per_class_f1_loss_ablation.png",
                               label_b="Standard loss", label_c="Class-balanced loss",
                               color_b="#B0B0B0", color_c="#DD8452",
                               title="Effect of class-balanced loss on per-class F1 (optical-only model, 50% coverage)",
                               excluded_classes=set(excluded_classes))
    else:
        print(f"NOTE: {unweighted_pc_path} not found -- run train_baseline_unweighted.py first "
              f"to also generate the class-balanced-loss ablation figure.")

    plot_per_class_f1(per_class_by_cov[0.5]["B"], per_class_by_cov[0.5][best_c_name], "50%",
                       out_name="per_class_f1_broad_experiment_50pct.png",
                       label_b="B: degraded optical only", label_c="C: degraded optical + SAR",
                       title="Per-class F1 at 50% coverage: optical-only vs. SAR-fusion (broad experiment)",
                       excluded_classes=set(excluded_classes))

    # ---------------- Calibration: temperature scaling + ECE for B and best-C, at 50% coverage --
    print(f"\n=== [Broad experiment] Calibration analysis (B and {best_c_name}, @ 50% coverage) ===")
    val_ds_b = BENSubset(val_df, mode="optical_masked", coverage=0.5, seed_eval=SEED + 1)
    val_loader_b = torch.utils.data.DataLoader(val_ds_b, batch_size=BATCH_SIZE, shuffle=False)
    y_val_b, logits_val_b, _ = predict_logits(model_a2, val_loader_b, DEVICE)
    T_b = fit_temperature(logits_val_b, y_val_b)

    val_ds_c = BENSubset(val_df, mode="fusion", coverage=0.5, seed_eval=SEED + 1)
    val_loader_c = torch.utils.data.DataLoader(val_ds_c, batch_size=BATCH_SIZE, shuffle=False)
    y_val_c, logits_val_c, _ = predict_logits(c_models[best_c_name], val_loader_c, DEVICE)
    T_c = fit_temperature(logits_val_c, y_val_c)
    print(f"Fitted temperature: T_B={T_b:.3f}  T_{best_c_name}={T_c:.3f}")

    test_ds_b = BENSubset(test_df, mode="optical_masked", coverage=0.5, seed_eval=SEED)
    test_loader_b = torch.utils.data.DataLoader(test_ds_b, batch_size=BATCH_SIZE, shuffle=False)
    y_test_b, logits_test_b, _ = predict_logits(model_a2, test_loader_b, DEVICE)
    probs_before_b = 1.0 / (1.0 + np.exp(-logits_test_b))
    probs_after_b = apply_temperature(logits_test_b, T_b)
    ece_before_b = expected_calibration_error(probs_before_b, y_test_b)
    ece_after_b = expected_calibration_error(probs_after_b, y_test_b)

    test_ds_c = BENSubset(test_df, mode="fusion", coverage=0.5, seed_eval=SEED)
    test_loader_c = torch.utils.data.DataLoader(test_ds_c, batch_size=BATCH_SIZE, shuffle=False)
    y_test_c, logits_test_c, _ = predict_logits(c_models[best_c_name], test_loader_c, DEVICE)
    probs_before_c = 1.0 / (1.0 + np.exp(-logits_test_c))
    probs_after_c = apply_temperature(logits_test_c, T_c)
    ece_before_c = expected_calibration_error(probs_before_c, y_test_c)
    ece_after_c = expected_calibration_error(probs_after_c, y_test_c)

    print(f"B:  ECE before={ece_before_b['ece']:.4f}  after={ece_after_b['ece']:.4f}  (T={T_b:.3f})")
    print(f"{best_c_name}: ECE before={ece_before_c['ece']:.4f}  after={ece_after_c['ece']:.4f}  (T={T_c:.3f})")

    plot_reliability_diagram(ece_before_b["bins"], ece_after_b["bins"],
                              ece_before_b["ece"], ece_after_b["ece"],
                              title="Optical-only model (B) at 50% simulated cloud coverage",
                              out_name="reliability_B.png")
    plot_reliability_diagram(ece_before_c["bins"], ece_after_c["bins"],
                              ece_before_c["ece"], ece_after_c["ece"],
                              title="SAR-fusion model at 50% simulated cloud coverage",
                              out_name="reliability_fusion.png")

    calibration_summary = {
        "best_c_name": best_c_name, "c_scores_mean_valid_f1_under_degradation": c_scores,
        "T_B": T_b, f"T_{best_c_name}": T_c,
        "ece_B_before": ece_before_b["ece"], "ece_B_after": ece_after_b["ece"],
        f"ece_{best_c_name}_before": ece_before_c["ece"], f"ece_{best_c_name}_after": ece_after_c["ece"],
    }
    with open(METRICS_DIR / "v2_calibration_summary.json", "w") as f:
        json.dump(calibration_summary, f, indent=2)

    # ---------------- Qualitative cases with the best C model @ 50% coverage ----------------
    build_case_figures_v2(case_pool[0.5], classes, test_df, best_c_name)

    print("\n[Broad experiment] Done. See outputs/figures/v2_*.png, outputs/metrics/*_v2*, "
          "outputs/checkpoints/*_v2.pt.")


def build_case_figures_v2(cov_probs, classes, test_df, best_c_name, threshold=0.5):
    from masking import make_mask, apply_mask
    from dataset import normalize, load_norm_stats

    cache_dir = ROOT / "outputs" / "cache" / "preprocessed"
    df_idx = test_df.set_index("patch_id")
    y_true, y_prob_b, ids_b = cov_probs["B"]
    _, y_prob_c, ids_c = cov_probs[best_c_name]
    assert ids_b == ids_c

    def to_labelset(vec):
        return [classes[i] for i in range(len(classes)) if vec[i] >= threshold]

    def exact_correct(yt, yp):
        return bool(((yp >= threshold).astype(int) == yt).all())

    sar_helps, sar_hurts, both_fail = [], [], []
    for i, pid in enumerate(ids_b):
        b_ok = exact_correct(y_true[i], y_prob_b[i])
        c_ok = exact_correct(y_true[i], y_prob_c[i])
        entry = {"idx": i, "patch_id": pid}
        if not b_ok and c_ok:
            sar_helps.append(entry)
        elif b_ok and not c_ok:
            sar_hurts.append(entry)
        elif not b_ok and not c_ok:
            both_fail.append(entry)

    print(f"[v2, {best_c_name}] At 50% coverage: SAR helps n={len(sar_helps)}; "
          f"SAR hurts n={len(sar_hurts)}; both fail n={len(both_fail)}")

    import random
    random.seed(SEED)
    chosen = []
    for pool in [sar_helps, sar_hurts, both_fail]:
        if pool:
            chosen.append(random.choice(pool))
    if not chosen:
        return

    cases_for_fig = []
    for entry in chosen:
        i, pid = entry["idx"], entry["patch_id"]
        row = df_idx.loc[pid]
        s1_name = row["s1_name"]
        clean_optical = np.load(cache_dir / "S2" / f"{pid}.npy")
        sar = np.load(cache_dir / "S1" / f"{s1_name}.npy")
        idx_in_test = test_df.index[test_df["patch_id"] == pid][0]
        rng = np.random.default_rng(SEED * 1_000_003 + idx_in_test)
        mask = make_mask(0.5, rng)
        masked_optical = apply_mask(clean_optical.copy(), mask)

        true_labels = [classes[j] for j in range(len(classes)) if y_true[i][j] == 1]
        cases_for_fig.append({
            "clean_optical": clean_optical, "masked_optical": masked_optical, "sar": sar,
            "true_labels": true_labels,
            "pred_b": to_labelset(y_prob_b[i]), "pred_c": to_labelset(y_prob_c[i]),
            "coverage": 0.5,
            "correct_b": exact_correct(y_true[i], y_prob_b[i]),
            "correct_c": exact_correct(y_true[i], y_prob_c[i]),
        })

    plot_case_grid(cases_for_fig, out_name="qualitative_cases_broad_experiment.png")
    print(f"Saved outputs/figures/qualitative_cases_broad_experiment.png (selected configuration = {best_c_name})")


if __name__ == "__main__":
    main()
