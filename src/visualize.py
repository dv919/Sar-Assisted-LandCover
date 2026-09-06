"""Figures for the README: metric-vs-coverage curves, per-class F1 bars, and qualitative
success/failure examples comparing condition B (degraded optical only) vs condition C
(degraded optical + SAR) at matched mask realizations."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# brand-neutral, colorblind-safe qualitative palette
COLOR_A = "#4C72B0"  # optical only (clean)
COLOR_B = "#DD8452"  # degraded optical only
COLOR_C = "#55A868"  # degraded optical + SAR
GRID = "#D9D9D9"


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_metric_vs_coverage(results_df, metric="macro_f1", out_name=None):
    """results_df has columns: condition ('A'|'B'|'C'), coverage (float), <metric>."""
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    for cond, color, label in [("B", COLOR_B, "B: degraded optical only"),
                                ("C", COLOR_C, "C: degraded optical + SAR")]:
        sub = results_df[results_df["condition"] == cond].sort_values("coverage")
        ax.plot(sub["coverage"] * 100, sub[metric], marker="o", color=color, label=label, linewidth=2)
    a_row = results_df[results_df["condition"] == "A"]
    if len(a_row):
        ax.axhline(a_row[metric].iloc[0], color=COLOR_A, linestyle="--", linewidth=1.5,
                   label="A: optical only (clean)")
    _style_axes(ax)
    ax.set_xlabel("Simulated cloud coverage (%)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_', ' ').title()} vs. cloud coverage")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out_name = out_name or f"metric_vs_coverage_{metric}.png"
    fig.savefig(FIG_DIR / out_name)
    plt.close(fig)
    return FIG_DIR / out_name


MULTI_PALETTE = {
    "B": ("#DD8452", "B: degraded optical only"),
    "C-early": ("#55A868", "C: early fusion (concat)"),
    "C-late": ("#4C72B0", "C: late fusion (two-branch)"),
    "C-fixed50": ("#8172B2", "C: early fusion, trained @ fixed 50%"),
}


def plot_multi_condition_vs_coverage(results_df, metric="macro_f1_valid", out_name=None,
                                      conditions=None, a_reference=None):
    """results_df has columns: condition (any key in MULTI_PALETTE, or 'A'), coverage, <metric>.
    Lets round 2 plot several C variants (different fusion architectures / training regimes)
    against B on one chart, not just a single B-vs-C pair."""
    conditions = conditions or [c for c in MULTI_PALETTE if c in set(results_df["condition"])]
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    for cond in conditions:
        color, label = MULTI_PALETTE[cond]
        sub = results_df[results_df["condition"] == cond].sort_values("coverage")
        if len(sub) == 0:
            continue
        ax.plot(sub["coverage"] * 100, sub[metric], marker="o", color=color, label=label, linewidth=2)
    if a_reference is not None:
        ax.axhline(a_reference, color=COLOR_A, linestyle="--", linewidth=1.5, label="A: optical only (clean)")
    _style_axes(ax)
    ax.set_xlabel("Simulated cloud coverage (%)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_', ' ').title()} vs. cloud coverage — fusion configuration comparison")
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    fig.tight_layout()
    out_name = out_name or f"v2_multi_{metric}.png"
    fig.savefig(FIG_DIR / out_name)
    plt.close(fig)
    return FIG_DIR / out_name


def plot_reliability_diagram(bins_before, bins_after, ece_before, ece_after, title, out_name):
    """bins_*: the 'bins' list from calibration.expected_calibration_error."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=150, sharey=True)
    for ax, bins, ece, subtitle in [(axes[0], bins_before, ece_before, "Before calibration"),
                                     (axes[1], bins_after, ece_after, "After temperature scaling")]:
        confs = [b["avg_confidence"] for b in bins if b["count"] > 0]
        accs = [b["avg_accuracy"] for b in bins if b["count"] > 0]
        ax.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color=GRID, linewidth=1.5, zorder=1)
        ax.bar(confs, accs, width=0.04, color=COLOR_C, alpha=0.85, edgecolor="white", zorder=2)
        _style_axes(ax)
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(0.45, 1.02)
        ax.set_xlabel("Confidence")
        ax.set_title(f"{subtitle}\nECE = {ece:.4f}", fontsize=10)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name)
    plt.close(fig)
    return FIG_DIR / out_name


def plot_per_class_f1(per_class_dict_b, per_class_dict_c, coverage_label, out_name=None,
                       label_b="B: degraded optical", label_c="C: degraded + SAR",
                       color_b=None, color_c=None, title=None, excluded_classes=None):
    """excluded_classes: optional set of class names to mark (not drop) with a hatch + note,
    e.g. classes with zero train/eval support in this subset (see evaluate.compute_valid_class_mask)."""
    classes = list(per_class_dict_b.keys())
    x = np.arange(len(classes))
    width = 0.38
    color_b = color_b or COLOR_B
    color_c = color_c or COLOR_C
    excluded_classes = excluded_classes or set()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bars_b = ax.bar(x - width / 2, [per_class_dict_b[c] for c in classes], width, color=color_b, label=label_b)
    bars_c = ax.bar(x + width / 2, [per_class_dict_c[c] for c in classes], width, color=color_c, label=label_c)
    for i, c in enumerate(classes):
        if c in excluded_classes:
            bars_b[i].set_hatch("//")
            bars_c[i].set_hatch("//")
            bars_b[i].set_alpha(0.35)
            bars_c[i].set_alpha(0.35)
    _style_axes(ax)
    ax.set_xticks(x)
    tick_labels = [f"{c}*" if c in excluded_classes else c for c in classes]
    ax.set_xticklabels(tick_labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("F1")
    ax.set_title(title or f"Per-class F1 at {coverage_label} simulated cloud coverage")
    handles, labels_ = ax.get_legend_handles_labels()
    if excluded_classes:
        labels_.append("* zero train and/or eval support in this subset (see README §5)")
        import matplotlib.patches as mpatches
        handles.append(mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", alpha=0.35))
    ax.legend(handles, labels_, frameon=False, fontsize=8)
    fig.tight_layout()
    out_name = out_name or f"per_class_f1_{coverage_label}.png"
    fig.savefig(FIG_DIR / out_name)
    plt.close(fig)
    return FIG_DIR / out_name


def s2_to_rgb(optical_raw):
    """optical_raw: (12,H,W) in S2 L2A reflectance*10000 convention, band order matches
    preprocess.S2_BAND_ORDER = [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12].
    B04=idx3 (Red), B03=idx2 (Green), B02=idx1 (Blue)."""
    rgb = optical_raw[[3, 2, 1]].transpose(1, 2, 0)
    rgb = np.clip(rgb / 2500.0, 0, 1)  # simple stretch, typical BigEarthNet reflectance scale
    return rgb


def sar_to_gray(sar_db, band_idx=0):
    """sar_db: (2,H,W) in dB (VV, VH). Return a displayable single-band grayscale image."""
    band = sar_db[band_idx]
    lo, hi = np.percentile(band, [2, 98])
    return np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)


def plot_case_grid(cases, out_name):
    """cases: list of dicts with keys:
       clean_optical (12,H,W), masked_optical (12,H,W), sar (2,H,W), true_labels (list[str]),
       pred_b (list[str]), pred_c (list[str]), coverage (float), correct_b (bool), correct_c (bool)
    Renders a row per case: [clean RGB | masked RGB | SAR VV] with a text panel of labels/preds."""
    n = len(cases)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), dpi=150)
    if n == 1:
        axes = axes[None, :]
    for i, case in enumerate(cases):
        axes[i, 0].imshow(s2_to_rgb(case["clean_optical"]))
        axes[i, 0].set_title("Clean optical (RGB)" if i == 0 else "", fontsize=9)
        axes[i, 1].imshow(s2_to_rgb(case["masked_optical"]))
        axes[i, 1].set_title(f"Degraded optical ({case['coverage']*100:.0f}% masked)" if i == 0 else "", fontsize=9)
        axes[i, 2].imshow(sar_to_gray(case["sar"]), cmap="gray")
        axes[i, 2].set_title("SAR (VV)" if i == 0 else "", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")
        b_mark = "✓" if case["correct_b"] else "✗"
        c_mark = "✓" if case["correct_c"] else "✗"
        caption = (f"True: {', '.join(case['true_labels'])}\n"
                   f"B (optical only) {b_mark}: {', '.join(case['pred_b']) or '(none)'}\n"
                   f"C (optical+SAR) {c_mark}: {', '.join(case['pred_c']) or '(none)'}")
        axes[i, 2].text(1.05, 0.5, caption, transform=axes[i, 2].transAxes,
                         fontsize=7.5, va="center", ha="left", wrap=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, bbox_inches="tight")
    plt.close(fig)
    return FIG_DIR / out_name
