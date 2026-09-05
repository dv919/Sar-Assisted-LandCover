"""Prediction + metrics utilities shared by training (for validation) and the final B-vs-C sweep."""
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from labels import load_classes


@torch.no_grad()
def predict(model, loader, device="cpu"):
    """Returns (y_true, y_prob, ids) -- probabilities (post-sigmoid)."""
    model.eval()
    all_true, all_prob, all_ids = [], [], []
    for x, y, ids in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy()
        all_true.append(y.numpy())
        all_prob.append(prob)
        all_ids.extend(ids)
    return np.concatenate(all_true), np.concatenate(all_prob), all_ids


@torch.no_grad()
def predict_logits(model, loader, device="cpu"):
    """Returns (y_true, logits, ids) -- raw pre-sigmoid logits, needed for temperature scaling."""
    model.eval()
    all_true, all_logits, all_ids = [], [], []
    for x, y, ids in loader:
        x = x.to(device)
        logits = model(x)
        all_true.append(y.numpy())
        all_logits.append(logits.cpu().numpy())
        all_ids.extend(ids)
    return np.concatenate(all_true), np.concatenate(all_logits), all_ids


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5,
                     valid_class_mask=None) -> dict:
    """
    valid_class_mask: optional boolean array/list over the 19 classes. When given, an additional
    `macro_f1_valid` / `macro_precision_valid` / `macro_recall_valid` is computed over only the
    classes with a mask value of True -- classes that have zero examples in a given split can't be
    learned (0 train support) or can't be scored meaningfully (0 eval-split support), and folding
    them into a naive 19-class macro average penalizes the model for something no model could fix.
    The naive all-19-class numbers are still returned (as *_all) for transparency/comparability.
    """
    y_pred = (y_prob >= threshold).astype(int)
    classes = load_classes()

    macro_f1_all = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    macro_precision_all = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall_all = recall_score(y_true, y_pred, average="macro", zero_division=0)
    hamming_acc = float((y_true == y_pred).mean())  # per-label accuracy averaged over labels+samples
    exact_match = float((y_true == y_pred).all(axis=1).mean())  # fraction of samples fully correct

    # mean predictive entropy of the sigmoid outputs -- a simple uncertainty proxy (per REQUIREMENTS #10)
    eps = 1e-7
    p = np.clip(y_prob, eps, 1 - eps)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    mean_entropy = float(entropy.mean())

    result = {
        "macro_f1_all": float(macro_f1_all),
        "micro_f1": float(micro_f1),
        "macro_precision_all": float(macro_precision_all),
        "macro_recall_all": float(macro_recall_all),
        "hamming_acc": hamming_acc,
        "exact_match_acc": exact_match,
        "mean_entropy": mean_entropy,
        "per_class_f1": {c: f for c, f in zip(classes, per_class_f1)},
    }
    # backward-compat alias used by round-1 code/plots
    result["macro_f1"] = result["macro_f1_all"]

    if valid_class_mask is not None:
        mask = np.asarray(valid_class_mask, dtype=bool)
        yt_v, yp_v = y_true[:, mask], y_pred[:, mask]
        result["macro_f1_valid"] = float(f1_score(yt_v, yp_v, average="macro", zero_division=0))
        result["macro_precision_valid"] = float(precision_score(yt_v, yp_v, average="macro", zero_division=0))
        result["macro_recall_valid"] = float(recall_score(yt_v, yp_v, average="macro", zero_division=0))
        result["n_valid_classes"] = int(mask.sum())

    return result


def class_support(df, classes=None):
    """Per-class positive-example count in a dataframe with a JSON-encoded 'labels' column."""
    import json
    classes = classes or load_classes()
    counts = {c: 0 for c in classes}
    for row in df["labels"]:
        labs = json.loads(row) if isinstance(row, str) else row
        for l in labs:
            if l in counts:
                counts[l] += 1
    return counts


def compute_valid_class_mask(train_df, val_df, test_df, classes=None):
    """A class is 'valid' for evaluation only if it has at least one positive example in ALL
    THREE splits: zero train support means no model could ever learn it; zero val/test support
    means it can never be scored (F1 silently defaults to 0 via zero_division, misleadingly
    counted as a failure). See README §5 for the concrete classes this excludes in our subset."""
    classes = classes or load_classes()
    train_sup = class_support(train_df, classes)
    val_sup = class_support(val_df, classes)
    test_sup = class_support(test_df, classes)
    mask = [train_sup[c] > 0 and val_sup[c] > 0 and test_sup[c] > 0 for c in classes]
    excluded = [c for c, m in zip(classes, mask) if not m]
    return mask, excluded, {"train": train_sup, "validation": val_sup, "test": test_sup}
