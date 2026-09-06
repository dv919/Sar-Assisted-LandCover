"""
Temperature scaling (Guo et al. 2017) + Expected Calibration Error (ECE) for the multi-label
setting, extending REQUIREMENTS.md #10's "how you reason about uncertainty" beyond a raw
mean-entropy proxy with a standard, quantifiable calibration metric.

Multi-label extension used here: each of the 19 (sample, class) sigmoid outputs is treated as an
independent binary probabilistic prediction; a single learned scalar temperature T rescales all
logits (logits / T) before the sigmoid, fit by minimizing BCE on the VALIDATION set (never test).
"""
import numpy as np
import torch
import torch.nn as nn


def fit_temperature(logits: np.ndarray, y_true: np.ndarray, max_iter: int = 100, lr: float = 0.05):
    """logits, y_true: (N, C) arrays from the VALIDATION split. Returns a single float T > 0."""
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    y_t = torch.as_tensor(y_true, dtype=torch.float32)
    log_T = torch.zeros(1, requires_grad=True)  # optimize in log-space so T stays positive
    opt = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T)
        loss = criterion(logits_t / T, y_t)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(torch.exp(log_T).item())
    return T


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits / T))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> dict:
    """Standard ECE over all (sample, class) predictions pooled together: bin by the model's own
    confidence in its binary call (max(p, 1-p)), compare bin-average confidence to bin-average
    accuracy. Returns overall ECE plus the per-bin table (for reliability diagrams)."""
    p = probs.ravel()
    y = y_true.ravel()
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    confidence = np.where(pred == 1, p, 1 - p)

    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)  # confidence is always in [0.5, 1]
    bin_ids = np.digitize(confidence, bin_edges[1:-1], right=True)

    ece = 0.0
    bins = []
    n = len(p)
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin": b, "count": 0, "avg_confidence": None, "avg_accuracy": None})
            continue
        avg_conf = float(confidence[mask].mean())
        avg_acc = float(correct[mask].mean())
        ece += (count / n) * abs(avg_acc - avg_conf)
        bins.append({"bin": b, "count": count, "avg_confidence": avg_conf, "avg_accuracy": avg_acc,
                     "lo": float(bin_edges[b]), "hi": float(bin_edges[b + 1])})

    return {"ece": float(ece), "bins": bins, "n_predictions": n}
