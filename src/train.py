"""Training loop shared by conditions A (clean optical) and C (masked optical + SAR fusion)."""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluate import compute_metrics, predict


def train_model(model, train_ds, val_ds, epochs=8, lr=1e-3, batch_size=32,
                 device="cpu", num_workers=0, log_prefix="", pos_weight=None,
                 valid_class_mask=None, classes=None):
    """
    pos_weight: optional per-class tensor for BCEWithLogitsLoss (class-balanced loss). See
    class_weights.py for how this is derived from train-split label frequency.
    valid_class_mask: optional boolean mask (see evaluate.compute_valid_class_mask) -- when given,
    checkpoint selection uses macro_f1_valid (excludes classes with 0 examples in some split)
    instead of the naive all-19-class macro_f1, so model selection isn't skewed by structurally
    unlearnable/untestable classes.
    classes: explicit class list matching the model's output dimension -- required whenever the
    vocabulary isn't the default 19 (the deep experiment's 8-class run), otherwise compute_metrics's
    per-class F1 dict silently mismatches class names to columns.
    """
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pw = torch.as_tensor(pos_weight, dtype=torch.float32, device=device) if pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    select_key = "macro_f1_valid" if valid_class_mask is not None else "macro_f1"

    best_state = None
    best_score = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1
        sched.step()

        y_true, y_prob, _ = predict(model, val_loader, device)
        metrics = compute_metrics(y_true, y_prob, valid_class_mask=valid_class_mask, classes=classes)
        elapsed = time.time() - t0
        print(f"{log_prefix}epoch {epoch}/{epochs} train_loss={running_loss/max(n_batches,1):.4f} "
              f"val_{select_key}={metrics[select_key]:.4f} val_microF1={metrics['micro_f1']:.4f} "
              f"val_hamming_acc={metrics['hamming_acc']:.4f} ({elapsed:.0f}s)")
        history.append({"epoch": epoch, "train_loss": running_loss / max(n_batches, 1), **metrics})

        if metrics[select_key] > best_score:
            best_score = metrics[select_key]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
