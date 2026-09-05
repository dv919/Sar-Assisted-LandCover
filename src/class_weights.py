"""Class-balanced loss weighting: standard inverse-frequency pos_weight for BCEWithLogitsLoss,
computed from the TRAIN split only (never val/test, to avoid leaking eval-set class balance into
training). Capped to avoid one or two near-empty classes producing an unstable, huge weight."""
import numpy as np

from evaluate import class_support
from labels import load_classes

MAX_POS_WEIGHT = 15.0


def compute_pos_weights(train_df, classes=None, n_train=None, max_weight=MAX_POS_WEIGHT):
    classes = classes or load_classes()
    n_train = n_train or len(train_df)
    support = class_support(train_df, classes)
    weights = []
    for c in classes:
        pos = support[c]
        neg = n_train - pos
        if pos == 0:
            # no positive examples to weight at all (see README: "Beaches, dunes, sands" has
            # zero train support in this subset) -- weight is moot since it never multiplies a
            # real loss term, but keep it finite and neutral rather than inf/nan.
            w = 1.0
        else:
            w = min(neg / pos, max_weight)
        weights.append(w)
    return np.array(weights, dtype=np.float32)


if __name__ == "__main__":
    import pandas as pd
    import pathlib
    DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
    train_df = pd.read_csv(DATA_DIR / "subset_patches_complete.csv")
    train_df = train_df[train_df["split"] == "train"]
    classes = load_classes()
    w = compute_pos_weights(train_df, classes)
    for c, wi in sorted(zip(classes, w), key=lambda t: -t[1]):
        print(f"{c}: {wi:.2f}")
