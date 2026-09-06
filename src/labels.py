"""Shared label vocabulary: fixed alphabetical order over the 19 BigEarthNet-19 classes
actually present in our subset, so every script encodes labels the same way."""
import json
import pathlib

import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
CLASSES_PATH = DATA_DIR / "classes.json"


def build_and_save_classes(subset_csv: pathlib.Path = None) -> list:
    subset_csv = subset_csv or (DATA_DIR / "subset_patches.csv")
    df = pd.read_csv(subset_csv)
    # `labels` column is stored as a JSON-encoded list string (see select_subset._jsonify_labels).
    labels_col = df["labels"].apply(json.loads) if isinstance(df["labels"].iloc[0], str) else df["labels"]
    all_labels = sorted({lbl for row in labels_col for lbl in row})
    with open(CLASSES_PATH, "w") as f:
        json.dump(all_labels, f, indent=2)
    return all_labels


def load_classes(classes_path: pathlib.Path = None) -> list:
    classes_path = classes_path or CLASSES_PATH
    if not classes_path.exists():
        return build_and_save_classes()
    with open(classes_path) as f:
        return json.load(f)


def encode_labels(label_list, classes=None):
    """label_list: python list of class-name strings -> multi-hot vector (list[int]).
    Labels not present in `classes` are silently ignored (the deep experiment restricts to an
    8-class vocabulary, but a patch's original annotation can carry other BigEarthNet labels too
    -- those are simply not a prediction target, not an error)."""
    classes = classes or load_classes()
    idx = {c: i for i, c in enumerate(classes)}
    vec = [0] * len(classes)
    for lbl in label_list:
        if lbl in idx:
            vec[idx[lbl]] = 1
    return vec


if __name__ == "__main__":
    classes = build_and_save_classes()
    print(f"{len(classes)} classes:")
    for c in classes:
        print(" -", c)
