"""
Round 3: fewer classes, more examples per class.

Rounds 1-2 stratified a fixed 4,200-patch budget across all 19 BigEarthNet classes present in
our 5 tiles, including some with only 29-51 total examples in the whole candidate pool -- which
directly caused round 2's "3 structurally unlearnable/unscoreable classes" finding. This script
instead restricts to the 8 best-supported classes (verified to have solid examples in ALL three
splits) and takes a much larger sample from the SAME already-scanned candidate pool
(candidate_patches.csv, 13,932 rows, same 5 tiles / S1A-only filter as rounds 1-2 -- no new tile
selection, no new network cost beyond re-streaming the same byte range in extract_subset.py).

These 8 classes cover 97.8% of the entire candidate pool (verified directly), so dropping to 8
classes costs almost no data -- it just stops reserving budget for classes that were always going
to be data-starved.
"""
import json
import pathlib

import numpy as np
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
TARGET_TOTAL = 8000
RNG_SEED = 42

TOP8_CLASSES = [
    "Pastures",
    "Coniferous forest",
    "Mixed forest",
    "Arable land",
    "Transitional woodland, shrub",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Broad-leaved forest",
]


def main():
    cand = pd.read_csv(DATA_DIR / "candidate_patches.csv")
    labs = cand["labels"].apply(json.loads)
    has_top = labs.apply(lambda l: any(c in l for c in TOP8_CLASSES))
    filtered = cand[has_top].copy()
    print(f"Candidates containing >=1 of the 8 target classes: {len(filtered)} / {len(cand)} "
          f"({len(filtered)/len(cand)*100:.1f}%)")

    # restrict each patch's label vector to only the 8 target classes (other BigEarthNet labels
    # the patch may also carry are simply not a prediction target -- not dropped, not treated as
    # negative evidence, just outside this round's scope; see labels.encode_labels)
    filtered["labels"] = labs[has_top].apply(lambda l: json.dumps([c for c in l if c in TOP8_CLASSES]))

    split_counts = filtered["split"].value_counts()
    target_per_split = {s: int(round(TARGET_TOTAL * c / len(filtered))) for s, c in split_counts.items()}
    print("Candidate split counts:", split_counts.to_dict())
    print("Target per split:", target_per_split)

    rng = np.random.default_rng(RNG_SEED)
    parts = []
    for split_name, n_target in target_per_split.items():
        split_df = filtered[filtered["split"] == split_name]
        n = min(n_target, len(split_df))
        parts.append(split_df.sample(n=n, random_state=RNG_SEED))
    subset = pd.concat(parts).reset_index(drop=True)
    print(f"Final v3 subset size: {len(subset)}")
    print("Subset split counts:\n", subset["split"].value_counts())

    subset.to_csv(DATA_DIR / "subset_patches_v3.csv", index=False)

    with open(DATA_DIR / "classes_v3.json", "w") as f:
        json.dump(TOP8_CLASSES, f, indent=2)

    # per-class counts in the final v3 subset, per split -- the whole point of this round
    report = {}
    for split_name in ["train", "validation", "test"]:
        sub = subset[subset["split"] == split_name]
        from collections import Counter
        cc = Counter()
        for row in sub["labels"].apply(json.loads):
            cc.update(row)
        report[split_name] = dict(cc)
        print(f"\n--- v3 {split_name} ({len(sub)} patches) per-class counts ---")
        for cls in TOP8_CLASSES:
            print(f"  {cls}: {cc.get(cls, 0)}")

    with open(DATA_DIR / "v3_class_support_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
