"""
Recompute (in code, not from memory) the subset-selection plan described in DATA_ACCESS.md:

1. Load both metadata parquet files, combine, derive S2/S1 "product name" from patch_id/s1_name.
2. Confirm archive ordering assumption: S2 entries are alphabetical by product name; S1 entries
   are alphabetical by product name (S1A block then S1B block).
3. Take the first N_S2_PRODUCTS alphabetically-earliest S2 products.
4. Within those, keep only patches whose paired S1 scene is an S1A product (cheap to stream).
5. From that candidate set, draw a stratified training-sized sample (by split, trying to keep
   every one of the 19 classes represented) so we don't have to extract/preprocess/train on
   the full candidate set on CPU-only hardware.

Writes:
  data/candidate_patches.csv   -- full candidate set (S1A-paired patches from first N S2 products)
  data/subset_patches.csv      -- the smaller stratified sample actually used for training/eval
  data/selection_report.json   -- summary stats for the README
"""
import json
import pathlib

import numpy as np
import pandas as pd


def _jsonify_labels(df: pd.DataFrame) -> pd.DataFrame:
    """pandas serializes list-cells to CSV via numpy's str() (space-separated, no commas --
    e.g. "['a' 'b']"), which is NOT valid to parse back. Force proper JSON so every downstream
    reader can just json.loads() the column."""
    df = df.copy()
    df["labels"] = df["labels"].apply(lambda lbls: json.dumps(list(lbls)))
    return df

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
N_S2_PRODUCTS = 5          # how many alphabetically-earliest S2 products to draw from
TARGET_SUBSET_SIZE = 4200  # ~3000 train / 600 val / 600 test target
RNG_SEED = 42


def s2_product_name(patch_id: str) -> str:
    """S2 archive top-level folder = patch_id minus trailing '_row_col' (confirmed via probe)."""
    parts = patch_id.split("_")
    return "_".join(parts[:-2])


def s1_product_name(s1_name: str) -> str:
    """S1 archive top-level folder = s1_name minus trailing '_tile_row_col' (confirmed via
    probe_archive.py: e.g. 'S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39' lives under archive
    folder 'S1A_IW_GRDH_1SDV_20170613T165043' -- one S1 scene covers multiple tiles, so the tile
    is NOT part of the top-level product folder, unlike S2)."""
    parts = s1_name.split("_")
    return "_".join(parts[:-3])


def main():
    meta = pd.read_parquet(DATA_DIR / "metadata.parquet")
    meta_sc = pd.read_parquet(DATA_DIR / "metadata_for_patches_with_snow_cloud_or_shadow.parquet")
    meta["contains_seasonal_snow"] = meta.get("contains_seasonal_snow", False)
    meta["contains_cloud_or_shadow"] = meta.get("contains_cloud_or_shadow", False)
    df = pd.concat([meta, meta_sc], ignore_index=True)
    assert df["patch_id"].is_unique
    print(f"Combined metadata rows: {len(df)}")

    df["s2_product"] = df["patch_id"].map(s2_product_name)
    df["s1_product"] = df["s1_name"].map(s1_product_name)
    df["s1_mission"] = df["s1_name"].str.slice(0, 3)  # 'S1A' or 'S1B'

    s2_products_sorted = sorted(df["s2_product"].unique())
    print(f"Total distinct S2 products: {len(s2_products_sorted)}")
    chosen_products = s2_products_sorted[:N_S2_PRODUCTS]
    print("Chosen S2 products (alphabetically earliest):")
    for p in chosen_products:
        print(" -", p)

    in_products = df[df["s2_product"].isin(chosen_products)].copy()
    print(f"Patches in chosen S2 products: {len(in_products)}")
    n_classes_all = in_products["labels"].explode().nunique()
    print(f"Classes covered by chosen products (pre S1A-filter): {n_classes_all}")

    candidate = in_products[in_products["s1_mission"] == "S1A"].copy()
    print(f"Patches after S1A-only filter: {len(candidate)}")
    n_classes_candidate = candidate["labels"].explode().nunique()
    print(f"Classes covered after S1A-filter: {n_classes_candidate}")

    s1_products_sorted = sorted(df["s1_product"].unique())
    needed_s1_products = candidate["s1_product"].unique()
    max_rank = max(s1_products_sorted.index(p) for p in needed_s1_products)
    print(f"S1 products needed: {len(needed_s1_products)}; max rank among {len(s1_products_sorted)}: {max_rank}")

    _jsonify_labels(candidate).to_csv(DATA_DIR / "candidate_patches.csv", index=False)

    # ---- stratified subsample for actual training/eval ----
    rng = np.random.default_rng(RNG_SEED)
    split_counts = candidate["split"].value_counts()
    print("Candidate split counts:\n", split_counts)

    # proportional allocation of TARGET_SUBSET_SIZE across splits, matching official proportions
    target_per_split = {
        s: int(round(TARGET_SUBSET_SIZE * c / len(candidate)))
        for s, c in split_counts.items()
    }
    print("Target per split:", target_per_split)

    # rarest-label-first stratification: assign each patch a priority key = its rarest label,
    # then sample so every class keeps some representation before filling the rest randomly.
    label_freq = candidate["labels"].explode().value_counts()

    def rarest_label(labels):
        return min(labels, key=lambda l: label_freq[l])

    candidate["rarest_label"] = candidate["labels"].map(rarest_label)

    selected_parts = []
    for split_name, n_target in target_per_split.items():
        split_df = candidate[candidate["split"] == split_name]
        if n_target >= len(split_df):
            selected_parts.append(split_df)
            continue
        # guarantee >=1 per class present in this split (up to a small cap), then fill randomly
        picks = []
        for cls, grp in split_df.groupby("rarest_label"):
            take = grp.sample(n=min(len(grp), max(1, n_target // max(1, split_df["rarest_label"].nunique()))),
                               random_state=RNG_SEED)
            picks.append(take)
        picked = pd.concat(picks).drop_duplicates(subset="patch_id")
        if len(picked) > n_target:
            picked = picked.sample(n=n_target, random_state=RNG_SEED)
        elif len(picked) < n_target:
            remaining = split_df[~split_df["patch_id"].isin(picked["patch_id"])]
            extra = remaining.sample(n=min(len(remaining), n_target - len(picked)), random_state=RNG_SEED)
            picked = pd.concat([picked, extra])
        selected_parts.append(picked)

    subset = pd.concat(selected_parts).drop_duplicates(subset="patch_id").reset_index(drop=True)
    print(f"Final subset size: {len(subset)}")
    print("Subset split counts:\n", subset["split"].value_counts())
    n_classes_subset = subset["labels"].explode().nunique()
    print(f"Classes covered in final subset: {n_classes_subset}")

    _jsonify_labels(subset.drop(columns=["rarest_label"])).to_csv(DATA_DIR / "subset_patches.csv", index=False)

    report = {
        "n_s2_products_used": N_S2_PRODUCTS,
        "chosen_s2_products": chosen_products,
        "n_candidate_patches": len(candidate),
        "n_classes_candidate": int(n_classes_candidate),
        "n_s1_products_needed": len(needed_s1_products),
        "max_s1_rank_needed": int(max_rank),
        "n_s1_products_total": len(s1_products_sorted),
        "subset_size": len(subset),
        "subset_split_counts": subset["split"].value_counts().to_dict(),
        "n_classes_subset": int(n_classes_subset),
        "all_19_classes_present": bool(n_classes_subset == 19),
    }
    with open(DATA_DIR / "selection_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
