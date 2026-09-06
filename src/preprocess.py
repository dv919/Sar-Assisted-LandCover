"""
Convert extracted raw GeoTIFF band files into per-patch .npy tensors (one file per patch per
modality), resized to a common 120x120 grid, so training doesn't re-read/resize 14 small TIFFs
per sample on every epoch. Also computes per-band normalization stats from the TRAIN split only.

Run after extract_subset.py has produced data/raw/S2/<patch_id>/*.tif and
data/raw/S1/<s1_name>/*.tif for (at least most of) subset_patches.csv.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = ROOT / "outputs" / "cache" / "preprocessed"
S2_BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
S1_BAND_ORDER = ["VV", "VH"]
IMG_SIZE = 120


def _read_and_resize(path: pathlib.Path, size: int = IMG_SIZE) -> np.ndarray:
    arr = tifffile.imread(path).astype(np.float32)
    if arr.shape != (size, size):
        arr = np.array(Image.fromarray(arr).resize((size, size), Image.BILINEAR), dtype=np.float32)
    return arr


def preprocess_s2(patch_id: str) -> np.ndarray | None:
    d = RAW_DIR / "S2" / patch_id
    bands = []
    for b in S2_BAND_ORDER:
        f = d / f"{patch_id}_{b}.tif"
        if not f.exists():
            return None
        bands.append(_read_and_resize(f))
    return np.stack(bands, axis=0)  # (12, 120, 120) float32, raw reflectance*10000 (S2 L2A convention)


def preprocess_s1(s1_name: str) -> np.ndarray | None:
    d = RAW_DIR / "S1" / s1_name
    bands = []
    for b in S1_BAND_ORDER:
        f = d / f"{s1_name}_{b}.tif"
        if not f.exists():
            return None
        bands.append(_read_and_resize(f))
    stacked = np.stack(bands, axis=0)  # (2, 120, 120), linear backscatter (gamma0-ish, per BEN convention)
    # convert to dB for numerical stability / more Gaussian-like distribution
    eps = 1e-6
    stacked_db = 10.0 * np.log10(np.clip(stacked, eps, None))
    return stacked_db.astype(np.float32)


def main():
    import sys
    subset_csv = sys.argv[1] if len(sys.argv) > 1 else "subset_patches.csv"
    out_suffix = sys.argv[2] if len(sys.argv) > 2 else ""  # e.g. "_v3" -> subset_patches_v3_complete.csv
    print(f"Using subset file: {subset_csv}; output suffix: '{out_suffix}'")
    subset = pd.read_csv(DATA_DIR / subset_csv)
    (CACHE_DIR / "S2").mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "S1").mkdir(parents=True, exist_ok=True)

    ok_s2, ok_s1 = [], []
    n_missing_s2, n_missing_s1 = 0, 0
    for i, row in subset.iterrows():
        patch_id, s1_name = row["patch_id"], row["s1_name"]

        s2_out = CACHE_DIR / "S2" / f"{patch_id}.npy"
        if not s2_out.exists():
            arr = preprocess_s2(patch_id)
            if arr is None:
                n_missing_s2 += 1
            else:
                np.save(s2_out, arr)
        if s2_out.exists():
            ok_s2.append(patch_id)

        s1_out = CACHE_DIR / "S1" / f"{s1_name}.npy"
        if not s1_out.exists():
            arr = preprocess_s1(s1_name)
            if arr is None:
                n_missing_s1 += 1
            else:
                np.save(s1_out, arr)
        if s1_out.exists():
            ok_s1.append(s1_name)

        if (i + 1) % 500 == 0:
            print(f"processed {i+1}/{len(subset)} rows "
                  f"(s2 ok={len(ok_s2)} missing={n_missing_s2}; s1 ok={len(ok_s1)} missing={n_missing_s1})")

    print(f"FINAL: s2 ok={len(ok_s2)} missing={n_missing_s2}; s1 ok={len(ok_s1)} missing={n_missing_s1}")

    complete = subset[subset["patch_id"].isin(ok_s2) & subset["s1_name"].isin(ok_s1)].copy()
    complete_path = DATA_DIR / f"subset_patches{out_suffix}_complete.csv"
    complete.to_csv(complete_path, index=False)
    print(f"Patches with BOTH modalities available: {len(complete)} / {len(subset)} -> {complete_path.name}")

    # --- normalization stats from TRAIN split only ---
    train_ids = complete[complete["split"] == "train"]
    s2_stack = np.stack([np.load(CACHE_DIR / "S2" / f"{pid}.npy") for pid in train_ids["patch_id"]])
    s1_stack = np.stack([np.load(CACHE_DIR / "S1" / f"{n}.npy") for n in train_ids["s1_name"]])

    def robust_stats(stack, axis_keep=1):
        # stack: (N, C, H, W) -> per-channel 2nd/98th percentile and mean/std within that range
        stats = []
        for c in range(stack.shape[axis_keep]):
            vals = stack[:, c].ravel()
            p2, p98 = np.percentile(vals, [2, 98])
            clipped = np.clip(vals, p2, p98)
            stats.append({"p2": float(p2), "p98": float(p98),
                          "mean": float(clipped.mean()), "std": float(clipped.std() + 1e-6)})
        return stats

    norm_stats = {
        "s2_bands": S2_BAND_ORDER,
        "s1_bands": S1_BAND_ORDER,
        "img_size": IMG_SIZE,
        "s2": robust_stats(s2_stack),
        "s1": robust_stats(s1_stack),
        "n_train_patches_used_for_stats": len(train_ids),
    }
    stats_path = DATA_DIR / f"norm_stats{out_suffix}.json"
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved normalization stats computed from {len(train_ids)} train patches -> {stats_path.name}")


if __name__ == "__main__":
    main()
