"""PyTorch Dataset for the three experimental conditions (A/B/C), built on top of the
preprocessed per-patch .npy tensors and the pre-computed normalization stats."""
import json
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset

from labels import encode_labels, load_classes
from masking import apply_mask, make_mask

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "outputs" / "cache" / "preprocessed"


def load_norm_stats(stats_path: pathlib.Path = None):
    stats_path = stats_path or (DATA_DIR / "norm_stats.json")
    with open(stats_path) as f:
        return json.load(f)


def normalize(arr: np.ndarray, stats: list) -> np.ndarray:
    """arr: (C,H,W). stats: list of {p2,p98,mean,std} per channel -> z-score after percentile clip."""
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[0]):
        s = stats[c]
        clipped = np.clip(arr[c], s["p2"], s["p98"])
        out[c] = (clipped - s["mean"]) / s["std"]
    return out


class BENSubset(Dataset):
    """
    mode: 'optical_only'   -> returns (optical[12,H,W], label)          [condition A]
          'optical_masked' -> returns (masked_optical[12,H,W], label)   [condition B eval]
          'fusion'         -> returns (concat[14,H,W], label)           [condition C]

    coverage: float (fixed mask level) or (lo, hi) tuple to sample uniformly per-item (used for
    training condition C so it sees a range of degradation severities).
    seed_eval: if not None, masks are deterministic per-item (same mask every epoch/run) -- used
    for evaluation so B and C are compared on the *same* masked realization per test patch.
    """

    def __init__(self, df, mode: str, coverage=0.0, seed_eval: int | None = None,
                 classes: list = None, stats: dict = None):
        """classes/stats: optional overrides (round 3 uses an 8-class vocabulary and its own
        norm stats, computed from its own larger train split -- see run_experiment_v3.py).
        Defaults to the round-1/2 19-class vocabulary and stats for backward compatibility."""
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.coverage = coverage
        self.seed_eval = seed_eval
        self.classes = classes if classes is not None else load_classes()
        self.stats = stats if stats is not None else load_norm_stats()

    def __len__(self):
        return len(self.df)

    def _coverage_for(self, idx):
        if isinstance(self.coverage, tuple):
            rng = np.random.default_rng()  # training-time randomness, fresh each call
            return rng.uniform(*self.coverage)
        return float(self.coverage)

    def _rng_for(self, idx):
        if self.seed_eval is not None:
            return np.random.default_rng(self.seed_eval * 1_000_003 + idx)
        return np.random.default_rng()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch_id, s1_name = row["patch_id"], row["s1_name"]

        optical_raw = np.load(CACHE_DIR / "S2" / f"{patch_id}.npy")
        optical = normalize(optical_raw, self.stats["s2"])

        cov = self._coverage_for(idx)
        if self.mode in ("optical_masked", "fusion") and cov > 0:
            rng = self._rng_for(idx)
            mask = make_mask(cov, rng)
            optical = apply_mask(optical, mask)
        else:
            mask = np.zeros(optical.shape[1:], dtype=bool)

        if self.mode == "fusion":
            sar_raw = np.load(CACHE_DIR / "S1" / f"{s1_name}.npy")
            sar = normalize(sar_raw, self.stats["s1"])
            x = np.concatenate([optical, sar], axis=0)  # (14, H, W)
        else:
            x = optical  # (12, H, W)

        labels = row["labels"]
        if isinstance(labels, str):
            labels = json.loads(labels)
        y = np.array(encode_labels(labels, self.classes), dtype=np.float32)

        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y), patch_id
