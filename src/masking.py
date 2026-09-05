"""
Simulated cloud/occlusion masking for the "degraded optical" conditions (B and C).

[SUGGESTED design, per REQUIREMENTS.md #4]: the PDF leaves the masking mechanism unspecified
beyond "artificially mask portions of Sentinel-2 imagery" and "different levels of simulated
cloud coverage". We use randomly placed, randomly sized rectangular occlusions (1-3 per patch)
whose combined area is tuned to hit a target coverage fraction, applied identically across all
optical bands (spatial mask, not per-band) and replacing masked pixels with the per-band mean
(a neutral, "no information" value rather than 0, which would otherwise look like a real -- and
informative -- extreme reflectance value). This is a simplification of true cloud shape/texture,
chosen for speed and reproducibility, not because it is claimed to be realistic.
"""
import numpy as np

IMG_SIZE = 120


def make_mask(coverage: float, rng: np.random.Generator, size: int = IMG_SIZE,
              max_rects: int = 12) -> np.ndarray:
    """Boolean (size, size) array, True = masked/occluded pixel. `coverage` in [0, 1].
    Rectangles are placed one at a time against the *actual* accumulated mask (rather than a
    pre-computed area budget) so overlap between rectangles doesn't cause under-coverage at
    higher target levels."""
    mask = np.zeros((size, size), dtype=bool)
    if coverage <= 0:
        return mask
    target_area = coverage * size * size
    for _ in range(max_rects):
        current_area = mask.sum()
        if current_area >= target_area:
            break
        # size this rectangle to the actual remaining shortfall (recomputed from the real mask
        # each iteration, so overlap with previous rectangles is naturally corrected for)
        frac = (target_area - current_area) / (size * size)
        aspect = rng.uniform(0.5, 2.0)
        h = int(round((frac * size * size / aspect) ** 0.5))
        w = int(round(frac * size * size / max(h, 1)))
        h = min(max(h, 1), size)
        w = min(max(w, 1), size)
        top = int(rng.integers(0, max(1, size - h + 1)))
        left = int(rng.integers(0, max(1, size - w + 1)))
        mask[top:top + h, left:left + w] = True
    return mask


def apply_mask(optical: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """optical: (C, H, W) float array. mask: (H, W) bool, True=masked.
    Masked pixels are replaced with that band's mean over the *unmasked* region of this patch
    (falls back to the global patch mean if the whole patch would otherwise be masked)."""
    out = optical.copy()
    if not mask.any():
        return out
    for c in range(out.shape[0]):
        band = out[c]
        unmasked = band[~mask]
        fill = float(unmasked.mean()) if unmasked.size > 0 else float(band.mean())
        band[mask] = fill
    return out


def masked_fraction(mask: np.ndarray) -> float:
    return float(mask.mean())
