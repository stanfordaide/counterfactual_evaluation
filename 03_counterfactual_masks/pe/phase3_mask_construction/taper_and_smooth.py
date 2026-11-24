"""End-tapering and Gaussian smoothing for constructed PE masks.

Tapering avoids abrupt PE start/stop at segment endpoints.
Smoothing rounds jagged voxel edges for realistic appearance.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def compute_taper_weights(n_points: int, taper_length: int = 3) -> np.ndarray:
    """Compute per-skeleton-point taper weights.

    Creates a trapezoidal profile that ramps up from 0 at the ends
    to 1.0 in the middle, over `taper_length` points on each side.

    Args:
        n_points: Number of skeleton points in the segment.
        taper_length: Number of points to taper on each end.

    Returns:
        Array of shape (n_points,) with values in [0, 1].
    """
    if n_points <= 0:
        return np.array([])

    if n_points <= 2:
        # Very short segment - use uniform moderate weight
        return np.full(n_points, 0.6)

    # Limit taper to at most 1/3 of segment length on each side
    taper_length = min(taper_length, max(1, n_points // 3))

    weights = np.ones(n_points)

    # Ramp up at start
    for i in range(taper_length):
        weights[i] = (i + 1) / (taper_length + 1)

    # Ramp down at end
    for i in range(taper_length):
        weights[n_points - 1 - i] = (i + 1) / (taper_length + 1)

    return weights


def smooth_and_clean(pe_mask: np.ndarray,
                     vessel_mask: np.ndarray,
                     sigma: float = 0.8,
                     threshold: float = 0.3) -> np.ndarray:
    """Gaussian smooth the PE mask and enforce vessel containment.

    Args:
        pe_mask: Binary PE mask (uint8).
        vessel_mask: Binary vessel mask (uint8).
        sigma: Gaussian sigma in voxels. ~0.8 rounds edges
               without losing small features.
        threshold: Re-binarization threshold after smoothing.
                   0.3 preserves more volume than 0.5.

    Returns:
        Cleaned binary PE mask (uint8).
    """
    if np.sum(pe_mask) == 0:
        return pe_mask.copy()

    # Gaussian blur
    pe_float = pe_mask.astype(np.float32)
    pe_smooth = gaussian_filter(pe_float, sigma=sigma)

    # Re-binarize
    pe_binary = (pe_smooth > threshold).astype(np.uint8)

    # Enforce vessel containment
    pe_binary = pe_binary & (vessel_mask > 0).astype(np.uint8)

    return pe_binary
