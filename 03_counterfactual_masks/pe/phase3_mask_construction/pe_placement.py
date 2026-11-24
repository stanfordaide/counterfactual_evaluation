"""Lumen voxel extraction for PE mask construction.

Given a vessel segment (skeleton coordinates from the graph), extracts the
full set of lumen voxels that belong to that segment by intersecting
balls of local radius with the vessel mask.
"""

import math
import numpy as np
from scipy import ndimage
from typing import Dict, List, Set, Tuple


def compute_full_distance_transform(vessel_mask: np.ndarray,
                                     spacing: np.ndarray) -> np.ndarray:
    """Compute distance transform over the full vessel mask.

    Returns the distance (in mm) from each vessel voxel to the nearest
    boundary. This gives the local vessel radius at every point.

    Args:
        vessel_mask: Binary vessel mask (uint8 or bool)
        spacing: Voxel spacing in mm, shape (3,) for (z, y, x)

    Returns:
        Distance transform array (float32), same shape as vessel_mask.
        Non-zero only inside vessels.
    """
    dt = ndimage.distance_transform_edt(vessel_mask.astype(bool),
                                         sampling=spacing)
    return dt.astype(np.float32)


def extract_segment_lumen(segment_coords: List[Tuple[int, int, int]],
                          vessel_mask: np.ndarray,
                          dt: np.ndarray,
                          spacing: np.ndarray,
                          radius_padding: float = 0.5) -> Set[Tuple[int, int, int]]:
    """Extract all vessel lumen voxels for a skeleton segment.

    Uses a ball-union approach: for each skeleton point, collects all
    vessel mask voxels within the local radius (from the distance transform).

    Args:
        segment_coords: List of (z, y, x) integer voxel coordinates
                        along the skeleton path.
        vessel_mask: Binary vessel mask.
        dt: Full distance transform of vessel_mask (in mm).
        spacing: Voxel spacing in mm, shape (3,).
        radius_padding: Extra radius (mm) to add for slight over-capture.
                        Helps ensure full lumen coverage at boundaries.

    Returns:
        Set of (z, y, x) tuples - the lumen voxels for this segment.
    """
    if not segment_coords:
        return set()

    shape = vessel_mask.shape
    lumen_voxels = set()

    for sz, sy, sx in segment_coords:
        # Get local radius at this skeleton point
        if 0 <= sz < shape[0] and 0 <= sy < shape[1] and 0 <= sx < shape[2]:
            r_mm = float(dt[sz, sy, sx]) + radius_padding
        else:
            continue

        if r_mm <= 0:
            # Skeleton point outside vessel mask - shouldn't happen, but handle it
            lumen_voxels.add((sz, sy, sx))
            continue

        # Compute bounding box in voxel units
        r_vox = [math.ceil(r_mm / s) for s in spacing]

        z_lo = max(0, sz - r_vox[0])
        z_hi = min(shape[0], sz + r_vox[0] + 1)
        y_lo = max(0, sy - r_vox[1])
        y_hi = min(shape[1], sy + r_vox[1] + 1)
        x_lo = max(0, sx - r_vox[2])
        x_hi = min(shape[2], sx + r_vox[2] + 1)

        # Build local coordinate grid
        zz = np.arange(z_lo, z_hi)
        yy = np.arange(y_lo, y_hi)
        xx = np.arange(x_lo, x_hi)
        gz, gy, gx = np.meshgrid(zz, yy, xx, indexing='ij')

        # Euclidean distance from skeleton point (in mm)
        dist_mm = np.sqrt(
            ((gz - sz) * spacing[0]) ** 2 +
            ((gy - sy) * spacing[1]) ** 2 +
            ((gx - sx) * spacing[2]) ** 2
        )

        # Mask: within radius AND inside vessel
        mask = (dist_mm <= r_mm) & (vessel_mask[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi] > 0)

        # Collect matching voxel coordinates
        local_coords = np.argwhere(mask)
        for lc in local_coords:
            vz = int(lc[0] + z_lo)
            vy = int(lc[1] + y_lo)
            vx = int(lc[2] + x_lo)
            lumen_voxels.add((vz, vy, vx))

    return lumen_voxels


def extract_cross_sections(segment_coords: List[Tuple[int, int, int]],
                           lumen_voxels: Set[Tuple[int, int, int]],
                           spacing: np.ndarray) -> Dict[int, List[Tuple[int, int, int]]]:
    """Group lumen voxels by their nearest skeleton point.

    For each lumen voxel, finds the closest skeleton point in the segment
    and assigns it to that cross-section index. This enables per-slice
    occlusion pattern application.

    Args:
        segment_coords: Ordered list of skeleton (z, y, x) coordinates.
        lumen_voxels: Set of lumen voxel coordinates.
        spacing: Voxel spacing in mm.

    Returns:
        Dict mapping skeleton index (0..N-1) to list of lumen voxels
        belonging to that cross-section.
    """
    if not segment_coords or not lumen_voxels:
        return {}

    # Convert to arrays for vectorized distance computation
    skel_arr = np.array(segment_coords, dtype=np.float64)  # (N_skel, 3)
    skel_mm = skel_arr * spacing[np.newaxis, :]

    lumen_arr = np.array(list(lumen_voxels), dtype=np.float64)  # (N_lumen, 3)
    lumen_mm = lumen_arr * spacing[np.newaxis, :]

    # For each lumen voxel, find nearest skeleton point
    # Use broadcasting: (N_lumen, 1, 3) - (1, N_skel, 3) → (N_lumen, N_skel, 3)
    # This can be memory-heavy for large sets; chunk if needed
    n_lumen = len(lumen_arr)
    n_skel = len(skel_arr)

    cross_sections: Dict[int, List[Tuple[int, int, int]]] = {
        i: [] for i in range(n_skel)
    }

    chunk_size = 5000
    for start in range(0, n_lumen, chunk_size):
        end = min(start + chunk_size, n_lumen)
        chunk_mm = lumen_mm[start:end]  # (chunk, 3)
        chunk_vox = lumen_arr[start:end]

        # Distance from each lumen voxel to each skeleton point
        diff = chunk_mm[:, np.newaxis, :] - skel_mm[np.newaxis, :, :]  # (chunk, N_skel, 3)
        dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (chunk, N_skel)

        nearest = np.argmin(dists, axis=1)  # (chunk,)

        for i, skel_idx in enumerate(nearest):
            vz, vy, vx = int(chunk_vox[i, 0]), int(chunk_vox[i, 1]), int(chunk_vox[i, 2])
            cross_sections[int(skel_idx)].append((vz, vy, vx))

    return cross_sections
