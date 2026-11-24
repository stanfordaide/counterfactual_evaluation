"""Occlusion pattern application for PE mask construction.

Implements complete, eccentric partial, and concentric partial fill patterns
that determine which lumen voxels become PE at each cross-section.
"""

import numpy as np
from typing import Dict, List, Set, Tuple


def compute_fill_direction(segment_coords: List[Tuple[int, int, int]],
                           spacing: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """Compute a consistent perpendicular fill direction for eccentric occlusion.

    The fill direction is a unit vector perpendicular to the segment's
    tangent, chosen randomly (seeded). PE voxels on the positive side
    of this direction are filled first (simulating wall-hugging clot).

    Args:
        segment_coords: Ordered skeleton coordinates.
        spacing: Voxel spacing in mm.
        rng: Seeded random generator.

    Returns:
        Unit vector (3,) perpendicular to segment tangent.
    """
    if len(segment_coords) < 2:
        # Degenerate: pick a random direction
        d = rng.standard_normal(3)
        return d / np.linalg.norm(d)

    # Compute tangent from first to last skeleton point
    p0 = np.array(segment_coords[0], dtype=np.float64) * spacing
    p1 = np.array(segment_coords[-1], dtype=np.float64) * spacing
    tangent = p1 - p0
    t_norm = np.linalg.norm(tangent)

    if t_norm < 1e-8:
        d = rng.standard_normal(3)
        return d / np.linalg.norm(d)

    tangent = tangent / t_norm

    # Generate a random vector and project out the tangent component
    # (Gram-Schmidt to get perpendicular direction)
    random_vec = rng.standard_normal(3)
    perp = random_vec - np.dot(random_vec, tangent) * tangent
    perp_norm = np.linalg.norm(perp)

    if perp_norm < 1e-8:
        # Random vector was nearly parallel to tangent - try another
        random_vec = np.array([0.0, 1.0, 0.0])
        perp = random_vec - np.dot(random_vec, tangent) * tangent
        perp_norm = np.linalg.norm(perp)

    return perp / perp_norm


def apply_complete_occlusion(lumen_voxels: Set[Tuple[int, int, int]]
                              ) -> Set[Tuple[int, int, int]]:
    """Complete occlusion: fill the entire lumen."""
    return set(lumen_voxels)


def apply_eccentric_occlusion(
        cross_sections: Dict[int, List[Tuple[int, int, int]]],
        segment_coords: List[Tuple[int, int, int]],
        fill_direction: np.ndarray,
        fraction: float,
        spacing: np.ndarray,
        taper_weights: np.ndarray = None) -> Set[Tuple[int, int, int]]:
    """Eccentric partial fill: crescent-shaped clot hugging one vessel wall.

    For each cross-section, projects lumen voxels onto the fill direction
    and keeps the top `fraction` (closest to the wall on the fill side).

    Args:
        cross_sections: Dict mapping skeleton index → lumen voxel list.
        segment_coords: Ordered skeleton coordinates.
        fill_direction: Unit vector for eccentric fill direction.
        fraction: Target fraction of cross-section to fill (0.0-1.0).
        spacing: Voxel spacing in mm.
        taper_weights: Optional per-skeleton-point taper weights [0..1].
                       Multiplied with fraction at each cross-section.

    Returns:
        Set of filled voxel coordinates.
    """
    filled = set()

    for skel_idx, voxels in cross_sections.items():
        if not voxels:
            continue

        # Local fraction (apply taper if provided)
        local_frac = fraction
        if taper_weights is not None and skel_idx < len(taper_weights):
            local_frac = fraction * taper_weights[skel_idx]

        if local_frac <= 0:
            continue

        # Skeleton point position in mm
        sz, sy, sx = segment_coords[skel_idx]
        skel_mm = np.array([sz, sy, sx], dtype=np.float64) * spacing

        # Compute projection of each voxel onto fill direction
        projections = []
        for vz, vy, vx in voxels:
            voxel_mm = np.array([vz, vy, vx], dtype=np.float64) * spacing
            offset = voxel_mm - skel_mm
            proj = np.dot(offset, fill_direction)
            projections.append(proj)

        # Sort by projection (descending = closest to wall on fill side)
        sorted_indices = np.argsort(projections)[::-1]

        # Keep top fraction
        n_fill = max(1, int(len(voxels) * local_frac))
        n_fill = min(n_fill, len(voxels))

        for i in range(n_fill):
            filled.add(voxels[sorted_indices[i]])

    return filled


def apply_concentric_occlusion(
        cross_sections: Dict[int, List[Tuple[int, int, int]]],
        segment_coords: List[Tuple[int, int, int]],
        fraction: float,
        spacing: np.ndarray,
        taper_weights: np.ndarray = None) -> Set[Tuple[int, int, int]]:
    """Concentric (acute PE) fill: central clot floating inside the vessel.

    For each cross-section, computes distance of each lumen voxel from the
    skeleton center point and selects the CLOSEST voxels (near center).
    Produces a bullet-shaped / concentric circle pattern.

    Args:
        cross_sections: Dict mapping skeleton index -> lumen voxel list.
        segment_coords: Ordered skeleton coordinates.
        fraction: Target fraction of cross-section to fill (0.0-1.0).
        spacing: Voxel spacing in mm.
        taper_weights: Optional per-skeleton-point taper weights [0..1].

    Returns:
        Set of filled voxel coordinates.
    """
    filled = set()

    for skel_idx, voxels in cross_sections.items():
        if not voxels:
            continue

        # Local fraction (apply taper if provided)
        local_frac = fraction
        if taper_weights is not None and skel_idx < len(taper_weights):
            local_frac = fraction * taper_weights[skel_idx]

        if local_frac <= 0:
            continue

        # Skeleton point position in mm
        sz, sy, sx = segment_coords[skel_idx]
        skel_mm = np.array([sz, sy, sx], dtype=np.float64) * spacing

        # Compute distance of each voxel from the skeleton center
        distances = []
        for vz, vy, vx in voxels:
            voxel_mm = np.array([vz, vy, vx], dtype=np.float64) * spacing
            dist = np.linalg.norm(voxel_mm - skel_mm)
            distances.append(dist)

        # Sort by distance ASCENDING (closest to center first)
        sorted_indices = np.argsort(distances)

        # Keep closest fraction (center-filling)
        n_fill = max(1, int(len(voxels) * local_frac))
        n_fill = min(n_fill, len(voxels))

        for i in range(n_fill):
            filled.add(voxels[sorted_indices[i]])

    return filled


def sample_occlusion_fraction(occlusion_degree: str,
                               rng: np.random.Generator) -> float:
    """Sample a specific occlusion fraction from the degree's range.

    Args:
        occlusion_degree: "partial", "near_complete", or "complete".
        rng: Random generator.

    Returns:
        Float fraction in [0, 1].
    """
    ranges = {
        'partial': (0.30, 0.50),
        'near_complete': (0.70, 0.90),
        'complete': (1.0, 1.0),
    }
    lo, hi = ranges.get(occlusion_degree, (1.0, 1.0))
    if lo == hi:
        return lo
    return float(rng.uniform(lo, hi))


def apply_occlusion(cross_sections: Dict[int, List[Tuple[int, int, int]]],
                    segment_coords: List[Tuple[int, int, int]],
                    lumen_voxels: Set[Tuple[int, int, int]],
                    occlusion_degree: str,
                    spacing: np.ndarray,
                    rng: np.random.Generator,
                    taper_weights: np.ndarray = None,
                    acute: bool = False,
                    ) -> Set[Tuple[int, int, int]]:
    """Apply occlusion pattern to a segment's lumen voxels.

    Dispatches to the correct fill pattern based on occlusion degree.

    Args:
        cross_sections: Dict mapping skeleton index → lumen voxel list.
        segment_coords: Ordered skeleton coordinates.
        lumen_voxels: Full set of lumen voxels for the segment.
        occlusion_degree: "partial", "near_complete", or "complete".
        spacing: Voxel spacing in mm.
        rng: Random generator.
        taper_weights: Optional per-skeleton-point taper weights.
        acute: If True, use concentric (center-filling) pattern instead of
               eccentric (wall-hugging) pattern.

    Returns:
        Set of filled voxel coordinates.
    """
    fraction = sample_occlusion_fraction(occlusion_degree, rng)

    if fraction >= 1.0:
        # Complete occlusion with optional tapering
        if taper_weights is not None:
            if acute:
                return apply_concentric_occlusion(
                    cross_sections, segment_coords,
                    1.0, spacing, taper_weights)
            else:
                fill_dir = compute_fill_direction(segment_coords, spacing, rng)
                return apply_eccentric_occlusion(
                    cross_sections, segment_coords, fill_dir,
                    1.0, spacing, taper_weights)
        return apply_complete_occlusion(lumen_voxels)

    # Partial or near-complete
    if acute:
        return apply_concentric_occlusion(
            cross_sections, segment_coords,
            fraction, spacing, taper_weights)
    else:
        fill_dir = compute_fill_direction(segment_coords, spacing, rng)
        return apply_eccentric_occlusion(
            cross_sections, segment_coords, fill_dir,
            fraction, spacing, taper_weights)
