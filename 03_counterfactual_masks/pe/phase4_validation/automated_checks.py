"""Phase 4: automated validation checks for synthetic PE masks.

Four checks per config:
  1. Vessel containment (at least 99% of PE voxels inside the vessel mask)
  2. Volume plausibility (compared to empirical bounds)
  3. Shape plausibility (elongation/sphericity from inertia tensor)
  4. Anatomical integrity (label 121 present, no airway overlap, anatomy preserved)
"""

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from utils.nifti_io import load_nifti


def compute_inertia_features(coords: np.ndarray) -> dict:
    """Elongation + sphericity from voxel coordinates via the inertia tensor."""
    if len(coords) < 4:
        return {"elongation": np.nan, "sphericity": np.nan}
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]  # descending
    if eigvals[0] == 0:
        return {"elongation": np.nan, "sphericity": np.nan}
    elongation = 1.0 - (eigvals[1] / eigvals[0])
    sphericity = eigvals[2] / eigvals[0] if eigvals[0] > 0 else np.nan
    return {
        "eigval_1": float(eigvals[0]),
        "eigval_2": float(eigvals[1]),
        "eigval_3": float(eigvals[2]),
        "elongation": float(elongation),
        "sphericity": float(sphericity),
    }

# Labels that PE must never overlap with (non-vascular structures)
AIRWAY_LABELS = {16}  # trachea in TotalSegmentator merged mask
LUNG_LOBE_LABELS = {10, 11, 12, 13, 14}
PE_LABEL = 121

# Volume bounds per location (generous, empirically derived)
# Format: (min_ml, max_ml)
VOLUME_BOUNDS = {
    'subsegmental': (0.0001, 60.0),
    'segmental': (0.0001, 10.0),
    'lobar': (0.001, 30.0),
    'central': (0.01, 50.0),
}


def check_containment(pe_mask: np.ndarray,
                       vessel_mask: np.ndarray) -> dict:
    """Check that PE voxels are contained within the vessel mask.

    Args:
        pe_mask: Binary PE mask.
        vessel_mask: Binary vessel mask.

    Returns:
        Dict with containment_fraction and pass_containment flag.
    """
    pe_voxels = int(np.sum(pe_mask > 0))
    if pe_voxels == 0:
        return {
            'pe_voxels': 0,
            'containment_fraction': 1.0,
            'pass_containment': True,
            'containment_note': 'empty_pe_mask',
        }

    overlap = int(np.sum((pe_mask > 0) & (vessel_mask > 0)))
    containment = overlap / pe_voxels

    return {
        'pe_voxels': pe_voxels,
        'vessel_overlap_voxels': overlap,
        'containment_fraction': round(containment, 6),
        'pass_containment': containment >= 0.99,
    }


def check_volume(pe_mask: np.ndarray,
                  spacing: np.ndarray,
                  config_metadata: dict) -> dict:
    """Check PE volume is plausible for the specified location.

    Args:
        pe_mask: Binary PE mask.
        spacing: Voxel spacing (z, y, x) in mm.
        config_metadata: Dict with 'location' key.

    Returns:
        Dict with volume metrics and pass_volume flag.
    """
    pe_voxels = int(np.sum(pe_mask > 0))
    voxel_vol_ml = float(np.prod(spacing)) / 1000.0
    pe_volume_ml = pe_voxels * voxel_vol_ml

    location = config_metadata.get('location', 'subsegmental')
    min_ml, max_ml = VOLUME_BOUNDS.get(location, (0.0001, 60.0))

    pass_volume = (pe_voxels > 0) and (min_ml <= pe_volume_ml <= max_ml)

    return {
        'pe_volume_ml': round(pe_volume_ml, 6),
        'volume_min_bound_ml': min_ml,
        'volume_max_bound_ml': max_ml,
        'pass_volume': bool(pass_volume),
    }


def check_shape(pe_mask: np.ndarray,
                 spacing: np.ndarray) -> dict:
    """Check PE shape is plausible (tube-like, not spherical blob).

    Args:
        pe_mask: Binary PE mask.
        spacing: Voxel spacing in mm.

    Returns:
        Dict with shape metrics and pass_shape flag.
    """
    pe_coords_idx = np.argwhere(pe_mask > 0)
    if len(pe_coords_idx) < 4:
        return {
            'n_components': 0,
            'elongation': np.nan,
            'sphericity': np.nan,
            'pass_shape': len(pe_coords_idx) == 0,  # empty mask passes trivially
            'shape_note': 'too_few_voxels' if len(pe_coords_idx) > 0 else 'empty',
        }

    # Connected components
    labeled, n_components = ndimage.label(pe_mask > 0)

    # Global shape from inertia tensor
    pe_coords_mm = pe_coords_idx.astype(np.float64) * spacing
    shape_stats = compute_inertia_features(pe_coords_mm)

    elongation = shape_stats.get('elongation', np.nan)
    sphericity = shape_stats.get('sphericity', np.nan)

    # Pass criteria: not a perfect sphere
    # Real PE case-level: elongation ~0.72, sphericity ~0.13
    # But isolated single-segment PE in short vessels can be compact (elongation ~0.1)
    # which is physically realistic. Only fail truly degenerate shapes.
    # For multi-component configs (n_components >= 3), spatial spread drives high elongation.
    pass_shape = True
    if not np.isnan(elongation) and not np.isnan(sphericity):
        if n_components >= 3:
            # Multi-component: spatial spread should give reasonable elongation
            pass_shape = elongation > 0.2 and sphericity < 0.5
        else:
            # Single/dual component: allow compact shapes (short vessel segments)
            pass_shape = sphericity < 0.7

    return {
        'n_components': n_components,
        'elongation': round(elongation, 4) if not np.isnan(elongation) else np.nan,
        'sphericity': round(sphericity, 4) if not np.isnan(sphericity) else np.nan,
        'pass_shape': bool(pass_shape),
    }


def check_anatomy(combined_seg: np.ndarray,
                   pe_mask: np.ndarray,
                   original_seg: np.ndarray,
                   vessel_mask: np.ndarray = None) -> dict:
    """Check anatomical integrity of the combined segmentation.

    Verifies:
    - PE label 121 is present where pe_mask > 0
    - PE doesn't overlap airway labels (trachea=16)
    - Original anatomy is preserved where neither PE nor vessel mask modified it

    The vessel_mask parameter accounts for voxels where the merge step
    overwrote the original label with 119 (lung_vessels). These are
    expected modifications, not corruption.

    Args:
        combined_seg: Combined segmentation (labels 0-120 + 121 for PE).
        pe_mask: Binary PE mask.
        original_seg: Original merged segmentation (no PE).
        vessel_mask: Optional binary vessel mask written as label 119.

    Returns:
        Dict with integrity metrics and pass_anatomy flag.
    """
    pe_binary = pe_mask > 0
    pe_voxels = int(np.sum(pe_binary))

    # Check label 121 presence
    label_121_voxels = int(np.sum(combined_seg == PE_LABEL))
    pe_label_match = (label_121_voxels == pe_voxels)

    # Check airway overlap
    airway_overlap = 0
    if pe_voxels > 0:
        pe_coords = np.argwhere(pe_binary)
        orig_labels_at_pe = original_seg[pe_coords[:, 0], pe_coords[:, 1], pe_coords[:, 2]]
        airway_overlap = int(np.sum(np.isin(orig_labels_at_pe, list(AIRWAY_LABELS))))

    # Check anatomy preservation (where neither PE nor vessel mask modified it)
    modified_mask = pe_binary
    if vessel_mask is not None:
        modified_mask = modified_mask | (vessel_mask > 0)
    unmodified = ~modified_mask
    anatomy_preserved = bool(np.all(combined_seg[unmodified] == original_seg[unmodified]))

    pass_anatomy = pe_label_match and (airway_overlap == 0) and anatomy_preserved

    return {
        'pe_label_121_voxels': label_121_voxels,
        'pe_label_match': pe_label_match,
        'airway_overlap_voxels': airway_overlap,
        'anatomy_preserved': anatomy_preserved,
        'pass_anatomy': bool(pass_anatomy),
    }


def validate_config(pe_mask_path: str,
                     combined_seg_path: str,
                     vessel_mask: np.ndarray,
                     original_seg: np.ndarray,
                     spacing: np.ndarray,
                     config_metadata: dict) -> dict:
    """Run all 4 validation checks for one counterfactual config.

    Args:
        pe_mask_path: Path to PE-only mask NIfTI.
        combined_seg_path: Path to combined segmentation NIfTI.
        vessel_mask: Pre-loaded binary vessel mask (also used as label 119
            in the combined seg).
        original_seg: Pre-loaded original merged segmentation.
        spacing: Voxel spacing in mm.
        config_metadata: Dict with config info (config_id, location, etc.).

    Returns:
        Combined dict with all check results and overall pass/fail.
    """
    config_id = config_metadata.get('config_id', 'unknown')

    # Load per-config masks
    pe_data, _, _ = load_nifti(pe_mask_path)
    pe_mask = (pe_data > 0).astype(np.uint8)

    combined_data, _, _ = load_nifti(combined_seg_path)

    # Run checks
    result = {'config_id': config_id}
    result.update(config_metadata)

    containment = check_containment(pe_mask, vessel_mask)
    result.update({f'c1_{k}': v for k, v in containment.items()})

    volume = check_volume(pe_mask, spacing, config_metadata)
    result.update({f'c2_{k}': v for k, v in volume.items()})

    shape = check_shape(pe_mask, spacing)
    result.update({f'c3_{k}': v for k, v in shape.items()})

    anatomy = check_anatomy(combined_data, pe_mask, original_seg,
                            vessel_mask=vessel_mask)
    result.update({f'c4_{k}': v for k, v in anatomy.items()})

    # Overall pass
    result['all_checks_pass'] = all([
        containment['pass_containment'],
        volume['pass_volume'],
        shape['pass_shape'],
        anatomy['pass_anatomy'],
    ])

    return result
