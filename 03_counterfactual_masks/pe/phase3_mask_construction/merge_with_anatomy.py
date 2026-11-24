"""Merge synthetic PE mask into the combined segmentation.

Loads the merged TotalSegmentator mask (body + organs + vessels + lobes)
and overlays vessels (label 119) and PE (label 121).

Label mapping (from TotalSegmentator legacy lung_vessels task):
  119 = lung_vessels (pulmonary vasculature)
  120 = lung_trachea_bronchia (airways)
  121 = pulmonary_embolism

The legacy lung_vessels task produced swapped outputs: its "lung_vessels"
channel is spatially airways, and its "lung_trachea_bronchia" channel is
spatially vessels. The diffusion model was trained with this convention,
so label 119 = vessels is the correct label for PE-adjacent vasculature.

When a vessel_mask is provided (e.g. from v2 TotalSegmentator or Phase 1
vessels_cleaned), it is written as label 119 before PE overlay. This
ensures the diffusion model sees PE (121) inside vessels (119), matching
the spatial relationship it learned during training.

Input merged masks are at:
  $CF_RAW/pe_training_data_all/
      total_segmentator_all_lung_pe/{patient_id}_segmentation_merged.nii.gz
"""

import numpy as np
from pathlib import Path
import sys

from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.nifti_io import load_nifti, save_nifti, DATA_ROOT

MERGED_SEG_DIR = DATA_ROOT / "pe_training_data_all" / "total_segmentator_all_lung_pe"

# Base segmentation (without lung_vessels overlay) for recovering
# original anatomy labels when clearing legacy label 119.
BASE_SEG_DIR = DATA_ROOT / "pe_training_data_all" / "total_segmentator_all"

VESSEL_LABEL = 119   # lung_vessels (pulmonary vasculature)
AIRWAY_LABEL = 120   # lung_trachea_bronchia (airways)
PE_LABEL = 121       # pulmonary_embolism


def get_merged_segmentation_path(patient_id: str) -> Path:
    """Return path to the merged TotalSegmentator segmentation."""
    return MERGED_SEG_DIR / f"{patient_id}_segmentation_merged.nii.gz"


def merge_pe_with_segmentation(pe_mask: np.ndarray,
                                patient_id: str,
                                vessel_mask: np.ndarray = None,
                                pe_label: int = PE_LABEL) -> tuple:
    """Overlay vessels and synthetic PE onto the merged segmentation mask.

    When vessel_mask is provided, it replaces the existing label 119
    region with the given vessel mask. This is critical when writing v2 vessels
    where Phase 1 uses v2 vessel segmentation (arteries + veins) which
    has different spatial extent than the legacy label 119.

    Args:
        pe_mask: Binary PE mask (uint8, same shape as segmentation).
        patient_id: Patient ID for loading the base segmentation.
        vessel_mask: Optional binary vessel mask to write as label 119.
        pe_label: Integer label to assign to PE voxels (default 121).

    Returns:
        (combined_mask, affine, header) - the merged segmentation with
        vessels (119) and PE (121) added.
    """
    seg_path = get_merged_segmentation_path(patient_id)
    seg_data, affine, header = load_nifti(str(seg_path))

    # Make a copy to avoid modifying the original
    combined = seg_data.copy()

    # If vessel_mask provided, replace label 119 with the given vessel mask.
    # First clear legacy label 119 (inaccurate predictions from the legacy
    # TotalSegmentator lung_vessels task), restoring the original anatomy
    # labels from the base segmentation (total_segmentator_all). Then write
    # the v2 vessel mask as label 119.
    if vessel_mask is not None:
        legacy_119 = combined == VESSEL_LABEL
        if np.any(legacy_119):
            base_path = BASE_SEG_DIR / f"{patient_id}_segmentation_merged.nii.gz"
            base_data, _, _ = load_nifti(str(base_path))
            combined[legacy_119] = base_data[legacy_119]

            # Some legacy-119 voxels map to background (label 0) in the
            # base seg - spurious predictions in air/parenchyma. Fill
            # these holes with the nearest non-zero neighbor label.
            holes = legacy_119 & (combined == 0)
            if np.any(holes):
                non_zero = combined > 0
                # Compute index of nearest non-zero voxel for each hole
                _, nearest_idx = ndimage.distance_transform_edt(
                    ~non_zero, return_distances=True, return_indices=True
                )
                combined[holes] = combined[
                    nearest_idx[0][holes],
                    nearest_idx[1][holes],
                    nearest_idx[2][holes],
                ]

        vessel_binary = vessel_mask > 0
        combined[vessel_binary] = VESSEL_LABEL

    # Overlay PE label wherever pe_mask is non-zero (on top of vessels)
    combined[pe_mask > 0] = pe_label

    return combined, affine, header


def save_combined_segmentation(pe_mask: np.ndarray,
                                patient_id: str,
                                config_id: str,
                                output_dir: str,
                                vessel_mask: np.ndarray = None,
                                pe_label: int = PE_LABEL) -> str:
    """Create and save combined segmentation with synthetic PE.

    Args:
        pe_mask: Binary PE mask.
        patient_id: Patient ID.
        config_id: Config ID for the output filename.
        output_dir: Where to save.
        vessel_mask: Optional binary vessel mask to write as label 119.
        pe_label: PE label index (default 121).

    Returns:
        Path to saved file.
    """
    combined, affine, header = merge_pe_with_segmentation(
        pe_mask, patient_id, vessel_mask=vessel_mask, pe_label=pe_label
    )

    output_path = Path(output_dir) / f"{config_id}_combined_seg.nii.gz"
    save_nifti(combined, affine, header, str(output_path))

    return str(output_path)
