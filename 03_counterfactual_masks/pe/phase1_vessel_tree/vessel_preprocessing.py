"""Morphological cleaning of vessel masks before skeletonization.

Cleans up fragmented vessel masks by:
1. Removing small disconnected components
2. Light morphological closing to bridge small gaps
3. Optional hole filling within vessel tubes
"""

import os
import numpy as np
from scipy import ndimage
from typing import Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pe/ folder (for utils)
from utils.nifti_io import load_nifti, save_nifti


def remove_small_components(mask: np.ndarray, min_size_voxels: int = 100) -> np.ndarray:
    """Remove connected components smaller than min_size_voxels."""
    labeled, n_components = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    component_sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))

    # Keep only components above threshold
    keep_labels = np.where(component_sizes >= min_size_voxels)[0] + 1
    cleaned = np.isin(labeled, keep_labels)

    removed = n_components - len(keep_labels)
    print(f"  Removed {removed} small components (< {min_size_voxels} voxels)")
    return cleaned.astype(np.uint8)


def morphological_closing(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply morphological closing to bridge small gaps."""
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    closed = ndimage.binary_closing(mask, struct, iterations=iterations)
    return closed.astype(np.uint8)


def clean_vessel_mask(
    vessel_mask: np.ndarray,
    min_component_size: int = 100,
    closing_iterations: int = 1,
    dilation_iterations: int = 0,
    keep_largest_n: int = None,
) -> Tuple[np.ndarray, dict]:
    """
    Clean vessel mask for skeletonization.

    Args:
        vessel_mask: Binary vessel mask
        min_component_size: Remove components smaller than this (voxels)
        closing_iterations: Iterations of morphological closing
        dilation_iterations: Iterations of morphological dilation (0 = off)
        keep_largest_n: If set, keep only the N largest components

    Returns:
        cleaned_mask: Cleaned binary mask
        stats: Dictionary with cleaning statistics
    """
    stats = {
        'input_voxels': int(np.sum(vessel_mask)),
        'input_components': 0,
    }

    # Count initial components
    _, n_initial = ndimage.label(vessel_mask)
    stats['input_components'] = n_initial
    print(f"Input: {stats['input_voxels']} voxels, {n_initial} components")

    # Step 1: Morphological closing
    if closing_iterations > 0:
        print(f"Applying morphological closing (iterations={closing_iterations})...")
        vessel_mask = morphological_closing(vessel_mask, closing_iterations)
        stats['after_closing_voxels'] = int(np.sum(vessel_mask))

    # Step 1.5: Morphological dilation
    if dilation_iterations > 0:
        print(f"Applying morphological dilation (iterations={dilation_iterations})...")
        struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
        vessel_mask = ndimage.binary_dilation(vessel_mask, struct, iterations=dilation_iterations).astype(np.uint8)
        stats['after_dilation_voxels'] = int(np.sum(vessel_mask))

    # Step 2: Remove small components
    print(f"Removing small components...")
    vessel_mask = remove_small_components(vessel_mask, min_component_size)

    # Step 3: Optionally keep only largest N
    if keep_largest_n is not None:
        labeled, n_comp = ndimage.label(vessel_mask)
        if n_comp > keep_largest_n:
            sizes = ndimage.sum(vessel_mask, labeled, range(1, n_comp + 1))
            largest_labels = np.argsort(sizes)[-keep_largest_n:] + 1
            vessel_mask = np.isin(labeled, largest_labels).astype(np.uint8)
            print(f"  Kept {keep_largest_n} largest components")

    # Final stats
    _, n_final = ndimage.label(vessel_mask)
    stats['output_voxels'] = int(np.sum(vessel_mask))
    stats['output_components'] = n_final
    stats['voxels_removed'] = stats['input_voxels'] - stats['output_voxels']

    print(f"Output: {stats['output_voxels']} voxels, {n_final} components")
    print(f"Removed: {stats['voxels_removed']} voxels ({100*stats['voxels_removed']/stats['input_voxels']:.1f}%)")

    return vessel_mask, stats


DEFAULT_OUTPUT_DIR = os.environ.get("CF_OUT", "/path/to/cf_mask_pipeline_output") + "/phase1_vessel_tree/output"


def preprocess_for_patient(patient_id: str,
                           min_component_size: int = 100,
                           closing_iterations: int = 1,
                           dilation_iterations: int = 0,
                           save_output: bool = True,
                           output_dir: str = None) -> Tuple[np.ndarray, dict]:
    """
    Load and preprocess the vessel mask for a patient from the TotalSegmentator v2
    lung-vessel segmentation (labels 3 = arteries, 4 = veins).

    Returns:
        cleaned_mask: Preprocessed vessel mask
        stats: Preprocessing statistics
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    from utils.nifti_io import get_vessel_v2_path
    vessel_path = get_vessel_v2_path(patient_id)

    print(f"Loading vessel mask: {vessel_path}")

    vessel_data, affine, header = load_nifti(vessel_path)

    # v2 labels: 3 = lung_arteries, 4 = lung_veins
    vessel_mask = ((vessel_data == 3) | (vessel_data == 4)).astype(np.uint8)

    print(f"Preprocessing vessel mask for {patient_id}...")
    cleaned_mask, stats = clean_vessel_mask(
        vessel_mask,
        min_component_size=min_component_size,
        closing_iterations=closing_iterations,
        dilation_iterations=dilation_iterations,
    )

    if save_output:
        output_path = f"{output_dir}/{patient_id}_vessels_cleaned.nii.gz"
        save_nifti(cleaned_mask, affine, header, output_path)
        print(f"Saved: {output_path}")
        stats['output_path'] = output_path

    return cleaned_mask, affine, header, stats


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {os.path.basename(__file__)} <STUDY_ID>")
    patient_id = sys.argv[1]
    cleaned, affine, header, stats = preprocess_for_patient(
        patient_id,
        min_component_size=100,
        closing_iterations=1,
    )
    print(f"\nPreprocessing complete. Stats: {stats}")
